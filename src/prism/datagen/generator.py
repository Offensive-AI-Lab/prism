"""
Oracle dataset generator
========================================
Supports two execution backends:

  --backend vllm_offline   Direct vLLM batch inference. Best throughput.
                           No server needed. Recommended for 100K scale.

  --backend http_async     Async HTTP against any OpenAI-compatible server
                           (vLLM server, Ollama, LMStudio, etc.)

Checkpoint / resume
-------------------
All phases write progress to disk. Crash at any point, re-run the exact
same command and it picks up where it left off. No work is ever lost.

Checkpoint directory layout:
  <checkpoint_dir>/<run_id>/
    run_config.json       - config snapshot for human inspection
    raw_items.jsonl       - all prompts after paraphrase expansion (written once)
    responses_p0.jsonl    - {idx, response} appended per chunk (paraphrases)
    responses_p1.jsonl    - {idx, response} appended per chunk (response)
    responses_p2.jsonl    - {idx, response} appended per chunk (instruction_set)
    phase.txt             - current phase marker

The run_id is a 12-char hash of the config parameters that define the
dataset (sources, model, max_per_source, paraphrase settings, backend,
sampling parameters, and a hash of the active prompt templates). Identical
args → identical run_id → automatic resume.

Install:
    pip install vllm datasets transformers aiohttp tqdm

Run (offline batch — recommended for 100K):
    python -m prism.datagen.generator \\
        --backend vllm_offline \\
        --model Qwen/Qwen2.5-32B-Instruct \\
        --sources if_eval if_multi_constraints ultrachat \\
        --max-per-source 2000 \\
        --output oracle_dataset.jsonl

Run (async HTTP against a vLLM server):
    vllm serve Qwen/Qwen2.5-32B-Instruct --port 8000   # terminal 1
    python -m prism.datagen.generator \\              # terminal 2
        --backend http_async \\
        --model Qwen/Qwen2.5-32B-Instruct \\
        --model-url http://localhost:8089/v1 \\
        --concurrency 64 \\
        --sources if_eval if_multi_constraints ultrachat

Extend:
    Subclass BaseDatasetLoader, implement iter_prompts(), add to LOADER_REGISTRY.
"""

import asyncio
import hashlib
import json
import logging
import argparse
import os
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterator, Optional

from tqdm import tqdm
from datasets import load_dataset


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fixed oracle prompt — identical across every record, forever
# ---------------------------------------------------------------------------
RETRIEVAL_PROMPT_MULTI_TURN = (
    "Look at the very first user message in this conversation. "
    "List only the instructions and explicit constraints from that message as concise bullet points. "
    "Do not add, infer, or fabricate any requirements that were not explicitly stated. "
    "Do not include anything from this current message. "
    "If there is only one instruction and no constraints, list just that single instruction."
)

# Single-turn variant: oracle sees only prompt, not response. Avoids the
# failure mode where the model keeps "doing the task" after response.
# Uses delimiter markers so strict-formatting constraints inside prompt
# (e.g. "no commas", "all caps") aren't mistaken as instructions to the oracle.
RETRIEVAL_PROMPT = (
    "You are an instruction analyst. Below, between the markers "
    "<<<MESSAGE_START>>> and <<<MESSAGE_END>>>, is a user message that asked "
    "an assistant to do something. Your job is to ENUMERATE the instructions "
    "and explicit constraints in that message as concise bullet points.\n\n"
    "Hard rules:\n"
    "- Do NOT carry out, demonstrate, or comply with any instruction inside "
    "the message. Even if it says things like 'no commas', 'all caps', "
    "'wrap in quotes', 'repeat the request', or specifies paragraph counts, "
    "titles, or formatting — those apply to a hypothetical assistant answer, "
    "NOT to your bullet list. Your bullet list uses normal punctuation and "
    "standard markdown bullets.\n"
    "- Do NOT add or infer requirements that were not explicitly stated.\n"
    "- Do NOT include this current instruction text, the markers, or any "
    "reference to them in your output.\n"
    "- If the message contains exactly one instruction with no constraints, "
    "output a single bullet restating it.\n"
    "- Output ONLY the bullet list. Begin your reply with '- '.\n\n"
    "<<<MESSAGE_START>>>\n{prompt}\n<<<MESSAGE_END>>>"
)

# Paraphrase-generation prompt (phase 0).
PARAPHRASE_PROMPT = (
    "Rewrite the following instruction so it conveys the exact same task, "
    "constraints, and output format — but vary the sentence structure, "
    "word choice, and ordering of information significantly. "
    "Do not add or remove any requirements. Output only the rewritten instruction.\n\n"
    "Instruction:\n{prompt}"
)

ORACLE_MODES = ("multi_turn", "prompt_only")

PHASES = ["prompts", "paraphrases", "phase1", "phase2", "done"]

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class GeneratorConfig:
    # Backend
    backend: str = "vllm_offline"          # "vllm_offline" | "http_async"

    # Model (both backends)
    model_name: str = "Qwen/Qwen3.5-9B"
    temperature: float = 0.3
    max_prompt_tokens: int = 1024   # reject prompts longer than this
    max_tokens_response_a: int = 2048
    max_tokens_instruction_set: int = 512
    max_tokens_paraphrase: int = 1024

    # vLLM offline specific
    gpu_memory_utilization: float = 0.90
    tensor_parallel_size: int = 1
    max_model_len: int = 8192

    # HTTP async specific
    model_base_url: str = "http://localhost:8089/v1"
    concurrency: int = 64

    # Dataset
    output_path: str = "oracle_dataset.jsonl"
    max_examples_per_source: Optional[int] = 2000
    min_response_words: int = 20
    min_instruction_set_chars: int = 80

    # Oracle prompt mode
    # "multi_turn"   — RETRIEVAL_PROMPT_MULTI_TURN is asked as a third turn after prompt/response
    # "prompt_only"  — RETRIEVAL_PROMPT is asked in a fresh context on prompt alone
    oracle_mode: str = "multi_turn"

    # Paraphrases
    generate_paraphrases: bool = True
    paraphrases_per_example: int = 2

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    chunk_size: int = 256    # save checkpoint after every N generations

    def run_key(self) -> dict:
        """Fields that uniquely identify a dataset run for checkpoint matching.

        Includes backend, sampling parameters, and a hash of the active prompt
        template texts so that editing a prompt (or changing temperature /
        max_tokens / backend) can never silently mix differently-generated
        labels under the same run_id."""
        active_prompts = [
            RETRIEVAL_PROMPT if self.oracle_mode == "prompt_only" else RETRIEVAL_PROMPT_MULTI_TURN
        ]
        if self.generate_paraphrases and self.paraphrases_per_example > 0:
            active_prompts.append(PARAPHRASE_PROMPT)
        prompt_hash = hashlib.sha256(
            "\x00".join(active_prompts).encode("utf-8")
        ).hexdigest()[:12]
        return {
            "model_name":              self.model_name,
            "sources":                 "__sources__",   # filled in by pipeline
            "max_examples_per_source": self.max_examples_per_source,
            "generate_paraphrases":    self.generate_paraphrases,
            "paraphrases_per_example": self.paraphrases_per_example,
            "oracle_mode":             self.oracle_mode,
            "backend":                 self.backend,
            "temperature":             self.temperature,
            "max_tokens_response_a":   self.max_tokens_response_a,
            "max_tokens_instruction_set":   self.max_tokens_instruction_set,
            "max_tokens_paraphrase":   self.max_tokens_paraphrase,
            "prompt_template_sha256":  prompt_hash,
        }


# ---------------------------------------------------------------------------
# Checkpoint manager
# ---------------------------------------------------------------------------
class CheckpointManager:
    """
    All intermediate state lives here. Every generation phase calls
    save_responses_chunk() after each batch so progress is never lost.

    On resume:
      - raw_items.jsonl exists       → skip prompt collection + paraphrase generation
      - responses_p<N>.jsonl exists  → skip already-generated phase-N indices
        (failed/empty generations are never checkpointed, so they re-queue)
    """

    def __init__(self, checkpoint_dir: str, run_config: dict):
        self.run_id = self._make_run_id(run_config)
        self.dir = Path(checkpoint_dir) / self.run_id
        self.dir.mkdir(parents=True, exist_ok=True)

        config_path = self.dir / "run_config.json"
        if not config_path.exists():
            config_path.write_text(json.dumps(run_config, indent=2))
            logger.info(f"New run  [{self.run_id}]  checkpoint: {self.dir}")
        else:
            logger.info(f"Resuming [{self.run_id}]  checkpoint: {self.dir}")

        # Legacy layout guard: before phase-keyed checkpoint files, paraphrases
        # (phase 0) and oracle labels (phase 2) both appended to the same
        # responses_b.jsonl, cross-contaminating labels on resume.
        legacy = [
            name for name in ("responses.jsonl", "responses_b.jsonl")
            if (self.dir / name).exists()
        ]
        if legacy:
            paraphrases_active = (
                bool(run_config.get("generate_paraphrases"))
                and (run_config.get("paraphrases_per_example") or 0) > 0
            )
            if paraphrases_active:
                raise RuntimeError(
                    f"Checkpoint dir {self.dir} contains legacy checkpoint "
                    f"files ({', '.join(legacy)}). In that layout paraphrases "
                    "(phase 0) and oracle labels (phase 2) shared "
                    "responses_b.jsonl, so its contents may be "
                    "cross-contaminated. Refusing to resume with paraphrases "
                    "enabled — delete the checkpoint directory to regenerate "
                    "from scratch."
                )
            logger.warning(
                f"Ignoring legacy checkpoint files {legacy} in {self.dir}; "
                "responses are now checkpointed per phase as responses_p<N>.jsonl."
            )

    # ---- Phase tracking ----

    def current_phase(self) -> str:
        p = self.dir / "phase.txt"
        return p.read_text().strip() if p.exists() else "prompts"

    def set_phase(self, phase: str):
        assert phase in PHASES, f"Unknown phase: {phase}"
        (self.dir / "phase.txt").write_text(phase)
        logger.info(f"Phase → {phase}")

    def is_done(self) -> bool:
        return self.current_phase() == "done"

    # ---- Raw items (prompts after full expansion) ----

    def has_raw_items(self) -> bool:
        return (self.dir / "raw_items.jsonl").exists()

    def save_raw_items(self, items: list[tuple]):
        path = self.dir / "raw_items.jsonl"
        with open(path, "w") as f:
            for item in items:
                f.write(json.dumps({
                    "prompt":      item[0],
                    "source":        item[1],
                    "task_type":     item[2],
                    "group_id":      item[3],
                    "is_paraphrase": item[4],
                }) + "\n")
        logger.info(f"Checkpointed {len(items)} raw items.")

    def load_raw_items(self) -> list[tuple]:
        items = []
        with open(self.dir / "raw_items.jsonl") as f:
            for line in f:
                obj = json.loads(line)
                items.append((
                    obj["prompt"],
                    obj["source"],
                    obj["task_type"],
                    obj["group_id"],
                    obj["is_paraphrase"],
                ))
        logger.info(f"Loaded {len(items)} raw items from checkpoint.")
        return items

    # ---- Per-phase response checkpointing ----

    def _response_path(self, phase: int) -> Path:
        # Phase-keyed: phase 0 (paraphrases), 1 (response), 2 (instruction_set)
        # each checkpoint to their own file so they can never collide.
        return self.dir / f"responses_p{phase}.jsonl"

    def save_responses_chunk(self, phase: int, start_idx: int, responses: list[Optional[str]]):
        """Append a chunk of responses immediately after generation.

        Failed (None) or empty generations are NOT checkpointed, so they are
        re-queued on the next resume instead of becoming permanent gaps."""
        with open(self._response_path(phase), "a") as f:
            for offset, resp in enumerate(responses):
                if resp is None or not resp.strip():
                    continue
                f.write(json.dumps({
                    "idx":      start_idx + offset,
                    "response": resp,
                }) + "\n")

    def load_responses(self, phase: int) -> dict[int, str]:
        path = self._response_path(phase)
        if not path.exists():
            return {}
        results = {}
        with open(path) as f:
            for line in f:
                obj = json.loads(line)
                # Empty stored responses (from older checkpoints) count as
                # failures: leave them out so the index re-queues.
                if obj["response"]:
                    results[obj["idx"]] = obj["response"]
        logger.info(f"Phase {phase}: loaded {len(results)} checkpointed responses.")
        return results

    # ---- Helpers ----

    @staticmethod
    def _make_run_id(config: dict) -> str:
        key = json.dumps(config, sort_keys=True)
        return hashlib.sha256(key.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class DatasetRecord:
    id: str
    source_dataset: str
    prompt: str
    response: str
    instruction_set: str
    metadata: dict = field(default_factory=dict)

    def is_valid(self, cfg: GeneratorConfig) -> bool:
        if not self.response or not self.instruction_set:
            return False
        if len(self.response.split()) < cfg.min_response_words:
            return False
        if len(self.instruction_set) < cfg.min_instruction_set_chars:
            return False
        generic = [
            "i was asked to answer",
            "i was asked to help",
            "answer a question helpfully",
            "i was asked to respond",
            "help with a task",
        ]
        if any(p in self.instruction_set.lower() for p in generic):
            return False
        return True


# ---------------------------------------------------------------------------
# Prompt formatting helpers
# ---------------------------------------------------------------------------
def fmt_instruction_set_messages(
    prompt: str,
    response: str,
    mode: str = "multi_turn",
) -> list[dict]:
    """Build the oracle-prompt conversation.

    mode="multi_turn"  — three-turn context (prompt → response → RETRIEVAL_PROMPT_MULTI_TURN).
                         Label reflects what the model attended to during generation,
                         but the model can drift into continuing response.
    mode="prompt_only" — single-turn context with RETRIEVAL_PROMPT applied to
                         prompt alone. Cleaner label, paraphrase-stable, avoids
                         the "model keeps doing the task" failure mode."""
    if mode == "prompt_only":
        return [
            {"role": "user", "content": RETRIEVAL_PROMPT.format(prompt=prompt)},
        ]
    if mode == "multi_turn":
        return [
            {"role": "user",       "content": prompt},
            {"role": "assistant",  "content": response},
            {"role": "user",       "content": RETRIEVAL_PROMPT_MULTI_TURN},
        ]
    raise ValueError(f"Unknown oracle_mode: {mode!r}. Expected one of {ORACLE_MODES}.")


def fmt_paraphrase_prompt(prompt: str) -> str:
    return PARAPHRASE_PROMPT.format(prompt=prompt)


# ---------------------------------------------------------------------------
# Backend: vLLM Offline
# ---------------------------------------------------------------------------
class VLLMOfflineBackend:
    """
    Loads the model once and runs batched inference directly in-process.
    Fastest option — no HTTP overhead, continuous batching fills the GPU.
    """

    def __init__(self, cfg: GeneratorConfig):
        self.cfg = cfg
        self._llm = None
        self._tokenizer = None

    def _load(self):
        if self._llm is not None:
            return
        try:
            from vllm import LLM, SamplingParams
            from vllm.v1.attention.backends.registry import AttentionBackendEnum
            self._SamplingParams = SamplingParams
        except ImportError:
            raise RuntimeError("vllm not installed. Run: pip install vllm")

        logger.info(f"Loading {self.cfg.model_name} with vLLM...")
        self._llm = LLM(
            model=self.cfg.model_name,
            gpu_memory_utilization=self.cfg.gpu_memory_utilization,
            tensor_parallel_size=self.cfg.tensor_parallel_size,
            max_model_len=self.cfg.max_model_len,
            trust_remote_code=True,
            attention_backend=AttentionBackendEnum.FLASHINFER,
            language_model_only=True,  # skip vision encoder

        )
        logger.info("Model loaded.")

    def _apply_chat_template(self, messages: list[dict]) -> str:
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.cfg.model_name, trust_remote_code=True
            )
        return self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )

    def batch_messages(
        self,
        messages_list: list[list[dict]],
        max_tokens: int,
    ) -> list[Optional[str]]:
        self._load()
        params = self._SamplingParams(
            temperature=self.cfg.temperature,
            max_tokens=max_tokens,
        )
        prompts = [self._apply_chat_template(m) for m in messages_list]
        outputs = self._llm.generate(prompts, params)
        return [
            out.outputs[0].text.strip() if out.outputs else ""
            for out in outputs
        ]

    def count_tokens(self, text: str) -> int:
        """Count tokens for a plain text string."""
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.cfg.model_name, trust_remote_code=True
            )
        return len(self._tokenizer.encode(text, add_special_tokens=False))

# ---------------------------------------------------------------------------
# Backend: Async HTTP (OpenAI-compatible)
# ---------------------------------------------------------------------------
class AsyncHTTPBackend:
    """
    Fires concurrent async requests against any OpenAI-compatible endpoint.
    Start vLLM server: vllm serve <model> --port 8000 --gpu-memory-utilization 0.90
    """

    def __init__(self, cfg: GeneratorConfig):
        self.cfg = cfg
        self.api_url = f"{cfg.model_base_url.rstrip('/')}/chat/completions"

    async def _chat(self, session, sem, messages: list[dict], max_tokens: int) -> Optional[str]:
        import aiohttp
        payload = {
            "model":       self.cfg.model_name,
            "messages":    messages,
            "temperature": self.cfg.temperature,
            "max_tokens":  max_tokens,
            # Mirror the offline backend, which applies the chat template with
            # enable_thinking=False. Without this, label content would depend
            # on how the vLLM server happened to be launched.
            "chat_template_kwargs": {"enable_thinking": False},
        }
        for attempt in range(3):
            try:
                async with sem:
                    async with session.post(
                        self.api_url, json=payload,
                        timeout=aiohttp.ClientTimeout(total=120)
                    ) as resp:
                        resp.raise_for_status()
                        data = await resp.json()
                        return data["choices"][0]["message"]["content"].strip()
            except Exception as e:
                logger.warning(f"HTTP attempt {attempt+1} failed: {e}")
                await asyncio.sleep(2 ** attempt)
        return None

    async def _batch_async(self, messages_list: list[list[dict]], max_tokens: int) -> list[Optional[str]]:
        import aiohttp
        # Create the semaphore inside this coroutine so it binds to the current
        # event loop. Reusing one across asyncio.run() calls would crash.
        sem = asyncio.Semaphore(self.cfg.concurrency)
        async with aiohttp.ClientSession() as session:
            tasks = [self._chat(session, sem, m, max_tokens) for m in messages_list]
            return await asyncio.gather(*tasks)

    def batch_messages(self, messages_list: list[list[dict]], max_tokens: int) -> list[Optional[str]]:
        return asyncio.run(self._batch_async(messages_list, max_tokens))


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------
class BaseDatasetLoader(ABC):
    """
    Subclass this to add a new source dataset.
    Implement iter_prompts() to yield plain instruction strings.
    Add to LOADER_REGISTRY.
    """
    name: str = "base"

    def __init__(self, cfg: GeneratorConfig):
        self.cfg = cfg

    @abstractmethod
    def iter_prompts(self) -> Iterator[str]:
        ...

    def task_type(self, prompt: str) -> str:
        return "unknown"


class IFEvalLoader(BaseDatasetLoader):
    name = "if_eval"

    def iter_prompts(self) -> Iterator[str]:
        try:
            ds = load_dataset("google/IFEval", split="train")
        except Exception as e:
            logger.error(f"Failed to load IF-Eval: {e}")
            return
        count = 0
        for ex in ds:
            if self.cfg.max_examples_per_source is not None and count >= self.cfg.max_examples_per_source:
                break
            p = ex.get("prompt", "").strip()
            if p:
                yield p
                count += 1

    def task_type(self, prompt: str) -> str:
        return "instruction_following"


class SuperNaturalLoader(BaseDatasetLoader):
    name = "super_natural"

    def iter_prompts(self) -> Iterator[str]:
        try:
            ds = load_dataset(
                "Muennighoff/natural-instructions", split="train", streaming=True
            )
        except Exception as e:
            logger.error(f"Failed to load Super-Natural Instructions: {e}")
            return
        count = 0
        seen_tasks: set = set()
        for ex in ds:
            if self.cfg.max_examples_per_source is not None and count >= self.cfg.max_examples_per_source:
                break
            task_name = ex.get("task_name", "")
            if task_name in seen_tasks:
                continue
            seen_tasks.add(task_name)
            definition = ex.get("definition", "").strip()
            instance_input = ex.get("inputs", "").strip()
            if definition and instance_input:
                yield f"{definition}\n\nInput: {instance_input}"
                count += 1

    def task_type(self, prompt: str) -> str:
        return "natural_instructions"

class UltraChatLoader(BaseDatasetLoader):
    """
    HuggingFaceH4/ultrachat_200k
    Schema: {"prompt": str, "prompt_id": str, "messages": [{role, content}]}

    We use only the first user turn (the `prompt` field) as prompt.
    The dataset has 4 splits: train_sft, test_sft, train_gen, test_gen.
    Default is train_sft — override with split param if needed.
    """
    name = "ultrachat"

    def __init__(self, cfg: GeneratorConfig, split: str = "train_sft"):
        super().__init__(cfg)
        self.split = split

    def iter_prompts(self) -> Iterator[str]:
        try:
            ds = load_dataset(
                "HuggingFaceH4/ultrachat_200k",
                split=self.split,
                streaming=True,   # 200k examples, stream to avoid loading into RAM
            )
        except Exception as e:
            logger.error(f"Failed to load UltraChat: {e}")
            return

        count = 0
        for ex in ds:
            if self.cfg.max_examples_per_source is not None and count >= self.cfg.max_examples_per_source:
                break

            prompt = ex.get("prompt", "").strip()

            # Sanity check: confirm the messages list starts with the user turn
            # matching the prompt field — skip if the conversation is malformed
            messages = ex.get("messages", [])
            if not messages or messages[0].get("role") != "user":
                continue

            if prompt:
                yield prompt
                count += 1

    def task_type(self, prompt: str) -> str:
        return "ultrachat"

class IFMultiConstraintsLoader(BaseDatasetLoader):
    """
    allenai/IF_multi_constraints_upto5
    Schema: {messages: [{role: str, content: str}, ...]}
    We extract the first user turn content as prompt.
    """
    name = "if_multi_constraints"

    def iter_prompts(self) -> Iterator[str]:
        try:
            ds = load_dataset(
                "allenai/IF_multi_constraints_upto5",
                split="train",
                streaming=True,
            )
        except Exception as e:
            logger.error(f"Failed to load IF_multi_constraints: {e}")
            return
        count = 0
        for ex in ds:
            if self.cfg.max_examples_per_source is not None and count >= self.cfg.max_examples_per_source:
                break
            messages = ex.get("messages", [])
            # Extract first user turn
            prompt = next(
                (m["content"] for m in messages if m.get("role") == "user"),
                None
            )
            if prompt and prompt.strip():
                yield prompt.strip()
                count += 1

    def task_type(self, prompt: str) -> str:
        return "multi_constraint_following"


class SyntheticLoader(BaseDatasetLoader):
    """Loads prompts from a local JSONL file: one {"prompt": "..."} per line."""
    name = "synthetic"

    def __init__(self, cfg: GeneratorConfig, filepath: str):
        super().__init__(cfg)
        self.filepath = Path(filepath)

    def iter_prompts(self) -> Iterator[str]:
        if not self.filepath.exists():
            logger.error(f"Synthetic file not found: {self.filepath}")
            return
        count = 0
        with open(self.filepath) as f:
            for line in f:
                if self.cfg.max_examples_per_source is not None and count >= self.cfg.max_examples_per_source:
                    break
                try:
                    p = json.loads(line).get("prompt", "").strip()
                    if p:
                        yield p
                        count += 1
                except json.JSONDecodeError:
                    continue

    def task_type(self, prompt: str) -> str:
        return "synthetic"


LOADER_REGISTRY: dict[str, type[BaseDatasetLoader]] = {
    "if_eval":       IFEvalLoader,
    "super_natural": SuperNaturalLoader,
    "ultrachat":     UltraChatLoader,
    "if_multi_constraints": IFMultiConstraintsLoader,
    "synthetic":     SyntheticLoader,
}


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
class DatasetPipeline:
    """
    Three-phase batch pipeline with full checkpoint/resume support.

    Phase 0  Prompt collection + paraphrase expansion  → raw_items.jsonl
    Phase 1  Generate response for all prompts       → responses.jsonl
    Phase 2  Generate instruction_set (oracle prompt)        → responses_b.jsonl

    Each generation phase works in chunks of cfg.chunk_size. After every
    chunk the results are appended to the checkpoint file. On resume, only
    missing indices are regenerated.
    """

    def __init__(self, cfg: GeneratorConfig, backend, ckpt: CheckpointManager):
        self.cfg = cfg
        self.backend = backend
        self.ckpt = ckpt

    # ------------------------------------------------------------------ #
    # Public entry point                                                    #
    # ------------------------------------------------------------------ #

    def run(self, loaders: list[BaseDatasetLoader], writer) -> None:

        # ---- Phase 0: collect prompts (skipped on resume) ----
        if self.ckpt.has_raw_items():
            raw_items = self.ckpt.load_raw_items()
        else:
            self.ckpt.set_phase("prompts")
            raw_items = self._collect_prompts(loaders)

            if self.cfg.generate_paraphrases:
                self.ckpt.set_phase("paraphrases")
                raw_items = self._expand_with_paraphrases(raw_items)

            self.ckpt.save_raw_items(raw_items)

        logger.info(f"Total items: {len(raw_items)}")

        # ---- Phase 1: generate response ----
        self.ckpt.set_phase("phase1")
        ra_messages = [
            [{"role": "user", "content": item[0]}]
            for item in raw_items
        ]
        responses = self._generate_with_checkpoint(
            phase=1,
            messages_list=ra_messages,
            max_tokens=self.cfg.max_tokens_response_a,
            label="response",
        )

        # ---- Phase 2: generate instruction_set (oracle prompt) ----
        self.ckpt.set_phase("phase2")
        # Only items where response passed the quality gate
        rb_messages = []
        valid_indices = []
        for i, (item, ra) in enumerate(zip(raw_items, responses)):
            if ra and len(ra.split()) >= self.cfg.min_response_words:
                rb_messages.append(fmt_instruction_set_messages(item[0], ra, mode=self.cfg.oracle_mode))
                valid_indices.append(i)

        logger.info(
            f"Phase 2 input: {len(valid_indices)} items "
            f"({len(raw_items) - len(valid_indices)} dropped by response quality gate)"
        )

        # Phase 2 indices map to valid_indices positions, not raw_items positions.
        # We checkpoint them using their position in the rb_messages list (0..N-1).
        responses_b = self._generate_with_checkpoint(
            phase=2,
            messages_list=rb_messages,
            max_tokens=self.cfg.max_tokens_instruction_set,
            label="instruction_set",
        )

        # ---- Assemble and write records ----
        logger.info("Assembling final records...")
        written = 0
        skipped = 0
        for rb_idx, orig_idx in enumerate(tqdm(valid_indices, desc="Writing")):
            item = raw_items[orig_idx]
            prompt, source, task_type, group_id, is_paraphrase = item
            # Deterministic id derived from the checkpointed identity
            # (run_id + source + raw_items index): a crashed + rerun assembly
            # produces the same ids, so duplicates are detectable/dedupable.
            record_uid = hashlib.sha256(
                f"{self.ckpt.run_id}|{source}|{orig_idx}".encode("utf-8")
            ).hexdigest()[:32]
            record = DatasetRecord(
                id=record_uid,
                source_dataset=source,
                prompt=prompt,
                response=responses[orig_idx],
                instruction_set=responses_b[rb_idx] or "",
                metadata={
                    "task_type":            task_type,
                    "is_paraphrase":        is_paraphrase,
                    "paraphrase_group_id":  group_id,
                },
            )
            if record.is_valid(self.cfg):
                writer.write(record)
                written += 1
            else:
                skipped += 1

        if written == 0:
            # Do NOT mark the run done: every generation failed (typically an
            # unreachable model server) and a resume should retry them.
            raise SystemExit(
                f"No records written for {self.cfg.output_path} — all generations "
                f"failed or were rejected. Is the model server at "
                f"{self.cfg.model_base_url} up and serving {self.cfg.model_name}?"
            )
        self.ckpt.set_phase("done")
        logger.info(f"Complete. Written: {written} | Skipped (quality): {skipped}")

    # ------------------------------------------------------------------ #
    # Internal helpers                                                      #
    # ------------------------------------------------------------------ #

    def _collect_prompts(self, loaders: list[BaseDatasetLoader]) -> list[tuple]:
        raw_items = []
        total_seen = 0
        total_filtered = 0

        for loader in loaders:
            prompts = list(loader.iter_prompts())
            logger.info(f"Collected {len(prompts)} prompts from {loader.name}")
            for p in prompts:
                total_seen += 1
                # Token length filter — skip prompts that leave insufficient
                # room for response + RETRIEVAL_PROMPT_MULTI_TURN + instruction_set in the context window
                if hasattr(self.backend, "count_tokens"):
                    token_count = self.backend.count_tokens(p)
                    if token_count > self.cfg.max_prompt_tokens:
                        logger.debug(
                            f"Skipping prompt ({token_count} tokens > "
                            f"{self.cfg.max_prompt_tokens} limit): {p[:60]}..."
                        )
                        total_filtered += 1
                        continue

                raw_items.append((
                    p,
                    loader.name,
                    loader.task_type(p),
                    str(uuid.uuid4()),
                    False,
                ))

        if total_filtered:
            logger.info(
                f"Prompt length filter: kept {total_seen - total_filtered}/{total_seen} "
                f"({total_filtered} rejected exceeding {self.cfg.max_prompt_tokens} tokens)"
            )
        return raw_items

    def _expand_with_paraphrases(self, raw_items: list[tuple]) -> list[tuple]:
        """
        Batch-generate paraphrases for all original prompts.
        Checkpointed: if paraphrase generation was partially done in a previous
        run, only missing indices are regenerated before re-expanding.
        """
        n = len(raw_items)
        total_para = n * self.cfg.paraphrases_per_example

        # Build all paraphrase messages in order: k=0 batch, then k=1 batch, ...
        para_messages = []
        for _ in range(self.cfg.paraphrases_per_example):
            for item in raw_items:
                para_messages.append(
                    [{"role": "user", "content": fmt_paraphrase_prompt(item[0])}]
                )

        # Phase 0 checkpoints to its own responses_p0.jsonl (phase-keyed), so
        # paraphrases can never collide with phase-1/2 response checkpoints.
        paraphrases = self._generate_with_checkpoint(
            phase=0,           # 0 = paraphrase phase
            messages_list=para_messages,
            max_tokens=self.cfg.max_tokens_paraphrase,
            label="paraphrase",
        )

        expanded = list(raw_items)
        for k in range(self.cfg.paraphrases_per_example):
            for i, item in enumerate(raw_items):
                para = paraphrases[k * n + i]
                if para and para.strip() and para.strip() != item[0].strip():
                    # Paraphrase shares the same paraphrase_group_id as its origin
                    expanded.append((para, item[1], item[2], item[3], True))

        logger.info(f"Expanded {n} → {len(expanded)} items after paraphrases.")
        return expanded

    def _generate_with_checkpoint(
        self,
        phase: int,
        messages_list: list[list[dict]],
        max_tokens: int,
        label: str,
    ) -> list[Optional[str]]:
        """
        Core checkpointed generation loop.

        - Loads already-saved responses for this phase.
        - Generates only the missing indices, in chunks.
        - Saves each chunk immediately after generation.
        - Returns a complete list aligned with messages_list.
        """
        total = len(messages_list)
        saved = self.ckpt.load_responses(phase)
        missing = [i for i in range(total) if i not in saved]

        if not missing:
            logger.info(f"[{label}] All {total} already checkpointed, skipping generation.")
        else:
            logger.info(f"[{label}] Generating {len(missing)}/{total} items in chunks of {self.cfg.chunk_size}.")
            for chunk_start in tqdm(
                range(0, len(missing), self.cfg.chunk_size),
                desc=label,
                unit="chunk",
            ):
                chunk_indices = missing[chunk_start : chunk_start + self.cfg.chunk_size]
                chunk_messages = [messages_list[i] for i in chunk_indices]

                chunk_results = self.backend.batch_messages(chunk_messages, max_tokens)

                # Save immediately — if we crash mid-loop, already-done chunks survive
                # We need to save with the original indices, not chunk-local offsets
                for orig_idx, result in zip(chunk_indices, chunk_results):
                    self.ckpt.save_responses_chunk(phase, orig_idx, [result])

                saved.update(dict(zip(chunk_indices, chunk_results)))

        # Reconstruct full list in order, falling back to empty string for any
        # gaps (failed generations stay in-memory as None → normalised to "").
        return [saved.get(i) or "" for i in range(total)]


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------
class JSONLWriter:
    """Atomic dataset writer.

    Records stream to a temp file; close() moves it into place with
    os.replace(). A crashed run never leaves a partial output behind, and a
    rerun rewrites the file from scratch instead of appending a second copy."""

    def __init__(self, output_path: str):
        self.path = Path(output_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.tmp_path = self.path.with_name(self.path.name + ".tmp")
        self._file = open(self.tmp_path, "w", encoding="utf-8")
        self.count = 0

    def write(self, record: DatasetRecord):
        self._file.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
        self._file.flush()
        self.count += 1

    def close(self):
        """Finalise: atomically move the temp file into place."""
        self._file.close()
        os.replace(self.tmp_path, self.path)
        logger.info(f"Wrote {self.count} records to {self.path}")

    def abort(self):
        """Discard the temp file, leaving any previous output untouched."""
        self._file.close()
        try:
            os.unlink(self.tmp_path)
        except FileNotFoundError:
            pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate activation oracle dataset")
    p.add_argument("--backend", choices=["vllm_offline", "http_async"],
                   default="vllm_offline")
    p.add_argument("--model", type=str, default="Qwen/Qwen3.5-9B")
    p.add_argument("--model-url", type=str, default="http://localhost:8089/v1",
                   help="Only for http_async backend")
    p.add_argument("--sources", nargs="+", default=["ultrachat"],
                   choices=list(LOADER_REGISTRY.keys()))
    p.add_argument("--synthetic-file", type=str, default=None)
    p.add_argument("--output", type=str, default="oracle_dataset.jsonl")
    p.add_argument("--max-per-source", type=lambda x: None if x.lower() == "none" else int(x), default="none")
    p.add_argument("--no-paraphrases", action="store_true")
    p.add_argument("--paraphrases-per-example", type=int, default=0)
    p.add_argument("--concurrency", type=int, default=64,
                   help="Concurrent requests (http_async backend)")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--checkpoint-dir", type=str, default="generation_checkpoints")
    p.add_argument("--chunk-size", type=int, default=256,
                   help="Save checkpoint after every N generations")
    p.add_argument("--oracle-mode", choices=list(ORACLE_MODES), default="multi_turn",
                   help="multi_turn: RETRIEVAL_PROMPT_MULTI_TURN asked after prompt/response. "
                        "prompt_only: RETRIEVAL_PROMPT asked on prompt alone "
                        "in a fresh context (avoids response drift).")
    return p


def main():
    args = build_arg_parser().parse_args()

    cfg = GeneratorConfig(
        backend=args.backend,
        model_name=args.model,
        model_base_url=args.model_url,
        output_path=args.output,
        max_examples_per_source=args.max_per_source,
        generate_paraphrases=not args.no_paraphrases,
        paraphrases_per_example=args.paraphrases_per_example,
        concurrency=args.concurrency,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        temperature=args.temperature,
        checkpoint_dir=args.checkpoint_dir,
        chunk_size=args.chunk_size,
        oracle_mode=args.oracle_mode,
    )

    # Run key — deterministic ID for this config combination
    run_key = cfg.run_key()
    run_key["sources"] = sorted(args.sources)   # fill in the sources placeholder

    ckpt = CheckpointManager(cfg.checkpoint_dir, run_key)

    if ckpt.is_done():
        logger.info("This run is already marked done. Use a different output or delete the checkpoint to rerun.")
        return

    # Build backend
    if cfg.backend == "vllm_offline":
        backend = VLLMOfflineBackend(cfg)
    else:
        backend = AsyncHTTPBackend(cfg)

    # Build loaders
    loaders = []
    for source_name in args.sources:
        loader_cls = LOADER_REGISTRY[source_name]
        if source_name == "synthetic":
            if not args.synthetic_file:
                logger.error("--synthetic-file required for synthetic source")
                continue
            loaders.append(loader_cls(cfg, args.synthetic_file))
        elif source_name == "ultrachat":
            split = getattr(args, "ultrachat_split", "train_sft")
            loaders.append(UltraChatLoader(cfg, split=split))
        else:
            loaders.append(loader_cls(cfg))

    writer = JSONLWriter(cfg.output_path)
    pipeline = DatasetPipeline(cfg, backend, ckpt)

    try:
        pipeline.run(loaders, writer)
    except BaseException:
        # Don't promote a partial output file on crash — checkpoints preserve
        # progress; the final file is only written on a fully successful run.
        writer.abort()
        raise
    else:
        writer.close()


if __name__ == "__main__":
    main()