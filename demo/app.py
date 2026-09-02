"""Local PRISM demo: chat with the target model, then read its activations.

One page, three panels: the prompt you send, the target model's answer, and
the instruction report PRISM decodes from the answer's activations — no judge,
no cloud services, nothing leaves your machine.

Launch (from the repository root):

    uv sync --extra demo
    uv run python demo/app.py            # http://127.0.0.1:7860

Requirements: a CUDA GPU with ~24 GB of free memory (Qwen3.5-9B in bf16 plus
two LoRA adapters). On first launch the released checkpoints
(prism-qwen3.5-9b-grpo.pt and prism-qwen3.5-9b-sft.pt, ~266 MB each) are
downloaded from Hugging Face into --checkpoint-dir (default ./checkpoints)
and verified against their SHA-256 digests; the Qwen3.5-9B base model is
pulled into the normal Hugging Face cache (~18 GB).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
logger = logging.getLogger("prism.demo")

ASSETS_DIR = Path(__file__).resolve().parent

# =============================================================================
# Released checkpoints (auto-downloaded, SHA-256 verified)
# =============================================================================

WEIGHTS_ORG = os.environ.get("PRISM_DEMO_WEIGHTS_ORG", "Offensive-AI-Lab")

VARIANTS: List[Dict[str, str]] = [
    {
        "key": "sft_rl",
        "display_name": "PRISM",
        "subtitle": "Base Chat → Layer Activations → PRISM Retrieval",
        "filename": "prism-qwen3.5-9b-grpo.pt",
        "sha256": "5bde25517e11ff26130c2d842dd01ebbf7b7ed5c997ce89b949ff05aa1d7d2d1",
    },
    {
        "key": "sft",
        "display_name": "PRISM w/o RL",
        "subtitle": "Base Chat → Layer Activations → PRISM w/o RL Retrieval",
        "filename": "prism-qwen3.5-9b-sft.pt",
        "sha256": "347cc6c6674a839dd995633a057f9ddb6cb45e153f912671445eed66c6a56002",
    },
]
VARIANT_KEYS = [v["key"] for v in VARIANTS]
PRIMARY_KEY = VARIANTS[0]["key"]

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
BASE_MAX_NEW_TOKENS = 192
RETRIEVAL_MAX_NEW_TOKENS = 256
RETRIEVAL_PLACEHOLDER = "Ask PRISM about hidden behavior or persona..."
RETRIEVAL_BUTTON_LABEL = "Retrieve with PRISM"

EXAMPLE_PROMPTS: List[Dict[str, str]] = json.loads(
    (ASSETS_DIR / "example_prompts.json").read_text(encoding="utf-8")
)


def _sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def ensure_checkpoints(ckpt_dir: Path) -> Dict[str, Path]:
    """Download any missing released checkpoint and verify every digest."""
    from huggingface_hub import hf_hub_download

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    resolved: Dict[str, Path] = {}
    for v in VARIANTS:
        path = ckpt_dir / v["filename"]
        if not path.exists():
            repo_id = f"{WEIGHTS_ORG}/{v['filename'][:-3]}"
            logger.info(f"[{v['key']}] downloading {repo_id} → {path}")
            hf_hub_download(repo_id=repo_id, filename=v["filename"], local_dir=str(ckpt_dir))
        digest = _sha256_of(path)
        if digest != v["sha256"]:
            raise RuntimeError(
                f"{path} has SHA-256 {digest}, expected {v['sha256']}. "
                f"Delete the file and re-run to re-download."
            )
        logger.info(f"[{v['key']}] checkpoint verified: {path.name}")
        resolved[v["key"]] = path
    return resolved


def _preflight_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA GPU available. The demo runs Qwen3.5-9B locally and needs "
            "a CUDA device with ~24 GB of free memory. If you have a GPU, check "
            "that PyTorch was installed with CUDA support (python -c "
            "'import torch; print(torch.version.cuda)')."
        )
    free, total = torch.cuda.mem_get_info()
    free_gb, total_gb = free / 1024**3, total / 1024**3
    if free_gb < 21:
        logger.warning(
            f"Only {free_gb:.1f} GB of {total_gb:.1f} GB GPU memory is free; "
            f"the demo needs about 24 GB. Close other GPU processes if the "
            f"model fails to load."
        )


# =============================================================================
# Model plumbing (mirrors the released checkpoints' inference contract)
# =============================================================================


class ActivationProjection(nn.Module):
    def __init__(self, dim: int = 4096):
        super().__init__()
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def _apply_template(tokenizer, messages, **kwargs):
    result = tokenizer.apply_chat_template(
        messages, tokenize=True, enable_thinking=False, **kwargs
    )
    if hasattr(result, "input_ids"):
        result = result["input_ids"]
    return result


class _EarlyExit(Exception):
    pass


def _find_layers(model):
    best = None

    def _walk(module, prefix=""):
        nonlocal best
        for name, child in module._modules.items():
            if child is None:
                continue
            path = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.ModuleList) and len(child) > 1:
                types = {type(c).__name__ for c in child}
                if len(types) == 1:
                    if best is None or len(child) > len(best[1]):
                        best = (path, child)
            _walk(child, path)

    _walk(model)
    if best is None:
        raise RuntimeError("Cannot locate transformer layers in model.")
    return best[1]


class PrismRuntime:
    """One target-model instance with the released PRISM variants as named
    LoRA adapters.

    Generation flow:
      - Base chat: LoRA OFF (output identical across variants).
      - PRISM retrieval: LoRA ON with the chosen variant's adapter; project the
        extracted activations into soft tokens; decode.
    """

    def __init__(self, checkpoint_paths: Dict[str, Path]):
        self.checkpoint_paths = checkpoint_paths
        self.tokenizer = None
        self.model = None
        self.target_norm = None
        self.act_store: Optional[Dict[str, Any]] = None
        self.hook_handle = None
        self.projections: Dict[str, Optional[nn.Module]] = {}
        self.train_cfgs: Dict[str, dict] = {}
        self.adapter_name_of: Dict[str, str] = {}

    def load(self) -> None:
        from peft import LoraConfig, TaskType, get_peft_model, set_peft_model_state_dict
        from transformers import AutoProcessor, AutoModelForImageTextToText

        _preflight_cuda()
        device = torch.device("cuda")

        primary_key = PRIMARY_KEY
        primary_ckpt = torch.load(
            self.checkpoint_paths[primary_key], map_location="cpu", weights_only=False
        )
        primary_train_cfg = primary_ckpt["config"]
        self.train_cfgs[primary_key] = primary_train_cfg

        model_id = primary_train_cfg["model_id"]
        logger.info(f"[{primary_key}] checkpoint loaded; model_id={model_id}")

        processor = AutoProcessor.from_pretrained(model_id)
        self.tokenizer = processor.tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info(f"Loading base model {model_id} (~18 GB on first run) ...")
        t0 = time.time()
        try:
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                dtype=torch.bfloat16,
                device_map=device,
            )
        except torch.cuda.OutOfMemoryError as exc:
            raise RuntimeError(
                "Out of GPU memory while loading the target model — the demo "
                "needs about 24 GB free. Close other GPU processes and retry."
            ) from exc
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        logger.info(f"Base model loaded in {time.time()-t0:.1f}s")

        primary_lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=primary_train_cfg["lora_r"],
            lora_alpha=primary_train_cfg["lora_alpha"],
            lora_dropout=0.0,
            target_modules=primary_train_cfg["lora_target_modules"],
        )
        self.model = get_peft_model(self.model, primary_lora_config)
        set_peft_model_state_dict(self.model, primary_ckpt["lora_state"])
        self.model.eval()
        self.adapter_name_of[primary_key] = "default"
        logger.info(f"[{primary_key}] LoRA loaded as adapter 'default'")

        hook_layer = primary_train_cfg["hook_layer"]
        base_model = self.model.base_model.model
        layers = _find_layers(base_model)
        self.act_store = {"active": False}

        def _hook(module, inp, out):
            if not self.act_store["active"]:
                return
            hidden = out[0] if isinstance(out, tuple) else out
            self.act_store["hidden"] = hidden.detach()
            raise _EarlyExit()

        self.hook_handle = layers[hook_layer].register_forward_hook(_hook)
        logger.info(f"Hook registered on layer {hook_layer}/{len(layers)-1}")

        with torch.no_grad():
            emb_weight = self.model.get_input_embeddings().weight
            self.target_norm = emb_weight.norm(dim=1).mean().to(device)

        self._load_projection(primary_key, primary_ckpt, device)

        for variant in VARIANTS[1:]:
            key = variant["key"]
            ckpt = torch.load(self.checkpoint_paths[key], map_location="cpu", weights_only=False)
            train_cfg = ckpt["config"]
            self.train_cfgs[key] = train_cfg

            if train_cfg.get("hook_layer") != hook_layer:
                logger.warning(
                    f"[{key}] hook_layer={train_cfg.get('hook_layer')} differs from "
                    f"primary hook_layer={hook_layer}; using primary's hook."
                )

            lora_cfg = LoraConfig(
                r=train_cfg["lora_r"],
                lora_alpha=train_cfg["lora_alpha"],
                lora_dropout=0.0,
                target_modules=train_cfg["lora_target_modules"],
            )
            self.model.add_adapter(key, lora_cfg)

            raw_state = ckpt["lora_state"]
            current_keys = list(self.model.state_dict().keys())
            needs_remap = any("language_model" in k for k in current_keys)
            if needs_remap and not any("language_model" in k for k in raw_state):
                raw_state = {
                    k.replace(
                        "base_model.model.model.layers.",
                        "base_model.model.model.language_model.layers.",
                    ): v
                    for k, v in raw_state.items()
                }
                logger.info(f"[{key}] remapped {len(raw_state)} LoRA keys (.language_model prefix)")

            set_peft_model_state_dict(self.model, raw_state, adapter_name=key)
            self.adapter_name_of[key] = key
            logger.info(f"[{key}] LoRA loaded as adapter '{key}'")
            self._load_projection(key, ckpt, device)

        self.model.eval()
        self.model.set_adapter("default")
        logger.info(f"All variants loaded: {list(self.adapter_name_of.keys())}")

    def _load_projection(self, key: str, ckpt: dict, device: torch.device) -> None:
        train_cfg = self.train_cfgs[key]
        use_projection = train_cfg.get("_use_projection", True)
        proj_state = ckpt.get("projection_state")
        if use_projection and proj_state is not None:
            projection = ActivationProjection(dim=train_cfg.get("projection_dim", 4096))
            projection.load_state_dict(proj_state)
            projection.to(device, dtype=torch.bfloat16)
            projection.eval()
            self.projections[key] = projection
        else:
            self.projections[key] = None

    # ── Public API ──────────────────────────────────────────────────────

    @property
    def device(self) -> torch.device:
        return torch.device("cuda")

    def default_max_act_tokens(self, variant_key: str) -> int:
        return int(self.train_cfgs[variant_key].get("max_act_tokens", 128))

    def _extract_activations(
        self,
        user_prompt: str,
        assistant_response: str,
        system_prompt: str,
        variant_key: str,
        max_act_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        if max_act_tokens is None:
            max_act_tokens = self.default_max_act_tokens(variant_key)
        max_act_tokens = max(1, int(max_act_tokens))

        messages_prompt_only = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        messages_full = messages_prompt_only + [
            {"role": "assistant", "content": assistant_response},
        ]
        prefix_ids = _apply_template(self.tokenizer, messages_prompt_only, add_generation_prompt=True)
        full_ids = _apply_template(self.tokenizer, messages_full, add_generation_prompt=False)

        prompt_len = len(prefix_ids)
        total_len = len(full_ids)
        resp_len = total_len - prompt_len
        n_act = min(resp_len, max_act_tokens)

        self.model.disable_adapter_layers()
        input_ids = torch.tensor([full_ids], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(input_ids)

        self.act_store["active"] = True
        with torch.inference_mode():
            try:
                self.model(input_ids=input_ids, attention_mask=attention_mask)
            except _EarlyExit:
                pass
        self.act_store["active"] = False
        self.model.enable_adapter_layers()

        hidden = self.act_store["hidden"]
        if n_act > 0:
            start = total_len - n_act
            activations = hidden[0, start:total_len, :]
        else:
            activations = hidden[0, :0, :]

        assistant_ids = full_ids[prompt_len:]
        if n_act < resp_len:
            pre_ids = assistant_ids[:resp_len - n_act]
            used_ids = assistant_ids[resp_len - n_act:]
        else:
            pre_ids = []
            used_ids = assistant_ids

        used_id_list = used_ids.tolist() if hasattr(used_ids, "tolist") else list(used_ids)
        extracted_tokens = [
            {
                "text": self.tokenizer.decode([tid], skip_special_tokens=True),
                "response_index": (resp_len - n_act) + i,
            }
            for i, tid in enumerate(used_id_list)
        ]

        return {
            "activations": activations,
            "assistant_token_count": resp_len,
            "extracted_token_count": n_act,
            "truncated_tokens": resp_len - n_act,
            "highlight_prefix_text": self.tokenizer.decode(pre_ids, skip_special_tokens=True),
            "highlight_text": self.tokenizer.decode(used_ids, skip_special_tokens=True),
            "extracted_tokens": extracted_tokens,
        }

    def _build_retrieval_inputs(
        self,
        user_prompt: str,
        assistant_response: str,
        retrieval_prompt: str,
        system_prompt: str,
        variant_key: str,
        max_act_tokens: Optional[int] = None,
    ):
        extraction = self._extract_activations(
            user_prompt, assistant_response, system_prompt, variant_key, max_act_tokens,
        )
        activations = extraction["activations"].unsqueeze(0).to(dtype=torch.bfloat16)

        projection = self.projections[variant_key]
        train_cfg = self.train_cfgs[variant_key]

        with torch.no_grad():
            soft = activations
            if projection is not None:
                soft = projection(soft)
            norms = soft.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            soft = soft / norms * self.target_norm

            skip_prompt_b = train_cfg.get("skip_prompt_b", False)
            content = "" if skip_prompt_b else retrieval_prompt
            prefix_ids = _apply_template(
                self.tokenizer,
                [{"role": "user", "content": content}],
                add_generation_prompt=True,
            )
            prefix_tensor = torch.tensor([prefix_ids], dtype=torch.long, device=self.device)

            self.model.set_adapter(self.adapter_name_of[variant_key])
            prefix_emb = self.model.get_input_embeddings()(prefix_tensor)
            inputs_embeds = torch.cat([soft, prefix_emb], dim=1)
            full_mask = torch.ones(1, inputs_embeds.size(1), device=self.device, dtype=torch.long)

        return inputs_embeds, full_mask, extraction

    def generate_base_stream(
        self,
        user_prompt: str,
        system_prompt: str,
        max_new_tokens: int,
        variant_key: str,
        max_act_tokens: Optional[int] = None,
    ):
        from transformers import TextIteratorStreamer

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt_ids = _apply_template(self.tokenizer, messages, add_generation_prompt=True)
        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)

        self.model.disable_adapter_layers()
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs = {
            "input_ids": prompt_tensor,
            "attention_mask": torch.ones_like(prompt_tensor),
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "use_cache": True,
            "streamer": streamer,
        }
        thread = threading.Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()
        chunks = []
        for text in streamer:
            if text:
                chunks.append(text)
                yield {"type": "token", "text": text}
        thread.join()
        self.model.enable_adapter_layers()
        self.model.set_adapter(self.adapter_name_of[variant_key])

        response = "".join(chunks).strip()
        extraction = self._extract_activations(
            user_prompt, response, system_prompt, variant_key, max_act_tokens,
        )
        yield {
            "type": "done",
            "assistant_response": response,
            **{k: v for k, v in extraction.items() if k != "activations"},
        }

    def generate_base(self, *args, **kwargs) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for event in self.generate_base_stream(*args, **kwargs):
            if event["type"] == "done":
                result = {k: v for k, v in event.items() if k != "type"}
        return result

    def answer_retrieval_stream(
        self,
        user_prompt: str,
        assistant_response: str,
        retrieval_prompt: str,
        system_prompt: str,
        max_new_tokens: int,
        variant_key: str,
        max_act_tokens: Optional[int] = None,
    ):
        from transformers import TextIteratorStreamer

        inputs_embeds, full_mask, extraction = self._build_retrieval_inputs(
            user_prompt, assistant_response, retrieval_prompt, system_prompt, variant_key, max_act_tokens,
        )
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=False, skip_special_tokens=True)
        gen_kwargs = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": full_mask,
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.eos_token_id,
            "use_cache": True,
            "streamer": streamer,
        }
        thread = threading.Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()
        chunks = []
        for text in streamer:
            if text:
                chunks.append(text)
                yield {"type": "token", "text": text}
        thread.join()
        yield {
            "type": "done",
            "retrieval_response": "".join(chunks).strip(),
            **{k: v for k, v in extraction.items() if k != "activations"},
        }

    def answer_retrieval(self, *args, **kwargs) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for event in self.answer_retrieval_stream(*args, **kwargs):
            if event["type"] == "done":
                result = {k: v for k, v in event.items() if k != "type"}
        return result


# =============================================================================
# Runtime management
# =============================================================================

_runtime: Optional[PrismRuntime] = None
_runtime_lock = threading.Lock()
_load_state: Dict[str, str] = {"status": "idle", "message": "Waiting to start..."}
_load_state_lock = threading.Lock()
_checkpoint_dir = Path(os.environ.get("PRISM_DEMO_CHECKPOINT_DIR", "checkpoints"))


def _set_state(status: str, message: str) -> None:
    with _load_state_lock:
        _load_state["status"] = status
        _load_state["message"] = message


def get_load_state() -> Dict[str, str]:
    with _load_state_lock:
        return dict(_load_state)


def _load_runtime():
    global _runtime
    with _runtime_lock:
        if _runtime is not None:
            _set_state("ready", "")
            return
        try:
            _set_state("loading", "Downloading / verifying released checkpoints...")
            paths = ensure_checkpoints(_checkpoint_dir)
            _set_state("loading", "Loading Qwen3.5-9B and the PRISM adapters into the GPU (~1 min)...")
            rt = PrismRuntime(paths)
            rt.load()
            _runtime = rt
            _set_state("ready", "")
        except Exception as exc:
            logger.exception("Runtime load failed")
            _set_state("error", f"Model load failed: {exc}")
            raise


def _start_background_load():
    with _load_state_lock:
        if _load_state["status"] in ("loading", "ready"):
            return
        _load_state["status"] = "loading"
        _load_state["message"] = "Starting up..."
    threading.Thread(target=_load_runtime, daemon=True).start()


def get_runtime() -> PrismRuntime:
    if _runtime is None:
        _load_runtime()
    assert _runtime is not None
    return _runtime


def _resolve_variant_key(requested: Optional[str]) -> str:
    if not requested:
        return PRIMARY_KEY
    if requested in VARIANT_KEYS:
        return requested
    raise HTTPException(status_code=400, detail="Unknown variant")


# =============================================================================
# API
# =============================================================================

HARD_LIMIT_BASE_MAX_NEW_TOKENS = 2048
HARD_LIMIT_RETRIEVAL_MAX_NEW_TOKENS = 1024
HARD_LIMIT_MAX_ACT_TOKENS = 512


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(value)))


class BaseChatRequest(BaseModel):
    user_prompt: str
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    max_new_tokens: int = BASE_MAX_NEW_TOKENS
    max_act_tokens: Optional[int] = None
    mode: str = ""


class RetrievalRequest(BaseModel):
    user_prompt: str
    assistant_response: str
    retrieval_prompt: str = ""
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    max_new_tokens: int = RETRIEVAL_MAX_NEW_TOKENS
    max_act_tokens: Optional[int] = None
    mode: str = ""


@asynccontextmanager
async def lifespan(app):
    if os.environ.get("PRISM_DEMO_PRELOAD", "1") != "0":
        logger.info("Startup: loading models in the background ...")
        _start_background_load()
    yield


web_app = FastAPI(
    title="PRISM Demo",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
    lifespan=lifespan,
)


@web_app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@web_app.get("/api/status")
async def api_status() -> Dict[str, Any]:
    return {**get_load_state(), "baselines": {"status": "idle", "message": ""}}


@web_app.get("/api/config")
async def api_config() -> Dict[str, Any]:
    variant_defaults: Dict[str, Dict[str, Any]] = {}
    if _runtime is not None:
        for v in VARIANTS:
            try:
                variant_defaults[v["key"]] = {
                    "max_act_tokens": _runtime.default_max_act_tokens(v["key"]),
                }
            except Exception:
                variant_defaults[v["key"]] = {"max_act_tokens": None}

    variants = [
        {
            "key": v["key"],
            "display_name": v["display_name"],
            "subtitle": v["subtitle"],
            "defaults": variant_defaults.get(v["key"], {"max_act_tokens": None}),
        }
        for v in VARIANTS
    ]
    return {
        "primary_variant": PRIMARY_KEY,
        "variants": variants,
        "compare_available": False,
        "baselines": [],
        "retrieval_button_label": RETRIEVAL_BUTTON_LABEL,
        "retrieval_placeholder": RETRIEVAL_PLACEHOLDER,
        "show_retrieval_input": True,
        "example_prompts": EXAMPLE_PROMPTS,
        "defaults": {
            "base_max_new_tokens": BASE_MAX_NEW_TOKENS,
            "retrieval_max_new_tokens": RETRIEVAL_MAX_NEW_TOKENS,
            "max_act_tokens": 128,
        },
        "limits": {
            "base_max_new_tokens": HARD_LIMIT_BASE_MAX_NEW_TOKENS,
            "retrieval_max_new_tokens": HARD_LIMIT_RETRIEVAL_MAX_NEW_TOKENS,
            "max_act_tokens": HARD_LIMIT_MAX_ACT_TOKENS,
        },
    }


def _clamped_base_max(req_value: int) -> int:
    return _clamp(req_value, 16, HARD_LIMIT_BASE_MAX_NEW_TOKENS)


def _clamped_retrieval_max(req_value: int) -> int:
    return _clamp(req_value, 16, HARD_LIMIT_RETRIEVAL_MAX_NEW_TOKENS)


def _clamped_act(req_value: Optional[int]) -> Optional[int]:
    if req_value is None:
        return None
    return _clamp(req_value, 1, HARD_LIMIT_MAX_ACT_TOKENS)


@web_app.post("/api/base")
async def api_base(req: BaseChatRequest) -> Dict[str, Any]:
    if not req.user_prompt.strip():
        raise HTTPException(status_code=400, detail="user_prompt is required")
    variant_key = _resolve_variant_key(req.mode)
    started = time.time()
    rt = await asyncio.to_thread(get_runtime)
    try:
        result = await asyncio.to_thread(
            rt.generate_base,
            req.user_prompt,
            req.system_prompt,
            _clamped_base_max(req.max_new_tokens),
            variant_key,
            _clamped_act(req.max_act_tokens),
        )
    except Exception:
        logger.exception("Base generation failed")
        raise HTTPException(status_code=500, detail="Generation failed")
    result["latency_sec"] = round(time.time() - started, 3)
    result["active_mode"] = variant_key
    return result


@web_app.post("/api/retrieval")
async def api_retrieval(req: RetrievalRequest) -> Dict[str, Any]:
    if not req.user_prompt.strip() or not req.assistant_response.strip():
        raise HTTPException(
            status_code=400,
            detail="user_prompt and assistant_response are required",
        )
    variant_key = _resolve_variant_key(req.mode)
    started = time.time()
    rt = await asyncio.to_thread(get_runtime)
    try:
        result = await asyncio.to_thread(
            rt.answer_retrieval,
            req.user_prompt,
            req.assistant_response,
            req.retrieval_prompt,
            req.system_prompt,
            _clamped_retrieval_max(req.max_new_tokens),
            variant_key,
            _clamped_act(req.max_act_tokens),
        )
    except Exception:
        logger.exception("Retrieval failed")
        raise HTTPException(status_code=500, detail="Retrieval failed")
    result["latency_sec"] = round(time.time() - started, 3)
    result["active_mode"] = variant_key
    return result


def _sse_event(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


@web_app.post("/api/base/stream")
async def api_base_stream(req: BaseChatRequest):
    if not req.user_prompt.strip():
        raise HTTPException(status_code=400, detail="user_prompt is required")
    variant_key = _resolve_variant_key(req.mode)
    rt = await asyncio.to_thread(get_runtime)
    started = time.time()

    async def _stream():
        try:
            gen = rt.generate_base_stream(
                req.user_prompt, req.system_prompt,
                _clamped_base_max(req.max_new_tokens), variant_key,
                _clamped_act(req.max_act_tokens),
            )
            for event in gen:
                if event["type"] == "done":
                    event["latency_sec"] = round(time.time() - started, 3)
                    event["active_mode"] = variant_key
                yield _sse_event(event)
        except Exception:
            logger.exception("Base streaming failed")
            yield _sse_event({"type": "error", "detail": "Generation failed"})

    return StreamingResponse(_stream(), media_type="text/event-stream")


@web_app.post("/api/retrieval/stream")
async def api_retrieval_stream(req: RetrievalRequest):
    if not req.user_prompt.strip() or not req.assistant_response.strip():
        raise HTTPException(
            status_code=400,
            detail="user_prompt and assistant_response are required",
        )
    variant_key = _resolve_variant_key(req.mode)
    rt = await asyncio.to_thread(get_runtime)
    started = time.time()

    async def _stream():
        try:
            gen = rt.answer_retrieval_stream(
                req.user_prompt, req.assistant_response, req.retrieval_prompt,
                req.system_prompt,
                _clamped_retrieval_max(req.max_new_tokens), variant_key,
                _clamped_act(req.max_act_tokens),
            )
            for event in gen:
                if event["type"] == "done":
                    event["latency_sec"] = round(time.time() - started, 3)
                    event["active_mode"] = variant_key
                yield _sse_event(event)
        except Exception:
            logger.exception("Retrieval streaming failed")
            yield _sse_event({"type": "error", "detail": "Retrieval failed"})

    return StreamingResponse(_stream(), media_type="text/event-stream")


@web_app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse((ASSETS_DIR / "index.html").read_text(encoding="utf-8"))


def main() -> None:
    import uvicorn

    global _checkpoint_dir
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--checkpoint-dir", default=None,
                        help="Where the released checkpoints live / are downloaded to "
                             "(default: $PRISM_DEMO_CHECKPOINT_DIR or ./checkpoints)")
    args = parser.parse_args()
    if args.checkpoint_dir:
        _checkpoint_dir = Path(args.checkpoint_dir)
    uvicorn.run(web_app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
