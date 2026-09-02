"""Compare-mode baselines for the local demo: LatentQA and Activation Oracles.

Both baselines read the same target model as PRISM, so their LoRA adapters are
attached to the demo's single Qwen3.5-9B instance as extra named adapters —
their read/collect passes run with all adapters disabled and their generate
passes with their own adapter selected. Nothing here loads a second copy of
the base model.

Adapter checkpoints are two PEFT directories (adapter_config.json +
adapter_model.safetensors) under PRISM_DEMO_BASELINES_DIR (default
./checkpoints/baselines): latentqa/ and ao/. Missing files are downloaded
from Hugging Face on the first compare request (~580 MB total) and verified
against their SHA-256 digests. Set PRISM_DEMO_DISABLE_COMPARE=1 to hide
compare mode entirely.

The `lit` (LatentQA) and `nl_probes` (Activation Oracles) packages are
vendored under third_party/ — see the LICENSE files there.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger("prism.demo.baselines")

sys.path.insert(0, str(Path(__file__).resolve().parent / "third_party"))

WEIGHTS_ORG = os.environ.get("PRISM_DEMO_WEIGHTS_ORG", "Offensive-AI-Lab")

# Prefiltered adapters published alongside the PRISM checkpoints. The LatentQA
# adapter is the lm_head/embed-stripped variant (116 MB instead of 2.1 GB).
BASELINE_ADAPTERS = {
    "latentqa": {
        "repo": "prism-baseline-latentqa-qwen3.5-9b",
        "files": {
            "adapter_config.json": "a952a74b6be55979834a491a7da2f1ddfff0f9ddc153116b692c6b6ab4a220a6",
            "adapter_model.safetensors": "8600f3ba51e60e53de72ff044adee962dca3c179a9b8a1212809c236a7852cd2",
        },
    },
    "ao": {
        "repo": "prism-baseline-activation-oracles-qwen3.5-9b",
        "files": {
            "adapter_config.json": "b049c48d32dfbb25e605949b6859dbe2eb53bd680d36eb6171f82e97003306cc",
            "adapter_model.safetensors": "1830598a70e652d4bf5a39de4439d7683e8e5825cc5c053b66cdcbbe187e1612",
        },
    },
}

LATENTQA_TAIL_TOKENS = 128
AO_SEGMENT_TOKENS = 128
DEFAULT_QUESTION = "List all instructions given to the assistant."

LATENTQA_META = {
    "key": "latentqa",
    "display_name": "LatentQA",
    "subtitle": "Decoder LoRA reads target activations and answers a free-form question.",
    "default_question": DEFAULT_QUESTION,
}
AO_META = {
    "key": "ao",
    "display_name": "Activation Oracles",
    "subtitle": "Verbalizer LoRA reads injected activations and answers a free-form question.",
    "default_question": DEFAULT_QUESTION,
}


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


def _prepare_filtered_adapter_dir(src_dir: Path, drop_modules=("lm_head", "embed_tokens")) -> Path:
    """Copy a PEFT adapter dir to /tmp with the named modules stripped from
    both adapter_config.json and adapter_model.safetensors — handles vocab
    drift between the training-time model revision and the current one."""
    from safetensors.torch import load_file, save_file

    dst_dir = Path(tempfile.mkdtemp(prefix=f"adapter_filtered_{src_dir.name}_"))
    with open(src_dir / "adapter_config.json") as f:
        cfg = json.load(f)

    tm = cfg.get("target_modules")
    if isinstance(tm, list):
        cfg["target_modules"] = [m for m in tm if m not in drop_modules]
    mts = cfg.get("modules_to_save")
    if isinstance(mts, list):
        cfg["modules_to_save"] = [m for m in mts if m not in drop_modules]

    with open(dst_dir / "adapter_config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    state = load_file(str(src_dir / "adapter_model.safetensors"))
    filtered = {k: v for k, v in state.items()
                if not any(s in k for s in drop_modules)}
    dropped = len(state) - len(filtered)
    if dropped:
        logger.info(f"[{src_dir.name}] dropped {dropped} {drop_modules} keys for vocab safety")
    save_file(filtered, str(dst_dir / "adapter_model.safetensors"))
    return dst_dir


def _patch_ao_get_hf_submodule() -> None:
    """nl_probes hard-codes layer paths that break on PEFT-wrapped multimodal
    models; replace with the homogeneous-ModuleList walker."""
    def _patched(model, layer, use_lora=False):
        return _find_layers(model)[layer]

    import nl_probes.utils.activation_utils as ao_act_utils
    import nl_probes.base_experiment as be
    ao_act_utils.get_hf_submodule = _patched
    be.get_hf_submodule = _patched


def _remap_language_model_keys(model, state: Dict[str, torch.Tensor], label: str) -> Dict[str, torch.Tensor]:
    live_keys = list(model.state_dict().keys())
    needs_remap = any("language_model" in k for k in live_keys)
    if needs_remap and not any("language_model" in k for k in state):
        state = {
            k.replace(
                "base_model.model.model.layers.",
                "base_model.model.model.language_model.layers.",
            ): v
            for k, v in state.items()
        }
        logger.info(f"[{label}] remapped {len(state)} LoRA keys (.language_model prefix)")
    return state


class BaselineManager:
    """Lazy-loads both baseline adapters onto the PRISM runtime's model."""

    def __init__(self, runtime, baselines_dir: Path):
        self.runtime = runtime            # PrismRuntime (already loaded)
        self.baselines_dir = baselines_dir
        self.lqa_tokenizer = None
        self.module_read = None
        self.module_write = None
        self.lit_cfg = None
        self._loaded = False
        self._lock = threading.Lock()
        self.state: Dict[str, str] = {"status": "idle", "message": ""}

    # ── availability / status ───────────────────────────────────────────

    def adapters_present(self) -> bool:
        return all(
            (self.baselines_dir / name / fname).exists()
            for name, spec in BASELINE_ADAPTERS.items()
            for fname in spec["files"]
        )

    def _sha256_of(self, path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                h.update(block)
        return h.hexdigest()

    def ensure_adapters(self) -> None:
        """Download any missing adapter file and verify every digest."""
        from huggingface_hub import hf_hub_download

        for name, spec in BASELINE_ADAPTERS.items():
            target_dir = self.baselines_dir / name
            target_dir.mkdir(parents=True, exist_ok=True)
            for fname, digest in spec["files"].items():
                path = target_dir / fname
                if not path.exists():
                    repo_id = f"{WEIGHTS_ORG}/{spec['repo']}"
                    logger.info(f"[{name}] downloading {repo_id}/{fname} → {path}")
                    hf_hub_download(repo_id=repo_id, filename=fname, local_dir=str(target_dir))
                got = self._sha256_of(path)
                if got != digest:
                    raise RuntimeError(
                        f"{path} has SHA-256 {got}, expected {digest}. "
                        f"Delete the file and retry to re-download."
                    )
            logger.info(f"[{name}] adapter verified under {target_dir}")

    def get_state(self) -> Dict[str, str]:
        return dict(self.state)

    # ── loading ─────────────────────────────────────────────────────────

    def ensure_loaded(self) -> None:
        with self._lock:
            if self._loaded:
                self.state = {"status": "ready", "message": ""}
                return
            try:
                if not self.adapters_present():
                    self.state = {"status": "loading",
                                  "message": "Downloading baseline adapters (~580 MB on first use)..."}
                self.ensure_adapters()
                self.state = {"status": "loading",
                              "message": "Attaching LatentQA + Activation Oracle adapters (~30s)..."}
                self._load()
                self._loaded = True
                self.state = {"status": "ready", "message": ""}
            except Exception:
                logger.exception("Baseline load failed")
                self.state = {"status": "error", "message": "Baseline load failed"}
                raise

    def _load(self) -> None:
        from peft import PeftConfig, set_peft_model_state_dict
        from safetensors.torch import load_file as load_safetensors
        from transformers import AutoTokenizer, PreTrainedModel
        from lit.configs.interpret_config import interpret_config
        from lit.reading import ForCausalLMLossPatched

        model = self.runtime.model
        model_id = self.runtime.train_cfgs[list(self.runtime.train_cfgs)[0]]["model_id"]

        _patch_ao_get_hf_submodule()
        # LatentQA patches the loss fn globally; safe to repeat.
        PreTrainedModel.loss_function = staticmethod(ForCausalLMLossPatched)

        # Left-padded tokenizer for the baselines (single-example batches).
        self.lqa_tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        if self.lqa_tokenizer.pad_token is None:
            self.lqa_tokenizer.pad_token = self.lqa_tokenizer.eos_token

        # ── AO verbalizer adapter ────────────────────────────────────────
        ao_dir = self.baselines_dir / "ao"
        ao_cfg = PeftConfig.from_pretrained(str(ao_dir))
        ao_cfg.lora_dropout = 0.0
        model.add_adapter("ao", ao_cfg)
        ao_state = load_safetensors(str(ao_dir / "adapter_model.safetensors"))
        ao_state = _remap_language_model_keys(model, ao_state, "ao")
        set_peft_model_state_dict(model, ao_state, adapter_name="ao")
        # nl_probes expects HF's PEFT-integration method names.
        model.disable_adapters = model.disable_adapter_layers
        model.enable_adapters = model.enable_adapter_layers
        logger.info("[baselines] AO LoRA attached as adapter 'ao'")

        # ── LatentQA decoder adapter (lm_head/embed_tokens stripped) ─────
        lqa_dir = _prepare_filtered_adapter_dir(self.baselines_dir / "latentqa")
        lqa_cfg = PeftConfig.from_pretrained(str(lqa_dir))
        lqa_cfg.lora_dropout = 0.0
        model.add_adapter("latentqa", lqa_cfg)
        lqa_state = load_safetensors(str(lqa_dir / "adapter_model.safetensors"))
        lqa_state = _remap_language_model_keys(model, lqa_state, "latentqa")
        set_peft_model_state_dict(model, lqa_state, adapter_name="latentqa")
        shutil.rmtree(lqa_dir, ignore_errors=True)
        logger.info("[baselines] LatentQA LoRA attached as adapter 'latentqa'")

        # ── LatentQA read/write modules on the shared layer stack ────────
        self.lit_cfg = interpret_config()
        self.lit_cfg.target_model_name = model_id
        layers = _find_layers(model)
        self.module_read = layers[self.lit_cfg.min_layer_to_read]
        self.module_write = layers[self.lit_cfg.layer_to_write]
        logger.info(
            f"[baselines] LatentQA modules: read layer {self.lit_cfg.min_layer_to_read}, "
            f"write layer {self.lit_cfg.layer_to_write} (shared stack of {len(layers)})"
        )
        model.set_adapter("default")
        model.eval()

    # ── LatentQA ────────────────────────────────────────────────────────

    def _lqa_build_batch(self, user_prompt, assistant_response, question, tail_tokens):
        from lit.utils.dataset_utils import (
            BASE_DIALOG,
            DECODER_CHAT_TEMPLATES,
            ENCODER_CHAT_TEMPLATES,
            NUM_READ_TOKENS_TO_SHIFT,
            NUM_WRITE_TOKENS_TO_SHIFT,
        )

        name = self.lit_cfg.target_model_name
        encoder_chat = ENCODER_CHAT_TEMPLATES.get(name, None)
        read_prompt = self.lqa_tokenizer.apply_chat_template(
            [
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": assistant_response},
            ],
            tokenize=False,
            chat_template=encoder_chat,
        )
        tokenized_read = self.lqa_tokenizer(
            [read_prompt], return_tensors="pt", padding=True, add_special_tokens=False,
        )

        resp_ids = self.lqa_tokenizer(assistant_response, add_special_tokens=False).input_ids
        read_len = min(tail_tokens, len(resp_ids))
        read_lengths = torch.tensor([read_len], dtype=torch.long)

        pad_count = max(1, read_len - NUM_READ_TOKENS_TO_SHIFT[name])
        query = [{"role": "user", "content": "? " * pad_count}]
        query += BASE_DIALOG + [{"role": "user", "content": question}]
        decoder_chat = (
            DECODER_CHAT_TEMPLATES[name] if self.lit_cfg.modify_chat_template else None
        )
        rendered = self.lqa_tokenizer.apply_chat_template(
            query, tokenize=False, add_generation_prompt=True, chat_template=decoder_chat,
        )
        tokenized_write = self.lqa_tokenizer(
            [rendered], return_tensors="pt", padding=True, add_special_tokens=False,
        )
        write_lengths = (
            torch.sum(tokenized_write.attention_mask, dim=1)
            - NUM_WRITE_TOKENS_TO_SHIFT[name]
        )
        return {
            "tokenized_read": tokenized_read,
            "tokenized_write": tokenized_write,
            "read_lengths": read_lengths,
            "write_lengths": write_lengths,
        }

    def latentqa_stream(self, user_prompt, assistant_response, question,
                        max_new_tokens, tail_tokens: int = LATENTQA_TAIL_TOKENS):
        from lit.utils.activation_utils import _forward_cache_outputs, no_op
        from transformers import TextIteratorStreamer

        if not question.strip():
            question = DEFAULT_QUESTION

        model = self.runtime.model
        device = next(model.parameters()).device
        batch = self._lqa_build_batch(user_prompt, assistant_response, question, tail_tokens)
        tokenized_read = batch["tokenized_read"].to(device)
        tokenized_write = batch["tokenized_write"].to(device)
        read_lengths = batch["read_lengths"]
        write_lengths = batch["write_lengths"]

        # Read pass on the shared model with every adapter off (= base model).
        model.disable_adapter_layers()
        try:
            activation_cache = _forward_cache_outputs(
                model, self.lqa_tokenizer, tokenized_read, [self.module_read],
                token_idx=None, no_grad=True, prepare_inputs=no_op,
            )
        finally:
            model.enable_adapter_layers()
        module_activations = [a.to(device) for a in activation_cache]

        # Write pass with the LatentQA decoder adapter + substitution hook.
        model.set_adapter("latentqa")
        num_hook_triggered = [0]

        def _sub_hook(module, inp, output):
            new_output = output[0] if isinstance(output, tuple) else output
            _, read_seq_len, _ = module_activations[0].shape
            _, write_seq_len, _ = new_output.shape
            for i in range(len(new_output)):
                read_mask_len = read_lengths[i].item()
                write_mask_len = write_lengths[i].item() + num_hook_triggered[0]
                new_output[i] = torch.cat(
                    [
                        new_output[i][: write_seq_len - write_mask_len, :],
                        module_activations[0][i, read_seq_len - read_mask_len:, :],
                        new_output[i][write_seq_len - (write_mask_len - read_mask_len):, :],
                    ],
                    dim=0,
                )
            num_hook_triggered[0] += 1
            if isinstance(output, tuple):
                return (new_output,) + output[1:]
            return new_output

        hook_handle = self.module_write.register_forward_hook(_sub_hook)
        streamer = TextIteratorStreamer(self.lqa_tokenizer, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs = {
            "input_ids": tokenized_write.input_ids,
            "attention_mask": tokenized_write.attention_mask,
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.lqa_tokenizer.eos_token_id,
            "eos_token_id": self.lqa_tokenizer.eos_token_id,
            "do_sample": True,
            "num_beams": 1,
            "temperature": 0.0001,
            "use_cache": False,
            "streamer": streamer,
        }
        error_box: Dict[str, Any] = {}

        def _run():
            try:
                with torch.no_grad():
                    model.generate(**gen_kwargs)
            except Exception as exc:
                error_box["error"] = exc

        thread = threading.Thread(target=_run)
        thread.start()
        try:
            for text in streamer:
                if text:
                    yield text
        finally:
            thread.join()
            hook_handle.remove()
            model.set_adapter("default")
            if "error" in error_box:
                raise error_box["error"]

    # ── Activation Oracles ──────────────────────────────────────────────

    def ao_stream(self, user_prompt, assistant_response, question,
                  max_new_tokens, segment_tokens: int = AO_SEGMENT_TOKENS):
        import nl_probes.base_experiment as be
        from nl_probes.base_experiment import VerbalizerInputInfo, VerbalizerEvalConfig
        from transformers import TextIteratorStreamer

        if not question.strip():
            question = DEFAULT_QUESTION

        model = self.runtime.model
        device = next(model.parameters()).device
        info = VerbalizerInputInfo(
            context_prompt=[
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": assistant_response},
            ],
            verbalizer_prompt=question,
            ground_truth="",
        )
        streamer = TextIteratorStreamer(self.lqa_tokenizer, skip_prompt=True, skip_special_tokens=True)
        cfg = VerbalizerEvalConfig(
            model_name=self.lit_cfg.target_model_name,
            activation_input_types=["orig"],
            verbalizer_input_types=["segment"],
            segment_start_idx=-segment_tokens,
            segment_end_idx=0,
            segment_repeats=1,
            verbalizer_generation_kwargs={
                "do_sample": False,
                "max_new_tokens": max_new_tokens,
                "streamer": streamer,
            },
            add_generation_prompt=False,
        )
        error_box: Dict[str, Any] = {}

        def _run():
            try:
                with torch.no_grad():
                    be.run_verbalizer(
                        model=model,
                        tokenizer=self.lqa_tokenizer,
                        verbalizer_prompt_infos=[info],
                        verbalizer_lora_path="ao",
                        target_lora_path=None,
                        config=cfg,
                        device=device,
                    )
            except Exception as exc:
                error_box["error"] = exc

        thread = threading.Thread(target=_run)
        thread.start()
        try:
            for text in streamer:
                if text:
                    yield text
        finally:
            thread.join()
            model.set_adapter("default")
            model.enable_adapter_layers()
            if "error" in error_box:
                raise error_box["error"]
