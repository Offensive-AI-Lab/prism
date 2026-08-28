#!/usr/bin/env python
"""Export a PRISM training checkpoint to the prism-eval release format.

Input:  a training checkpoint (best.pt / best_step*.pt) written by
        prism.sft.train or prism.rl.train.
Output: prism-{target}-{method}.pt with exactly the keys the prism-eval
        runner loads:

            config            training config, path/entity fields sanitized
            lora_state        PEFT adapter state dict (fp32)
            projection_state  ActivationProjection state dict (cast to bf16)
            opt_step          the optimizer step the checkpoint was taken at

        Optimizer/scheduler/reward-tracker state, W&B run ids, and val metrics
        are stripped. This matches the schema of the published checkpoints on
        HuggingFace (Offensive-AI-Lab/prism-*), with the improvement that no
        local filesystem paths survive in the embedded config.

Usage:
    uv run python scripts/export_checkpoint.py CKPT [--method sft|grpo] \
        [--out-dir exports/] [--name prism-....pt]

The released files (Offensive-AI-Lab/prism-*) are each training run's
best.pt exported with this tool.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

# Keys copied to the release file. Everything else (optimizer_state,
# scheduler_state, encoder_state, reward_tracker_state, val/best metrics,
# wandb_run_id, ...) is dropped.
_KEEP_TOP = ("config", "lora_state", "projection_state", "opt_step")

# Config fields whose values are local paths / run-tracking identity: blanked
# so no machine-specific string ships inside the release file.
_SCRUB_TO_NONE = (
    "checkpoint_dir", "precomputed_dir", "resume_from", "sft_init_from",
    "encoder_checkpoint", "hard_ids_json", "wandb_entity",
)
_SCRUB_STR = {
    "wandb_project": "prism",
    "wandb_run_name": "released",
}

_MODEL_TAGS = {
    "Qwen/Qwen3.5-9B": "qwen3.5-9b",
    "google/gemma-2-9b-it": "gemma-2-9b-it",
    "mistralai/Ministral-3-8B-Instruct-2512-BF16": "ministral-3-8b",
}


def _sanitize_config(cfg: dict) -> dict:
    out = dict(cfg)
    for k in _SCRUB_TO_NONE:
        if k in out:
            out[k] = None
    for k, v in _SCRUB_STR.items():
        if k in out:
            out[k] = v
    if "dataset_paths" in out:
        out["dataset_paths"] = []
    return out


def _assert_clean(cfg: dict) -> None:
    blob = json.dumps(cfg, default=str)
    for marker in ("/mnt/", "/home/", "/scratch/", "/groups/"):
        if marker in blob:
            raise SystemExit(
                f"sanitization failed: {marker!r} still present in config: "
                f"{[k for k, v in cfg.items() if marker in str(v)]}"
            )


def _infer_method(ckpt: dict) -> str:
    # RL checkpoints carry reward bookkeeping; SFT ones carry val-loss.
    if "val_reward" in ckpt or "best_reward" in ckpt or "reward_tracker_state" in ckpt:
        return "grpo"
    return "sft"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkpoint", help="training best.pt / best_step*.pt")
    ap.add_argument("--method", choices=("sft", "grpo"), default=None,
                    help="override the sft/grpo auto-detection")
    ap.add_argument("--name", default=None,
                    help="override the output filename (default prism-{target}-{method}.pt)")
    ap.add_argument("--out-dir", default="exports")
    args = ap.parse_args()

    src = Path(args.checkpoint)
    if not src.is_file():
        raise SystemExit(f"checkpoint not found: {src}")
    ckpt = torch.load(src, map_location="cpu", weights_only=False)

    missing = [k for k in ("config", "lora_state") if k not in ckpt]
    if missing:
        raise SystemExit(f"{src} does not look like a PRISM training checkpoint "
                         f"(missing {missing}); top keys: {sorted(ckpt)}")

    cfg = _sanitize_config(dict(ckpt["config"]))
    _assert_clean(cfg)

    method = args.method or _infer_method(ckpt)
    model_id = cfg.get("model_id", "unknown")
    tag = _MODEL_TAGS.get(model_id)
    if tag is None:
        tag = model_id.split("/")[-1].lower()
        print(f"WARNING: unknown model_id {model_id!r}; using tag {tag!r}", file=sys.stderr)
    name = args.name or f"prism-{tag}-{method}.pt"

    out = {
        "config": cfg,
        "lora_state": ckpt["lora_state"],
        "opt_step": ckpt.get("opt_step"),
    }
    proj = ckpt.get("projection_state")
    if proj is not None:
        # Published checkpoints store the projection in bf16 (halves file size;
        # the eval runner loads it to bf16 anyway). LoRA stays fp32 as-released.
        out["projection_state"] = {k: v.to(torch.bfloat16) for k, v in proj.items()}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / name
    torch.save(out, dst)

    sha = hashlib.sha256(dst.read_bytes()).hexdigest()
    size_mb = dst.stat().st_size / 1e6
    print(f"exported : {dst}")
    print(f"schema   : {sorted(out)}")
    print(f"model_id : {model_id}   hook_layer={cfg.get('hook_layer')}   "
          f"opt_step={out['opt_step']}   method={method}")
    print(f"lora     : {len(out['lora_state'])} tensors (fp32)")
    if proj is not None:
        print(f"proj     : {[tuple(v.shape) for v in out['projection_state'].values()]} (bf16)")
    print(f"size     : {size_mb:.1f} MB")
    print(f"sha256   : {sha}")


if __name__ == "__main__":
    main()
