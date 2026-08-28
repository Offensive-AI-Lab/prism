"""Export-tool contract: schema, sanitization, dtype handling."""

import runpy
import sys
from pathlib import Path

import pytest
import torch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "export_checkpoint.py"


def _fake_training_ckpt(method: str = "grpo") -> dict:
    cfg = {
        "model_id": "Qwen/Qwen3.5-9B",
        "hook_layer": 16,
        "lora_r": 32,
        "lora_alpha": 64,
        "lora_target_modules": ["q_proj"],
        "max_act_tokens": 128,
        "skip_prompt_b": True,
        "projection_dim": 8,
        "use_projection": True,
        "checkpoint_dir": "/mnt/cluster/somegroup/Users/x/ckpts/run1",
        "precomputed_dir": "/mnt/cluster/somegroup/Users/x/pre",
        "sft_init_from": "/home/somebody/sft/best.pt",
        "dataset_paths": ["/mnt/cluster/a.jsonl"],
        "wandb_entity": "some-lab",
        "wandb_project": "itm-rl-phase1",
        "wandb_run_name": "grpo-prio-run-xyz",
    }
    ckpt = {
        "config": cfg,
        "lora_state": {"base_model.model.layers.0.q_proj.lora_A.weight": torch.randn(4, 8)},
        "projection_state": {"proj.weight": torch.randn(8, 8), "proj.bias": torch.randn(8)},
        "opt_step": 1234,
        "optimizer_state": {"dummy": 1},
        "scheduler_state": {"dummy": 2},
        "wandb_run_id": "abc123",
    }
    if method == "grpo":
        ckpt["val_reward"] = 0.9
        ckpt["best_reward"] = 0.91
        ckpt["reward_tracker_state"] = {"w": [1.0]}
    else:
        ckpt["best_val_loss"] = 0.5
        ckpt["encoder_state"] = None
    return ckpt


def _run_export(tmp_path: Path, ckpt: dict, *extra: str) -> Path:
    src = tmp_path / "best.pt"
    torch.save(ckpt, src)
    out_dir = tmp_path / "exports"
    argv = ["export_checkpoint.py", str(src), "--out-dir", str(out_dir), *extra]
    old = sys.argv
    sys.argv = argv
    try:
        runpy.run_path(str(SCRIPT), run_name="__main__")
    finally:
        sys.argv = old
    files = list(out_dir.glob("*.pt"))
    assert len(files) == 1
    return files[0]


def test_export_schema_and_sanitization(tmp_path):
    out = _run_export(tmp_path, _fake_training_ckpt("grpo"))
    assert out.name == "prism-qwen3.5-9b-grpo.pt"
    exported = torch.load(out, map_location="cpu", weights_only=False)
    # Exactly the release schema — nothing else survives.
    assert sorted(exported) == ["config", "lora_state", "opt_step", "projection_state"]
    assert exported["opt_step"] == 1234
    # No machine-specific string anywhere in the config.
    import json
    blob = json.dumps(exported["config"], default=str)
    for marker in ("/mnt/", "/home/", "some-lab", "grpo-prio-run-xyz"):
        assert marker not in blob, marker
    # Projection cast to bf16; LoRA untouched (fp32).
    assert all(v.dtype == torch.bfloat16 for v in exported["projection_state"].values())
    assert all(v.dtype == torch.float32 for v in exported["lora_state"].values())


def test_export_sft_autodetect(tmp_path):
    out = _run_export(tmp_path, _fake_training_ckpt("sft"))
    assert out.name == "prism-qwen3.5-9b-sft.pt"


def test_export_rejects_non_prism_checkpoint(tmp_path):
    with pytest.raises(SystemExit):
        _run_export(tmp_path, {"weights": torch.randn(2)})
