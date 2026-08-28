"""Pin the config defaults + recipe literals that reproduce the released runs.

The released hyperparameters were recovered from the released checkpoints'
embedded configs (docs/RECIPES.md). If one of these asserts fires, either a
default drifted (breaking reproduction) or docs/RECIPES.md needs updating.
"""

import os
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def configs():
    # Configs are module-level dicts mutated by the target-model overlay at
    # import time; make sure no ambient profile/env override is active before
    # they are first imported (this module sorts first alphabetically).
    for var in list(os.environ):
        if var.startswith("PRISM_"):
            del os.environ[var]
    from prism.rl.config import RL_CONFIG
    from prism.sft.config import FINETUNE_CONFIG
    return FINETUNE_CONFIG, RL_CONFIG


def test_sft_defaults_match_released_recipe(configs):
    sft, _ = configs
    assert sft["hook_layer"] == 16
    assert sft["projection_dim"] == 4096
    assert sft["use_projection"] is True
    assert sft["skip_prompt_b"] is True
    assert sft["max_act_tokens"] == 128
    assert sft["lora_r"] == 32
    assert sft["lora_alpha"] == 64
    assert sft["lora_dropout"] == 0.05
    assert sorted(sft["lora_target_modules"]) == sorted(
        ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    assert sft["batch_size"] == 4
    assert sft["grad_accum"] == 16
    assert sft["epochs"] == 3
    assert sft["max_target_len"] == 1024
    assert sft["weight_decay"] == 0.01
    assert sft["warmup_ratio"] == 0.03


def test_rl_defaults_match_released_recipe(configs):
    _, rl = configs
    assert rl["pref_loss"] == "grpo"
    assert rl["kl_coef"] == 0.05
    assert rl["gen_temperature"] == 1.2
    assert rl["gen_top_p"] == 0.95
    assert rl["instruction_weight"] == 1.0
    assert rl["hallucination_weight"] == 0.4
    assert rl["length_penalty_k"] == 1.5
    assert rl["length_penalty_lambda"] == 0.15
    assert rl["length_penalty_under_k"] == 0.5
    assert rl["length_penalty_under_lambda"] == 0.15
    assert rl["dynamic_sampling_min_std"] == 0.05
    assert rl["dynamic_sampling_max_mean"] == 0.95
    assert rl["projection_lr"] == 5e-6
    assert rl["train_projection_in_rl"] is True
    # Dead flags removed in the public release.
    assert "use_token_kl" not in rl
    assert "inner_steps_per_round" not in rl


def _read(rel: str) -> str:
    return (REPO / rel).read_text()


def test_lib_recipe_pins_released_rl_overrides():
    lib = _read("recipes/_lib.sh")
    for flag in ("--lr 2e-5", "--kl-estimator k3", "--kl-coef 0.05",
                 "--gen-max-new-tokens 144"):
        assert flag in lib, flag


def test_qwen_sft_recipe_pins_exact_sweep_lrs():
    sft = _read("recipes/sft_qwen3.5-9b.sh")
    assert "PRISM_LR=4.176320076421569e-05" in sft
    assert "PRISM_PROJECTION_LR=3.205823229668696e-04" in sft


def test_qwen_grpo_recipe_matches_released_run():
    # Released prism-qwen3.5-9b-grpo.pt = the released GRPO run's best
    # checkpoint: standard do_rl defaults — 20,000 steps cap, prioritized
    # sampling ON, n_candidates 6.
    grpo = _read("recipes/grpo_qwen3.5-9b.sh")
    assert "RL_PRIO" not in grpo        # prio ON is the do_rl default
    assert "RL_STEPS" not in grpo       # 20000 is the do_rl default
    assert re.search(r"RL_NCAND", grpo) is None   # default 6
    assert "do_rl" in grpo


def test_target_profiles_match_released_checkpoints():
    from prism.target_models import PROFILES
    q = PROFILES["qwen3.5-9b"]
    assert (q["hidden_size"], q["hook_layer"]) == (4096, 16)
    g = PROFILES["gemma2-9b"]
    assert (g["hidden_size"], g["hook_layer"]) == (3584, 21)
    assert g["attn_implementation"] == "eager"
    m = PROFILES["ministral3-8b"]
    assert (m["hidden_size"], m["hook_layer"]) == (4096, 17)
    assert "language_model" in m["lora_target_modules"]


def test_hook_layer_env_override():
    from prism.target_models import apply_profile_overlay
    cfg = {"hook_layer": 16, "eval_samples": 500}
    os.environ["PRISM_HOOK_LAYER"] = "23"
    try:
        apply_profile_overlay(cfg)
    finally:
        del os.environ["PRISM_HOOK_LAYER"]
    assert cfg["hook_layer"] == 23


def test_seed_env_override():
    from prism.target_models import apply_profile_overlay
    cfg = {"seed": 42}
    os.environ["PRISM_SEED"] = "1337"
    try:
        apply_profile_overlay(cfg)
    finally:
        del os.environ["PRISM_SEED"]
    assert cfg["seed"] == 1337
