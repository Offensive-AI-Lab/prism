"""
config.py — Central configuration for PRISM monitor SFT.

Architecture (matches the released checkpoints and the prism-eval runner):
frozen target model (hooked at a mid layer) → linear projection into the
decoder embedding space → the same target model with LoRA adapters decodes
the ITM report. A single model instance with the LoRA toggle saves ~19 GB
VRAM (adapters OFF for activation extraction, ON for decoding).

Modes:
  - skip_prompt_b: decoder gets [soft_tokens | instruction_set] with no retrieval_prompt
    (the released recipe — the monitor decodes from activations alone)
"""

FINETUNE_CONFIG = {
    # ─── Model ─────────────────────────────────────────────────────────────────
    "model_id": "Qwen/Qwen3.5-9B",

    # Layer to hook for activation extraction
    "hook_layer": 16,

    # ─── Datasets (JSONL files from prism.datagen.generator) ───────────────────
    # Each record has: id, source_dataset, prompt, response, retrieval_prompt, instruction_set, metadata
    # Only used for on-the-fly activation extraction; the released runs trained
    # from precomputed shards (see precomputed_dir below). Set explicitly
    # (--dataset-paths / PRISM_DATASET_PATHS).
    "dataset_paths": [],
    # Record-id mask for the on-the-fly path (same file the cache honours).
    # None = auto-detect valid_record_ids.json next to the JSONL files.
    "valid_record_ids": None,
    # On-the-fly train/val/test split. These are the extractor's defaults
    # (recipes/_lib.sh do_precompute: --val-ratio 0.1 --test-ratio 0.1
    # --seed 42, no stratification), so an on-the-fly run sees the same
    # split membership as the activation cache built from the same files.
    # split_seed is separate from the training seed on purpose: PRISM_SEED
    # varies training stochasticity only, never the data split.
    "split_val_ratio": 0.1,
    "split_test_ratio": 0.1,
    "split_seed": 42,
    "split_stratify_by": None,

    "seed": 42,

    # ─── Precomputed activations ────────────────────────────────────────────────
    # Set to a directory path (produced by `python -m prism.activations`) to use
    # precomputed activations instead of extracting on-the-fly. When set, hook
    # registration is skipped. A prompt-only directory additionally carries a
    # valid_record_ids.json mask (≤6-bullet cap + template-leak / word-
    # fragmentation filter) honoured by PrecomputedActivationDataset.
    # Set via PRISM_PRECOMPUTED_DIR / --precomputed-dir; None = on-the-fly
    # extraction from dataset_paths (SFT and GRPO both support either).
    "precomputed_dir": None,

    # ─── Activation extraction ──────────────────────────────────────────────────
    # Max activation tokens to extract from response (tail).
    # If response is shorter, use however many tokens are available.
    "max_act_tokens": 128,

    # ─── Projection layer ──────────────────────────────────────────────────────
    # Linear(hidden → hidden) mapping raw activations → decoder embedding space.
    "use_projection": True,  # False = feed raw activations directly (ablation)
    "projection_dim": 4096,

    # ─── LoRA on decoder ───────────────────────────────────────────────────────
    "lora_r": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.05,
    "lora_target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],

    # ─── Training ──────────────────────────────────────────────────────────────
    "batch_size": 4,
    "grad_accum": 16,        # effective batch = 4 × 16 = 64
    "epochs": 3,
    "lr": 4e-5,
    "projection_lr": 3e-4,  # separate LR for projection layer
    "grad_clip": 1.0,
    "weight_decay": 0.01,
    "warmup_ratio": 0.03,
    "min_lr_ratio": 0.1,

    # Max tokens for retrieval_prompt + instruction_set combined (truncate instruction_set if longer)
    "max_target_len": 1024,

    # ─── Skip retrieval_prompt ──────────────────────────────────────────────────────────
    # When True, decoder input is [soft_tokens | instruction_set] with no retrieval_prompt.
    # The model learns to decode instruction_set directly from activations alone.
    "skip_prompt_b": True,

    # ─── Prompt B variations ─────────────────────────────────────────────────
    # When enabled, randomly replace retrieval_prompt during TRAINING with semantically
    # equivalent probes to improve generalisation. Validation always uses the
    # original retrieval_prompt from the JSONL for consistent comparison.
    "use_retrieval_prompt_variations": False,
    "retrieval_prompt_variations": [
    ],

    # ─── Evaluation ────────────────────────────────────────────────────────────
    "log_every": 10,        # log every N optimizer steps
    "eval_every": 500,      # validate every N optimizer steps (~50 evals over full run)
    "eval_samples": 1500,    # max val samples per eval
    "log_sample_count": 10,  # number of generated samples to log to WandB per eval
    "early_stop_patience": 5,
    "bert_score_samples": 50,  # 0 = disabled; number of generated samples to compute BERTScore on
    "run_test_eval": False,

    # ─── Checkpointing ─────────────────────────────────────────────────────────
    "checkpoint_dir": "./checkpoints/sft-default",
    "wandb_project": "prism-sft",
    "wandb_run_name": "sft-default",
    "resume_from": None,
}

# ─── Target-model overlay ─────────────────────────────────────────────────────
# When PRISM_TARGET_MODEL is set (gemma2-9b / ministral3-8b), rewrite model_id,
# hook_layer, projection_dim, lora_target_modules to match that target model and
# apply generic PRISM_* path/pacing env overrides. No-op for Qwen (default).
from prism.target_models import apply_profile_overlay as _apply_profile_overlay  # noqa: E402
_apply_profile_overlay(FINETUNE_CONFIG)
