# Ablations

## Layer selection (hook-layer ablation)

The released Qwen3.5-9B monitors use layer 16 of 32. The ablation evaluates
layers `{2, 6, 10, 13, 16, 19, 23, 27, 31}`, covering early, middle, and late
depths.

Protocol:

- Data: cleaned instruction-labelled JSONL files with on-the-fly activation
  extraction. This avoids building a separate activation cache for every layer.
  The loader applies `valid_record_ids.json` and the standard split function described in
  `docs/PIPELINE.md`, so every layer uses the same train and validation records.
- Training: the released SFT configuration and sweep-selected learning rates.
  Train each layer for one epoch to obtain the ranking curve, then train the
  best layer and layer 16 for the full three-epoch schedule.
- Selection: validation loss, with token accuracy reported as a secondary
  metric. Validation loss is also used to select `best.pt` in the SFT recipes.
- Reference: compare against the layer-16 run from this ablation rather than
  the layer-16 result printed in the paper. All ablation runs use the
  training-time model class, while the released precomputed activations used
  the causal-LM class.

Run one layer per GPU:

```bash
LAYER=13 PRISM_DATA_DIR=/path/to/data PRISM_CKPT_DIR=/path/to/ckpts \
    recipes/ablate_layer_qwen3.5-9b.sh
```

Compare the best validation loss from each run using W&B `val/loss` or the
value stored in `best.pt`. `ABLATE_EPOCHS` defaults to 1. `PRISM_PYTHON` can
select a cluster Python environment instead of `uv run`, and
`PRISM_HOOK_LAYER` overrides the configured layer.

## Seed variability (GRPO)

To estimate run-to-run variance, repeat the released GRPO recipe from the same
SFT checkpoint and activation cache while changing only `PRISM_SEED`. The seed
affects rollout sampling and data order; the precomputed validation split stays
fixed. Treat the released seed-42 run as one sample and report the mean and
standard deviation across all runs. Keep the remaining settings equal to the
configuration embedded in the released checkpoint.

```bash
SEED=1337 PRISM_SFT_INIT_FROM=/path/to/released-sft/best.pt \
    PRISM_DATA_DIR=... PRISM_CKPT_DIR=... PRISM_JUDGE_BASE_URL=... \
    recipes/grpo_seed_qwen3.5-9b.sh
```

Bootstrap confidence intervals over the 1,000 evaluation records provide a
separate estimate of measurement uncertainty. That analysis belongs in
`prism-eval`, where the per-record scores are stored.

GPU kernels are not bitwise deterministic, so repeated runs with the same seed
can still differ. The reported spread therefore includes both seed and hardware
run-to-run variability.
