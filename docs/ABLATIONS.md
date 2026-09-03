# Training ablations

Use the same [training data and setup](../README.md#training) as the main recipes.
Evaluation-only activation ablations are documented in
[prism-eval](https://github.com/Offensive-AI-Lab/prism-eval/blob/main/docs/ABLATION_REPORT.md).

## Hook layer

The Qwen layer sweep uses `{2, 6, 10, 13, 16, 19, 23, 27, 31}`. Run one layer
per GPU:

```bash
LAYER=13 recipes/ablate_layer_qwen3.5-9b.sh
```

The script uses on-the-fly extraction with the standard validity mask and
[split](DATA_CARD.md#filtering-and-splits). It keeps the Qwen SFT learning rates,
with micro-batch 2 and gradient accumulation 32. Runs default to one epoch;
set `ABLATE_EPOCHS=3` for the full SFT schedule.

Checkpoints go to `$PRISM_CKPT_DIR/ablate-sft-qwen-L<layer>/`. Compare
`best_val_loss` in each `best.pt`, or W&B's `val/loss`. Include layer 16
in the sweep: its on-the-fly result is the matched control, rather than the
released checkpoint trained from a cache.

## Training seed

Hold the SFT initialization and activation cache fixed while varying the GRPO
training seed. Keep the judge endpoint and remaining settings unchanged:

```bash
SEED=1337 PRISM_SFT_INIT_FROM=/path/to/sft/best.pt \
  recipes/grpo_seed_qwen3.5-9b.sh
```

This uses the [released GRPO settings](RECIPES.md#grpo-settings) and writes to
`$PRISM_CKPT_DIR/grpo-qwen3.5-9b-L16-seed<seed>/`. Use cached activations so
the comparison varies training randomness without adding in-loop extraction
differences.

Report the mean and standard deviation across the chosen seeds. This measures
training variability; bootstrap intervals over evaluation records measure a
different source of uncertainty.
