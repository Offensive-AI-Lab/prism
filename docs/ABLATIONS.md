# Training ablations

These runs use the same [data and setup](../README.md#training) as the main recipes. Ablations that only change activations at evaluation time are documented in [prism-eval](https://github.com/Offensive-AI-Lab/prism-eval/blob/main/docs/ABLATION_REPORT.md).

## Hook layer

The paper compares nine Qwen layers: 2, 6, 10, 13, 16, 19, 23, 27, and 31. Each layer is a separate SFT run, one per GPU:

```bash
LAYER=13 recipes/ablate_layer_qwen3.5-9b.sh
```

The sweep extracts activations on the fly, using the standard validity mask and [split](DATA_CARD.md#filtering-and-splits). It keeps the Qwen SFT learning rates with a micro-batch of 2 and 32 gradient-accumulation steps. Each run defaults to one epoch; set `ABLATE_EPOCHS=3` for the full SFT schedule.

Checkpoints are written to `$PRISM_CKPT_DIR/ablate-sft-qwen-L<layer>/`. Compare runs by `best_val_loss` in each `best.pt`, or by `val/loss` in W&B. Include layer 16 in the sweep: because the sweep extracts activations on the fly, its own layer-16 run is the fair comparison point, not the released checkpoint, which was trained from a cache.

## Training seed

To measure run-to-run variation, keep the SFT initialization, activation cache, judge endpoint, and every other setting fixed, and change only the GRPO seed:

```bash
SEED=1337 PRISM_SFT_INIT_FROM=/path/to/sft/best.pt \
  recipes/grpo_seed_qwen3.5-9b.sh
```

This uses the [released GRPO settings](RECIPES.md#grpo-settings) and writes to `$PRISM_CKPT_DIR/grpo-qwen3.5-9b-L16-seed<seed>/`. Use cached activations, so that training randomness is the only thing that differs between runs.

Report the mean and standard deviation across seeds. This measures training variability, which is a different source of uncertainty from bootstrap intervals over evaluation records.
