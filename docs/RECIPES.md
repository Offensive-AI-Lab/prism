# Released-run recipes

Hyperparameters below were recovered **directly from the released runs'
checkpoints** (each training checkpoint embeds its full config), not from
memory or lab notes. `scripts/export_checkpoint.py` applied to the source
runs reproduces the published files tensor-for-tensor.

## The four released checkpoints

| | SFT (qwen) | GRPO qwen | GRPO gemma-2 | GRPO ministral |
|---|---|---|---|---|
| released file | prism-qwen3.5-9b-sft.pt | prism-qwen3.5-9b-grpo.pt | prism-gemma-2-9b-it-grpo.pt | prism-ministral-3-8b-grpo.pt |
| recipe | `sft_qwen3.5-9b.sh` | `grpo_qwen3.5-9b.sh` | `grpo_gemma2-9b.sh` | `grpo_ministral3-8b.sh` |
| selected checkpoint | best val loss | best val reward | best val reward | best val reward |
| model / hook layer / proj dim | Qwen3.5-9B / 16 / 4096 | same | gemma-2-9b-it / 21 / 3584 | Ministral-3-8B / 17 / 4096 |
| use_projection / skip_prompt_b | ✓ / ✓ | same | same | same |
| LoRA r / α / dropout | 32 / 64 / 0.05 (7 proj modules) | same | same | same, regex-scoped to `language_model` |
| lr | 4.176320076421569e-05 † | 2e-5 | 2e-5 | 2e-5 |
| projection_lr | 3.205823229668696e-04 † | 5e-6 | 5e-6 | 5e-6 |
| batch × grad_accum × epochs | 4 × 16 × 3 | 2 × 1 × 1 | 2 × 1 | 2 × 1 |
| n_candidates | — | 6 | 6 | 6 |
| KL estimator / coef | — | k3 / 0.05 | k3 / 0.05 | k3 / 0.05 |
| gen max_new / temp / top_p | — | 144 / 1.2 / 0.95 | same | same |
| dynamic sampling min_std / max_mean | — | 0.05 / 0.95 | same | same |
| prioritized sampling | — | yes | yes | yes |
| under-length penalty | — | yes | yes | yes |
| reward weights inst / halluc | — | 1.0 / 0.4 | same | same |
| judge | — | gemma4-31B-it (`google/gemma-4-31B-it`, reasoning off) | same | same |
| step cap | — | 20 000 | 20 000 | 20 000 |
| data | the oracle dataset, **precomputed activation cache** (the default path; on-the-fly extraction is supported but was not used for any reported run) | same | same | same |

† sweep-sampled values, pinned as exact literals in `recipes/sft_qwen3.5-9b.sh`.

## Recipes vs. module defaults

The recipes are the reproduction path. `python -m prism.rl.train` on its own
starts from the config defaults in `src/prism/rl/config.py` (2,000 steps, 4
candidates, token-exact KL, lr 5e-6, no prioritized sampling); `recipes/_lib.sh`
passes the flags that turn those into the released settings above.

## Exact-reproduction caveats

1. **Use the released dataset for exact reproduction**
   (`scripts/download_dataset.py`). Regenerating with the generation scripts
   approximates it — sampling and judge filtering are not bitwise
   deterministic across hardware.
2. GRPO needs a running judge endpoint: `scripts/serve_judge.sh` (a
   Gemma-4-31B-it class model; ~2×80 GB or 1×95 GB GPU).
