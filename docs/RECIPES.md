# Training recipes

Run these scripts from the repository root after the
[training setup](../README.md#training). Each GRPO recipe reads `best.pt` from
its corresponding SFT run; `PRISM_SFT_INIT_FROM` selects another SFT checkpoint.

| Target | SFT recipe | GRPO recipe | Hook layer | Projection width |
|---|---|---|---:|---:|
| Qwen3.5-9B | [sft_qwen3.5-9b.sh](../recipes/sft_qwen3.5-9b.sh) | [grpo_qwen3.5-9b.sh](../recipes/grpo_qwen3.5-9b.sh) | 16 | 4096 |
| Gemma-2-9B-it | [sft_gemma2-9b.sh](../recipes/sft_gemma2-9b.sh) | [grpo_gemma2-9b.sh](../recipes/grpo_gemma2-9b.sh) | 21 | 3584 |
| Ministral-3-8B | [sft_ministral3-8b.sh](../recipes/sft_ministral3-8b.sh) | [grpo_ministral3-8b.sh](../recipes/grpo_ministral3-8b.sh) | 17 | 4096 |

All recipes use the same [training records](DATA_CARD.md), seed 42, and
up to 128 response-token activations. LoRA uses rank 32, alpha 64, and dropout
0.05 on the attention and MLP projection modules; Ministral restricts adapters
to its language-model stack. The projection is trainable in both stages, and
the decoder receives no text retrieval prompt.

## Optimization

| Setting | Qwen SFT | Gemma / Ministral SFT | GRPO, all targets |
|---|---|---|---|
| LoRA learning rate | `4.176320076421569e-05` | `4e-5` | `2e-5` |
| Projection learning rate | `3.205823229668696e-04` | `3e-4` | `5e-6` |
| Batch × gradient accumulation | 4 × 16 | 4 × 16 | 2 × 1 |
| Training length | 3 epochs | 3 epochs | 1 epoch, capped at 20,000 updates |
| Validation samples | 2,000 | 1,500 | 500 |
| Checkpoint selection | Lowest validation loss | Lowest validation loss | Highest validation judge reward |

The scripts and [shared recipe code](../recipes/_lib.sh) define these settings,
including overrides to the Python module defaults. Use the scripts for
reproduction.

## GRPO settings

| Setting | Value |
|---|---|
| Candidate reports per record | 6 |
| Generation | 144 new tokens; temperature 1.2; top-p 0.95 |
| KL regularization | k3 estimator; coefficient 0.05 |
| Dynamic sampling | Minimum reward standard deviation 0.05; maximum mean reward 0.95 |
| Prioritized sampling | Enabled |
| Coverage / hallucination weights | 1.0 / 0.4 |
| Over-length penalty | 0.15 per bullet beyond 1.5 × ground-truth count |
| Under-length penalty | 0.15 per bullet below 0.5 × ground-truth count |
| Judge | `google/gemma-4-31B-it`, reasoning disabled |

The released runs used cached activations. See the
[pipeline guide](PIPELINE.md#activation-extraction) for on-the-fly extraction and
cache reuse, and [training ablations](ABLATIONS.md) for layer and seed controls.
