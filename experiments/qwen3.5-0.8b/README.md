# PRISM Qwen3.5-0.8B (experimental)

A small (0.8B) PRISM monitor. Research-grade — a faithful-coverage ceiling of
**~0.54** on the 1000-record eval suite, below the 9B models; kept as a branch,
not merged to main.

## Canonical checkpoint
`grpo-qwen3.5-0.8b-L12-midkl/best_step5800` — promoted to
`prism-qwen3.5-0.8b-grpo.pt`. Chosen as the **faithful** operating point:
0% generic-opener (no reward-hack), lowest hallucination, coverage tied with
every later (partly reward-hacked) checkpoint. Model `Qwen/Qwen3.5-0.8B`, hook
layer 12.

## Official numbers (judge = gemma4-31B-it)
overall coverage 0.538, halluc 0.028 · BN 0.927 / BC 0.518 / AP 0.413 / HO 0.294.

## Reproduce
- `grpo_midkl.sh` — anti-collapse GRPO continuation (RL_KL_COEF=0.08,
  PRISM_GEN_TEMP=1.0, resume from the pre-collapse step-4400 checkpoint).
- `select_checkpoint.py <probe_dir ...>` — score checkpoints by coverage AND
  generic-opener % (val reward alone hid the collapse; select on faithfulness).
- Evaluate with [`prism-eval`](https://github.com/Offensive-AI-Lab/prism-eval)
  (branch `qwen3.5-0.8b`): `configs/main/qwen3.5-0.8b-grpo.yaml`.
