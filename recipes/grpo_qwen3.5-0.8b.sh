#!/usr/bin/env bash
# GRPO — Qwen3.5-0.8B, hook layer 12. Standard do_rl recipe (lr 2e-5,
# n_candidates 6, k3 KL @0.05, gen 144, prioritized sampling ON, under-length
# collapse penalty ON). Requires an SFT checkpoint (run sft_qwen3.5-0.8b.sh
# first, or set PRISM_SFT_INIT_FROM) and a judge endpoint (scripts/serve_judge.sh).
#
# GPU utilisation: raise the prompt batch far above the 9B default (2) for the
# tiny 0.8B target (each step runs batch x n_candidates rollouts). Env-
# overridable — calibrate from the smoke run.
STAGE=rl PROFILE=qwen3.5-0.8b TAG=qwen3.5-0.8b-L12 HOOK_LAYER=12
RL_RUN_NAME="${RL_RUN_NAME:-grpo-qwen3.5-0.8b-L12}"
export STAGE PROFILE TAG HOOK_LAYER RL_RUN_NAME
# ── Max-GPU knobs (override on the CLI to retune) ─────────────────────────────
export PRISM_BATCH_SIZE="${PRISM_BATCH_SIZE:-32}"   # prompts per GRPO step (x6 candidates)
source "$(dirname "$0")/_lib.sh"
do_rl
