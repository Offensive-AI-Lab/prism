#!/usr/bin/env bash
# GRPO — Qwen3.5-0.8B, hook layer 12. Standard do_rl recipe (lr 2e-5,
# n_candidates 6, k3 KL @0.05, gen 144, prioritized sampling ON, under-length
# collapse penalty ON). Requires an SFT checkpoint (run sft_qwen3.5-0.8b.sh
# first, or set PRISM_SFT_INIT_FROM) and a judge endpoint (scripts/serve_judge.sh).
#
# GPU utilisation: GRPO also hits the 248k-vocab logit wall — each step runs
# batch x n_candidates (=6) rollouts and takes per-token logprobs over the full
# vocab, so the effective logit batch is prompts x 6 x seq. gradient_checkpointing
# is ON (rl config default) which helps. This default is a modest bump over the
# 9B's batch 2 and is NOT yet memory-calibrated for the 0.8B — do a SMOKE run
# with the judge up (SMOKE=1) and raise PRISM_BATCH_SIZE from the observed peak.
STAGE=rl PROFILE=qwen3.5-0.8b TAG=qwen3.5-0.8b-L12 HOOK_LAYER=12
RL_RUN_NAME="${RL_RUN_NAME:-grpo-qwen3.5-0.8b-L12}"
export STAGE PROFILE TAG HOOK_LAYER RL_RUN_NAME
# ── GPU knobs (override on the CLI to retune) ─────────────────────────────────
export PRISM_MAX_TARGET_LEN="${PRISM_MAX_TARGET_LEN:-512}"  # harmless if unused by RL
export PRISM_BATCH_SIZE="${PRISM_BATCH_SIZE:-4}"    # prompts/step (x6 candidates); calibrate with judge up
source "$(dirname "$0")/_lib.sh"
do_rl
