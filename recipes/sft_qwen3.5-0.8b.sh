#!/usr/bin/env bash
# SFT (projection + LoRA) — Qwen3.5-0.8B, hook layer 12 (mid-depth of 24).
#
# New target model: there is no released checkpoint to pin against, so the
# learning rates use the config defaults (lr 4e-5, projection_lr 3e-4) rather
# than the 9B's sweep-sampled literals. Sweep/retune if convergence needs it —
# note the larger effective batch below may warrant a higher lr.
#
# GPU utilisation: the 0.8B target is ~1.6 GB in bf16, so on a 96 GB card there
# is huge headroom. We raise the precompute batch and the SFT micro-batch well
# above the 9B settings (batch 4 x grad_accum 16). All knobs are env-overridable
# — calibrate from the smoke run's peak memory (the recipe echoes nvidia-smi)
# to push closer to the card limit.
STAGE=sft PROFILE=qwen3.5-0.8b TAG=qwen3.5-0.8b-L12 HOOK_LAYER=12
export STAGE PROFILE TAG HOOK_LAYER
# ── Max-GPU knobs (override on the CLI to retune) ─────────────────────────────
export BATCH="${BATCH:-512}"                        # activation-precompute batch (frozen inference)
export PRISM_BATCH_SIZE="${PRISM_BATCH_SIZE:-128}"  # SFT micro-batch
export PRISM_GRAD_ACCUM="${PRISM_GRAD_ACCUM:-1}"    # effective batch = 128 x 1
export PRISM_EVAL_SAMPLES="${PRISM_EVAL_SAMPLES:-2000}"
source "$(dirname "$0")/_lib.sh"
do_sft
