#!/usr/bin/env bash
# SFT (projection + LoRA) — Qwen3.5-0.8B, hook layer 12 (mid-depth of 24).
#
# New target model: there is no released checkpoint to pin against, so the
# learning rates use the config defaults (lr 4e-5, projection_lr 3e-4) rather
# than the 9B's sweep-sampled literals. Sweep/retune if convergence needs it —
# note the larger effective batch below may warrant a higher lr.
#
# GPU utilisation (calibrated on a 96 GB card): the memory wall here is NOT the
# 0.8B model (~1.6 GB) but the LM-head cross-entropy over Qwen3.5's 248k-token
# vocab (logits are batch x seq x vocab, fp32-upcast) plus the on-the-fly
# extraction forward over long contexts. We therefore (a) cap the target length
# — instruction_set labels are p99=302 tokens, so a 512 cap truncates almost
# nothing while bounding the CE seq — and (b) use a micro-batch that survives the
# long-context / long-label tail of real shuffled batches. Measured on-the-fly:
# micro-batch 16 peaks ~66 GB (stable over 400+ real steps); 24 and 32 OOM.
# The PRECOMPUTE path (default below) drops the extraction forward, so it has
# headroom for a larger PRISM_BATCH_SIZE — raise it and watch nvidia-smi.
# All knobs are env-overridable.
STAGE=sft PROFILE=qwen3.5-0.8b TAG=qwen3.5-0.8b-L12 HOOK_LAYER=12
export STAGE PROFILE TAG HOOK_LAYER
# ── GPU knobs (override on the CLI to retune) ─────────────────────────────────
export BATCH="${BATCH:-128}"                              # activation-precompute batch (frozen, early-exit fwd)
export PRISM_MAX_TARGET_LEN="${PRISM_MAX_TARGET_LEN:-512}"  # cap label len (p99=302) -> bounds CE memory
export PRISM_BATCH_SIZE="${PRISM_BATCH_SIZE:-16}"        # SFT micro-batch (248k-vocab CE is the wall)
export PRISM_GRAD_ACCUM="${PRISM_GRAD_ACCUM:-4}"         # effective batch = 16 x 4 = 64 (matches the 9B)
export PRISM_EVAL_SAMPLES="${PRISM_EVAL_SAMPLES:-2000}"
source "$(dirname "$0")/_lib.sh"
do_sft
