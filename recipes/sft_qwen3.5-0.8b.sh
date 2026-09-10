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
# long-context / long-label tail of real shuffled batches. NOTE: SFT keeps full
# decoder activations (no gradient checkpointing, unlike RL), so the base memory
# of a max-length batch is high — micro-batch 16 looked fine for 400 steps
# (~66 GB) but OOM'd ~1.8 h into a full epoch on a worst-case batch (~88 GB base
# + the per-dataset logging cross-entropy). That logging CE is now computed
# sample-by-sample (train.py), so it no longer spikes; micro-batch 12 leaves
# clear margin and completes. To push GPU higher, enable gradient checkpointing
# for SFT (frees the activation memory that is the real wall) and raise the batch.
# All knobs are env-overridable.
STAGE=sft PROFILE=qwen3.5-0.8b TAG=qwen3.5-0.8b-L12 HOOK_LAYER=12
export STAGE PROFILE TAG HOOK_LAYER
# ── GPU knobs (override on the CLI to retune) ─────────────────────────────────
export BATCH="${BATCH:-128}"                              # activation-precompute batch (frozen, early-exit fwd)
export PRISM_MAX_TARGET_LEN="${PRISM_MAX_TARGET_LEN:-512}"  # cap label len (p99=302) -> bounds CE memory
export PRISM_BATCH_SIZE="${PRISM_BATCH_SIZE:-12}"        # SFT micro-batch (activation memory is the wall)
export PRISM_GRAD_ACCUM="${PRISM_GRAD_ACCUM:-5}"         # effective batch = 12 x 5 = 60 (~ the 9B's 64)
export PRISM_EVAL_SAMPLES="${PRISM_EVAL_SAMPLES:-2000}"
source "$(dirname "$0")/_lib.sh"
do_sft
