#!/usr/bin/env bash
# SFT — Qwen3.5-9B (hook layer 16). Produces the recipe behind the released
# prism-qwen3.5-9b-sft.pt (val-loss-selected best.pt).
#
# The learning rates are sweep-sampled — pinned here as exact literals
# recovered from the released checkpoint's embedded config:
#   lr            = 4.176320076421569e-05
#   projection_lr = 3.205823229668696e-04
# Everything else (r=32, alpha=64, dropout=0.05, batch 4 × grad_accum 16,
# 3 epochs, projection + LoRA) is the config default.
STAGE=sft PROFILE=qwen3.5-9b TAG=qwen3.5-9b-L16 HOOK_LAYER=16
export STAGE PROFILE TAG HOOK_LAYER
export PRISM_LR=4.176320076421569e-05
export PRISM_PROJECTION_LR=3.205823229668696e-04
export PRISM_EVAL_SAMPLES=2000
source "$(dirname "$0")/_lib.sh"
do_sft
