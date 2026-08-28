#!/usr/bin/env bash
# GRPO (20,000 steps) — gemma-2-9b-it (n_candidates=6, prioritized sampling).
STAGE=rl PROFILE=gemma2-9b TAG=gemma2-9b-L21 HOOK_LAYER=21
export STAGE PROFILE TAG HOOK_LAYER
source "$(dirname "$0")/_lib.sh"
do_rl
