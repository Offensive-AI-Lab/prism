#!/usr/bin/env bash
# GRPO (20,000 steps) — Ministral-3-8B (n_candidates=6, prioritized sampling).
STAGE=rl PROFILE=ministral3-8b TAG=ministral3-8b-L17 HOOK_LAYER=17
export STAGE PROFILE TAG HOOK_LAYER
source "$(dirname "$0")/_lib.sh"
do_rl
