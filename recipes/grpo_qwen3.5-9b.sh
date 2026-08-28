#!/usr/bin/env bash
# GRPO — Qwen3.5-9B. This is the RELEASED RL recipe: its best.pt is the
# published prism-qwen3.5-9b-grpo.pt.
#
# Recipe = the standard do_rl defaults (20,000 steps): lr 2e-5, n_candidates 6, k3 KL
# @0.05, gen 144, prioritized sampling ON, under-length collapse penalty ON
# (config default). Data: the activation cache (extracted automatically on first run).
STAGE=rl PROFILE=qwen3.5-9b TAG=qwen3.5-9b-L16 HOOK_LAYER=16
RL_RUN_NAME="${RL_RUN_NAME:-grpo-qwen3.5-9b-L16}"
export STAGE PROFILE TAG HOOK_LAYER RL_RUN_NAME
source "$(dirname "$0")/_lib.sh"
do_rl
