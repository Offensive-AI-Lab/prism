#!/usr/bin/env bash
# SFT (projection + LoRA) — Ministral-3-8B, released recipe.
STAGE=sft PROFILE=ministral3-8b TAG=ministral3-8b-L17 HOOK_LAYER=17
export STAGE PROFILE TAG HOOK_LAYER
source "$(dirname "$0")/_lib.sh"
do_sft
