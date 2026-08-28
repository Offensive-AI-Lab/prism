#!/usr/bin/env bash
# SFT (projection + LoRA) — gemma-2-9b-it, released recipe.
STAGE=sft PROFILE=gemma2-9b TAG=gemma2-9b-L21 HOOK_LAYER=21
export STAGE PROFILE TAG HOOK_LAYER
source "$(dirname "$0")/_lib.sh"
do_sft
