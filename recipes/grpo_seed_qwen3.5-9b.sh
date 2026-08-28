#!/usr/bin/env bash
# Seed-variability run: the RELEASED GRPO recipe (= recipes/grpo_qwen3.5-9b.sh, the run behind the published
# prism-qwen3.5-9b-grpo.pt), varying ONLY the training seed. The released
# run is seed 42. Everything else is identical: 20,000 steps, prioritized
# sampling ON, under-length penalty ON, n_candidates 6, k3 KL @0.05,
# lr 2e-5 / plr 5e-6, gen 144, same precompute, same SFT init.
#
#   SEED=1337 PRISM_SFT_INIT_FROM=/path/to/released-sft/best.pt \
#       recipes/grpo_seed_qwen3.5-9b.sh
STAGE=rl PROFILE=qwen3.5-9b TAG=qwen3.5-9b-L16 HOOK_LAYER=16
: "${SEED:?set SEED (released run is seed 42; new seeds e.g. 7 1337 2026)}"
: "${PRISM_SFT_INIT_FROM:?set PRISM_SFT_INIT_FROM to the released SFT best.pt}"
export PRISM_SEED=$SEED
RL_RUN_NAME="grpo-qwen3.5-9b-L16-seed$SEED"
RL_CKPT_DIR="${RL_CKPT_DIR:-${PRISM_CKPT_DIR:?}/grpo-qwen3.5-9b-L16-seed$SEED}"
export STAGE PROFILE TAG HOOK_LAYER RL_RUN_NAME RL_CKPT_DIR
source "$(dirname "$0")/_lib.sh"
do_rl
