#!/bin/bash
# Anti-collapse GRPO continuation: resume from the last SPECIFIC (pre-collapse) checkpoint
# with kl_coef 0.08 (between the collapsing 0.05 and the over-constraining 0.15) and
# gen_temperature 1.0 (was 1.2). Checkpoint SELECTION is external: the eval-suite probe
# harness scores saved best_step*.pt by real coverage (val reward is a broken selector).
set -uo pipefail
cd ~/prism-release
export HF_HUB_CACHE=/groups/yisroel_dtgroup/Users/models/hub
export UV_CACHE_DIR=/tmp/uv-cache-$USER
export TOKENIZERS_PARALLELISM=false PYTORCH_ALLOC_CONF=expandable_segments:True WANDB_MODE=online
export WANDB_DIR=/groups/yisroel_dtgroup/Users/pankajak/prism-08b-grpo/wandb
export PRISM_DATA_DIR=/groups/yisroel_dtgroup/Users/pankajak/prism-08b-grpo/data
export PRISM_CKPT_DIR=/groups/yisroel_dtgroup/Users/pankajak/prism-08b-grpo/ckpts
export PRISM_DATASET_SRC=/groups/yisroel_dtgroup/Users/itr_msft/smoke_release/prompt-only
export PRISM_SFT_INIT_FROM=/home/pankajak/prism-08b-runs/ckpts/sft-qwen3.5-0.8b-L12-lr1e4-3ep/best.pt
export PRISM_ON_THE_FLY=1
export PRISM_JUDGE_ENDPOINT_FILE=/groups/yisroel_dtgroup/Users/pankajak/prism-08b-grpo/judge_endpoint.env
export PRISM_JUDGE_MODEL=gemma4-31B-it PRISM_JUDGE_API_KEY=not-needed
export PRISM_BATCH_SIZE=6 RL_NCAND=6

# ---- the anti-collapse recipe ----
export RL_KL_COEF=0.08                      # between 0.05 (collapsed) and 0.15 (flat)
export PRISM_GEN_TEMP=1.0                    # was 1.2 (widened drift); pair with kl↑
export RL_STEPS="${RL_STEPS:-12000}"        # resume 4400 -> generous cap; eval-select picks the peak
export PRISM_EVAL_EVERY=250
export RESUME_FROM="${RESUME_FROM:-/groups/yisroel_dtgroup/Users/pankajak/prism-08b-grpo/ckpts/grpo-qwen3.5-0.8b-L12/best_step4400_v0.8596.pt}"
export RESUME_STRIP_WANDB=1
export RL_CKPT_DIR=/groups/yisroel_dtgroup/Users/pankajak/prism-08b-grpo/ckpts/grpo-qwen3.5-0.8b-L12-midkl
export RL_RUN_NAME=grpo-qwen3.5-0.8b-L12-midkl
mkdir -p "$WANDB_DIR" "$PRISM_DATA_DIR" "$RL_CKPT_DIR"
echo "branch=$(git branch --show-current) MIDKL kl=$RL_KL_COEF temp=$PRISM_GEN_TEMP resume=$(basename $RESUME_FROM) cap=$RL_STEPS"
bash recipes/grpo_qwen3.5-0.8b.sh
echo "EXIT=$?"
