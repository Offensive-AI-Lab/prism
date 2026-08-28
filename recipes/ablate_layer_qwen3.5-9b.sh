#!/usr/bin/env bash
# Layer-selection ablation — SFT with the activation hook
# at layer $LAYER, ON-THE-FLY extraction from the cleaned dataset
# JSONLs (the same data every recipe trains on). No training activations are stored.
# One GPU per run; launch one per layer in parallel. See docs/ABLATIONS.md.
#
#   LAYER=13 PRISM_DATA_DIR=... PRISM_CKPT_DIR=... recipes/ablate_layer_qwen3.5-9b.sh
# SMOKE=1 trains on a 170-record slice of each source file (minutes, not hours).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO_ROOT"
_load_dotenv() {
  local f="$1" line key
  [ -f "$f" ] || return 0
  while IFS= read -r line || [ -n "$line" ]; do
    case "$line" in ''|\#*) continue ;; esac
    case "$line" in *=*) ;; *) continue ;; esac
    key="${line%%=*}"
    [ -n "${!key:-}" ] && continue
    export "$line"
  done < "$f"
}
_load_dotenv .env
# Python launcher — override with PRISM_PYTHON=.venv/bin/python on clusters
# where `uv run` hits NFS flock issues (os error 37).
PY="${PRISM_PYTHON:-uv run python}"
: "${LAYER:?set LAYER (ablation grid: 2 6 10 13 16 19 23 27 31)}"
: "${PRISM_DATA_DIR:?set PRISM_DATA_DIR (root holding prompt-only/jsonl)}"
DATASET_JSONL_DIR="${PRISM_DATASET_SRC:-$PRISM_DATA_DIR/prompt-only}/jsonl"
if ! compgen -G "$DATASET_JSONL_DIR/*.jsonl" >/dev/null; then
  echo "ERROR: no oracle dataset at $DATASET_JSONL_DIR/*.jsonl" >&2
  echo "       Generate it first: scripts/generate_dataset.sh, then scripts/clean_dataset.sh" >&2
  exit 1
fi
: "${PRISM_CKPT_DIR:?set PRISM_CKPT_DIR}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PRISM_HOOK_LAYER=$LAYER
# Released SFT recipe values (sweep-sampled literals; see docs/RECIPES.md)
export PRISM_LR=4.176320076421569e-05
export PRISM_PROJECTION_LR=3.205823229668696e-04
export PRISM_EVAL_SAMPLES="${PRISM_EVAL_SAMPLES:-2000}"
# Micro-batch 2 x grad-accum 32 = the released effective batch of 64, at half
# the peak memory (the on-the-fly path holds extraction + decode together;
# long-sequence batches OOM at micro-batch 4 on 95 GB).
export PRISM_BATCH_SIZE="${PRISM_BATCH_SIZE:-2}"
export PRISM_GRAD_ACCUM="${PRISM_GRAD_ACCUM:-32}"
if [ "${SMOKE:-0}" = "1" ]; then
  SUB="$PRISM_DATA_DIR/precomputed/SMOKE-ablate-L$LAYER-subset"; rm -rf "$SUB"; mkdir -p "$SUB"
  for f in "$DATASET_JSONL_DIR"/*.jsonl; do head -n 170 "$f" > "$SUB/$(basename "$f")"; done
  DATASET_JSONL_DIR="$SUB"
  export PRISM_EVAL_EVERY=5 PRISM_EVAL_SAMPLES=32 PRISM_LOG_EVERY=1
  export PRISM_CHECKPOINT_DIR="$PRISM_CKPT_DIR/SMOKE-ablate-sft-qwen-L$LAYER"
  export PRISM_WANDB_RUN_NAME="SMOKE-ablate-sft-qwen-L$LAYER"
else
  export PRISM_CHECKPOINT_DIR="$PRISM_CKPT_DIR/ablate-sft-qwen-L$LAYER"
  export PRISM_WANDB_RUN_NAME="ablate-sft-qwen-L$LAYER"
fi
EPOCHS="${ABLATE_EPOCHS:-1}"   # 1 epoch ranks layers; finalists rerun with 3
echo "[ablate] layer=$LAYER epochs=$EPOCHS data=$DATASET_JSONL_DIR ckpt=$PRISM_CHECKPOINT_DIR"
$PY -m prism.sft.train --epochs "$EPOCHS" \
  --dataset-paths "$DATASET_JSONL_DIR"/*.jsonl
echo "ABLATION RUN DONE → $PRISM_CHECKPOINT_DIR"
