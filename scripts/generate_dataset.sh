#!/usr/bin/env bash
# Generate the instruction-set dataset (prompt-only labels) for the three sources,
# sequentially. Output lands in $PRISM_DATA_DIR/prompt-only/jsonl/ so the next
# steps (scripts/clean_dataset.sh, then the recipes' precompute) pick it up
# without rearranging files. Each source gets its own JSONL +
# checkpoint dir so a failure in one source does not affect the others. Needs an
# OpenAI-compatible server (e.g. vLLM) serving the target model at $DATAGEN_BASE_URL.
set -euo pipefail

# Run from the repo root so `uv run` resolves this project's environment
# regardless of the caller's cwd.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

: "${PRISM_DATA_DIR:?set PRISM_DATA_DIR to the root for generated datasets}"
OUT_DIR="${DATAGEN_OUT_DIR:-$PRISM_DATA_DIR/prompt-only/jsonl}"
CKPT_ROOT="${DATAGEN_CKPT_DIR:-$PRISM_DATA_DIR/datagen_checkpoints}"
BASE_URL="${DATAGEN_BASE_URL:-http://localhost:8089/v1}"
MODEL="${DATAGEN_MODEL:-Qwen/Qwen3.5-9B}"
MAX_PER="${DATAGEN_MAX_PER_SOURCE:-30000}"
PARAS="${DATAGEN_PARAPHRASES:-2}"
CONCURRENCY="${DATAGEN_CONCURRENCY:-64}"
mkdir -p "$OUT_DIR"

if ! curl -s -m 3 "${BASE_URL%/}/models" | grep -q "\"id\""; then
  echo "ERROR: no OpenAI-compatible server at $BASE_URL (expected the target model, e.g." >&2
  echo "       \`vllm serve $MODEL --port 8089\`). Set DATAGEN_BASE_URL if it runs elsewhere." >&2
  exit 1
fi

run_source () {
  local SRC="$1"
  local OUT="$OUT_DIR/prompt_only_instruction_set_dataset_${SRC}.jsonl"
  local CKPT="$CKPT_ROOT/${SRC}"
  echo "==== [$(date)] starting source=$SRC ===="
  uv run python -m prism.datagen.generator \
    --backend http_async \
    --model "$MODEL" \
    --model-url "$BASE_URL" \
    --sources "$SRC" \
    --max-per-source "$MAX_PER" \
    --paraphrases-per-example "$PARAS" \
    --instruction-set-mode prompt_only \
    --output "$OUT" \
    --checkpoint-dir "$CKPT" \
    --concurrency "$CONCURRENCY" \
    --chunk-size 512
  local N; N=$(wc -l < "$OUT")
  if [ "$N" -eq 0 ]; then
    echo "ERROR: source=$SRC produced 0 records — check the server log at $BASE_URL" >&2
    exit 1
  fi
  echo "==== [$(date)] finished source=$SRC ($N records) ===="
}

run_source if_eval
run_source if_multi_constraints
run_source ultrachat

echo "==== [$(date)] ALL SOURCES DONE ===="
ls -lh "$OUT_DIR"/prompt_only_instruction_set_dataset_*.jsonl
