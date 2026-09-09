#!/usr/bin/env bash
# Dataset cleaning: rules + LLM-judge filter → valid_record_ids.json mask.
#
# Input: the instruction-set JSONLs written by scripts/generate_dataset.sh into
# $DATASET_DIR/jsonl/. Output: filtered JSONLs (kept / .removed / .errors) in
# $FILTER_OUT_DIR and a valid_record_ids.json mask (bullet cap / template-leak /
# word-fragmentation rules) next to the data — the recipes' precompute copies
# the mask into the activation dir, where the training loaders apply it.
#
# Required: PRISM_DATA_DIR. Needs an OpenAI-compatible server serving the
# target model at $DATAGEN_BASE_URL.
set -euo pipefail

# Run from the repo root so `uv run` resolves this project's environment
# regardless of the caller's cwd.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

: "${PRISM_DATA_DIR:?set PRISM_DATA_DIR to the root for datasets}"
TGT_DIR="${DATASET_DIR:-$PRISM_DATA_DIR/prompt-only}"
FILTER_OUT="${FILTER_OUT_DIR:-$PRISM_DATA_DIR/filtered}"
BASE_URL="${DATAGEN_BASE_URL:-http://localhost:8089/v1}"
MODEL="${DATAGEN_MODEL:-Qwen/Qwen3.5-9B}"
JUDGE_CONCURRENCY="${FILTER_JUDGE_CONCURRENCY:-256}"
JUDGE_MAX_TOKENS="${FILTER_JUDGE_MAX_TOKENS:-200}"

if ! curl -s -m 3 "${BASE_URL%/}/models" | grep -q "\"id\""; then
  echo "ERROR: vLLM server not reachable at $BASE_URL" >&2
  exit 1
fi

echo "==== [$(date)] dataset summary ===="
for f in "$TGT_DIR"/jsonl/*.jsonl; do
  echo "  $(basename "$f"): $(wc -l < "$f") records"
done

echo "==== [$(date)] STEP 1: filter (rules + LLM judge) ===="
uv run python -m prism.datagen.filter \
    --input "$TGT_DIR/jsonl/*.jsonl" \
    --output-dir "$FILTER_OUT" \
    --judge-backend http_async \
    --judge-base-url "$BASE_URL" \
    --model "$MODEL" \
    --judge-concurrency "$JUDGE_CONCURRENCY" \
    --judge-max-tokens "$JUDGE_MAX_TOKENS" \
    --judge-temperature 0.0

echo "==== [$(date)] STEP 2: valid-record-id mask ===="
uv run python -m prism.datagen.build_valid_record_ids \
    --input-glob "$FILTER_OUT/*.jsonl" \
    --precomputed-dir "$TGT_DIR"

echo "==== [$(date)] PIPELINE COMPLETE ===="
echo "Dataset dir:     $TGT_DIR   (+ valid_record_ids.json mask)"
echo "Filtered JSONLs: $FILTER_OUT"
ls -lh "$FILTER_OUT"/*.jsonl 2>/dev/null | head -20
