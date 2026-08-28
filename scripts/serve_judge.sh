#!/usr/bin/env bash
# serve_judge.sh — launch a vLLM OpenAI-compatible server for the LLM judge
# used by the GRPO reward (default: a Gemma-4-31B-it judge, reasoning OFF at
# inference — the rubric is executed as a plain instruction-following task).
#
# Run it on a GPU node, note the endpoint it prints, then point training at it:
#     export PRISM_JUDGE_BASE_URL=http://<node-ip>:<port>/v1
#     export PRISM_JUDGE_MODEL=gemma4-31B-it

set -euo pipefail

# ── Config (override via env) ─────────────────────────────────────────────
# Local dir or HF repo id of the judge model (the released runs used Gemma 4 31B-it).
MODEL_PATH="${PRISM_JUDGE_MODEL_PATH:-google/gemma-4-31B-it}"
SERVED_NAME="${PRISM_JUDGE_SERVED_NAME:-gemma4-31B-it}"
PORT="${PRISM_JUDGE_PORT:-8088}"
MAX_MODEL_LEN="${PRISM_JUDGE_MAX_MODEL_LEN:-32384}"
REASONING_PARSER="${PRISM_JUDGE_REASONING_PARSER:-gemma4}"
# Dedicated, persistent vLLM env (kept outside the project venv: vLLM pins its
# own torch/CUDA stack).
SERVE_VENV="${SERVE_VENV:-$HOME/.venv-vllm}"

# Put the serving venv's bin first so helper binaries (e.g. `ninja`, needed by
# FlashInfer's JIT) resolve — we exec vllm by absolute path without activating.
export PATH="$SERVE_VENV/bin:$PATH"
# Disable FlashInfer's sampler: it JIT-compiles a CUDA kernel at startup (needs
# ninja + nvcc). The native torch sampler is identical for a temp-0 greedy judge
# and removes a fragile per-node build dependency. Override by exporting =1.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

# ── Ensure vLLM is available in a dedicated uv env (installs once) ────────
if [[ ! -x "$SERVE_VENV/bin/vllm" ]]; then
  echo "[serve_judge] vLLM not found in $SERVE_VENV — creating env and installing..."
  # Python 3.12: pick a version with vLLM wheels available.
  uv venv --clear --python 3.12 "$SERVE_VENV"
  # Two indexes: PyPI for vLLM itself + PyTorch CUDA wheels for torch.
  # unsafe-best-match is REQUIRED: the pytorch index also lists `vllm`, so uv's
  # default dependency-confusion guard would pin vllm to pytorch's stale set and
  # report "no versions of vllm". This strategy lets it pick vllm from PyPI.
  VIRTUAL_ENV="$SERVE_VENV" uv pip install vllm \
    --index-url https://pypi.org/simple \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    --index-strategy unsafe-best-match
fi

# ── Report the reachable endpoint (IP changes per node — read this!) ──────
NODE_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
echo "======================================================================"
echo " Serving:      $MODEL_PATH"
echo " Served name:  $SERVED_NAME"
echo " Node:         $(hostname)   IP: ${NODE_IP:-unknown}"
echo " Endpoint:     http://${NODE_IP:-<node-ip>}:${PORT}/v1"
echo ""
echo "   export PRISM_JUDGE_BASE_URL=http://${NODE_IP:-<node-ip>}:${PORT}/v1"
echo "======================================================================"

# ── Launch ───────────────────────────────────────────────────────────────
exec "$SERVE_VENV/bin/vllm" serve "$MODEL_PATH" \
  --served-model-name "$SERVED_NAME" \
  --port "$PORT" \
  --max-model-len "$MAX_MODEL_LEN" \
  --enable-prefix-caching \
  --reasoning-parser "$REASONING_PARSER"
