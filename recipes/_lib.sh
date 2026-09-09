# _lib.sh — shared setup + stage functions for the PRISM training recipes.
#
# Sourced by recipes/<stage>_<model>.sh. Each wrapper sets
#     PROFILE, TAG, HOOK_LAYER   (and optionally SMOKE=1, PRISM_ON_THE_FLY=1)
# then calls one of:  do_precompute / do_sft / do_rl
#
# Pipeline per target model:
#     precompute → SFT (projection + LoRA on frozen activations) → GRPO.
# The gemma-2 / ministral recipes reuse the Qwen-generated records
# verbatim (a deliberate off-manifold test; the instruction_set labels
# are target-model-independent); only activations are re-extracted per model.
#
# Required environment (see .env.example):
#   PRISM_DATA_DIR   root for datasets + precomputed activation shards
#   PRISM_CKPT_DIR   root for training checkpoints
#   PRISM_JUDGE_BASE_URL / PRISM_JUDGE_MODEL   judge endpoint (RL stage only;
#       serve one with scripts/serve_judge.sh)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Load .env WITHOUT overriding variables already set in the environment
# (same precedence as prism.common.env.load_env).
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

# Secrets (HF_TOKEN for gated gemma-2, WANDB_API_KEY) from .env, if present.
# Environment always wins over .env values.
_load_dotenv "$REPO_ROOT/.env"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
# Python launcher — override with PRISM_PYTHON=.venv/bin/python on clusters
# where `uv run` hits NFS flock issues (os error 37).
PY="${PRISM_PYTHON:-uv run python}"

: "${PRISM_DATA_DIR:?set PRISM_DATA_DIR to the root holding datasets + precomputed shards}"
: "${PRISM_CKPT_DIR:?set PRISM_CKPT_DIR to the root for training checkpoints}"

# Directory holding the instruction-set dataset: jsonl/*.jsonl plus
# valid_record_ids.json, as produced by scripts/generate_dataset.sh followed
# by scripts/clean_dataset.sh.
DATASET_SRC="${PRISM_DATASET_SRC:-$PRISM_DATA_DIR}"
PRECOMP_BASE="$PRISM_DATA_DIR/precomputed"
CKPT_BASE="$PRISM_CKPT_DIR"

echo "======================================================================"
echo " host=$(hostname)  date=$(date)  stage=${STAGE:-?}  profile=${PROFILE:-?}  smoke=${SMOKE:-0}"
nvidia-smi -L 2>/dev/null || echo "(no nvidia-smi)"
echo "======================================================================"

_precompute_dir() {
  if [ "${SMOKE:-0}" = "1" ]; then echo "$PRECOMP_BASE/SMOKE-${TAG}";
  else echo "$PRECOMP_BASE/${TAG}"; fi
}
_sft_ckpt_dir() {
  if [ "${SMOKE:-0}" = "1" ]; then echo "$CKPT_BASE/SMOKE-sft-${TAG}";
  else echo "$CKPT_BASE/sft-${TAG}"; fi
}
_rl_ckpt_dir() {
  if [ "${SMOKE:-0}" = "1" ]; then echo "$CKPT_BASE/SMOKE-grpo-${TAG}";
  else echo "$CKPT_BASE/grpo-${TAG}"; fi
}

# ── Activation precompute (run automatically by do_sft / do_rl) ──────────────────────────────────────────────
_require_dataset() {
  if ! compgen -G "$DATASET_SRC/jsonl/*.jsonl" >/dev/null; then
    echo "ERROR: no instruction-set dataset at $DATASET_SRC/jsonl/*.jsonl" >&2
    echo "       Download the released dataset: uv run python scripts/download_dataset.py" >&2
    echo "       (or generate one: scripts/generate_dataset.sh + scripts/clean_dataset.sh," >&2
    echo "       or point PRISM_DATASET_SRC at an existing dataset directory)." >&2
    exit 1
  fi
}

# ON-THE-FLY data arguments (PRISM_ON_THE_FLY=1): the trainers extract
# activations from the resident target model instead of reading the cache.
# Same files, sorted order, and the valid_record_ids.json mask next to jsonl/
# is auto-detected — so the train/val split equals the cache's. SMOKE=1 trains
# on a fresh 170-record slice per file (no mask, like the smoke cache).
_onthefly_args() {
  _require_dataset
  if [ "${SMOKE:-0}" = "1" ]; then
    local SUB="$PRECOMP_BASE/SMOKE-${TAG}-onthefly-subset"; rm -rf "$SUB"; mkdir -p "$SUB"
    for f in "$DATASET_SRC"/jsonl/*.jsonl; do head -n 170 "$f" > "$SUB/$(basename "$f")"; done
    echo "--dataset-paths $(ls "$SUB"/*.jsonl | sort | tr '\n' ' ') --valid-record-ids none"
  else
    echo "--dataset-paths $(ls "$DATASET_SRC"/jsonl/*.jsonl | sort | tr '\n' ' ')"
  fi
}

do_precompute() {
  local OUT SRC_GLOB FILES
  OUT="$(_precompute_dir)"
  export PRISM_TARGET_MODEL="$PROFILE"
  _require_dataset
  if [ "${SMOKE:-0}" = "1" ]; then
    rm -rf "$OUT"                                  # smoke: always fresh
    local SUB="$OUT/_subset"; mkdir -p "$SUB"
    for f in "$DATASET_SRC"/jsonl/*.jsonl; do head -n 170 "$f" > "$SUB/$(basename "$f")"; done
    SRC_GLOB="$SUB"/*.jsonl
  else
    [ -e "$OUT/manifest.json" ] && { echo "REFUSING: $OUT already has a manifest. Delete the directory to re-extract."; exit 1; }
    SRC_GLOB="$DATASET_SRC"/jsonl/*.jsonl
  fi
  # sorted file order → every target model gets an identical split (seed 42)
  FILES=$(ls $SRC_GLOB | sort | tr '\n' ' ')
  # gemma-2 uses eager attention (softcapping) which materialises the full
  # batch×heads×seq² matrix → OOMs on the long-sequence tail at batch 64.
  # Use a smaller batch for gemma; sdpa models (qwen/ministral) stay at 64.
  local DEFBATCH=64; [ "$PROFILE" = "gemma2-9b" ] && DEFBATCH=16
  local BS="${BATCH:-$DEFBATCH}"
  echo "[precompute] profile=$PROFILE layer=$HOOK_LAYER batch=$BS out=$OUT"
  echo "[precompute] files: $FILES"
  $PY -m prism.activations.extract \
    --dataset-paths $FILES \
    --model-profile "$PROFILE" --layers "$HOOK_LAYER" \
    --num-tokens 128 --token-position last --dtype bfloat16 \
    --records-per-shard 256 --batch-size "$BS" \
    --val-ratio 0.1 --test-ratio 0.1 --seed 42 \
    --output-dir "$OUT"
  if [ "${SMOKE:-0}" != "1" ]; then
    if [ -f "$DATASET_SRC/valid_record_ids.json" ]; then
      cp "$DATASET_SRC/valid_record_ids.json" "$OUT/"
      echo "[precompute] copied valid_record_ids.json mask into $OUT"
    else
      echo "WARNING: $DATASET_SRC/valid_record_ids.json not found — training would run on"
      echo "         UNFILTERED records. Run scripts/clean_dataset.sh (its"
      echo "         Step C builds the mask) before precompute." >&2
    fi
  fi
  echo "PRECOMPUTE DONE → $OUT"
}

# Activation cache: computed automatically on first use (idempotent — an
# existing manifest is left untouched). Set PRISM_PRECOMPUTED_DIR to reuse a
# cache extracted elsewhere.
_ensure_precompute() {
  local DIR="${PRISM_PRECOMPUTED_DIR:-$(_precompute_dir)}"
  if [ ! -e "$DIR/manifest.json" ]; then
    echo "[precompute] no activation cache at $DIR — extracting now"
    do_precompute
  fi
}

# ── Stage: SFT (raw activations → Linear projection → LoRA monitor) ───────────
do_sft() {
  export PRISM_TARGET_MODEL="$PROFILE"
  local DATA_ARGS=""
  if [ "${PRISM_ON_THE_FLY:-0}" = "1" ]; then
    DATA_ARGS="$(_onthefly_args)"
    unset PRISM_PRECOMPUTED_DIR
  else
    _ensure_precompute
    export PRISM_PRECOMPUTED_DIR="${PRISM_PRECOMPUTED_DIR:-$(_precompute_dir)}"
  fi
  export PRISM_CHECKPOINT_DIR="$(_sft_ckpt_dir)"
  export PRISM_WANDB_RUN_NAME="sft-${TAG}"
  local EXTRA="${SFT_EXTRA:-}"
  if [ "${SMOKE:-0}" = "1" ]; then
    export PRISM_WANDB_RUN_NAME="SMOKE-sft-${TAG}"
    export PRISM_EVAL_EVERY=5 PRISM_EVAL_SAMPLES=32 PRISM_LOG_EVERY=1
    EXTRA="$EXTRA --epochs 1"
  else
    # Cap eval (precomputed default = the full val split →
    # hours of eval overhead). SFT is just the RL init; best.pt selection
    # stays stable.
    export PRISM_EVAL_SAMPLES="${PRISM_EVAL_SAMPLES:-1500}"
  fi
  echo "[sft] data=${PRISM_PRECOMPUTED_DIR:-on-the-fly ($DATASET_SRC/jsonl)} ckpt=$PRISM_CHECKPOINT_DIR"
  $PY -m prism.sft.train $EXTRA $DATA_ARGS
  echo "SFT DONE → $PRISM_CHECKPOINT_DIR"
}

# ── Stage: GRPO RL (pinned to the paper's Qwen GRPO recipe) ───────────────────
do_rl() {
  export PRISM_TARGET_MODEL="$PROFILE"
  local PRE SFT CKPT RUN STEPS EVAL
  # Data + init follow the SFT stage: the full cache + full SFT checkpoint
  # normally, the SMOKE-* cache + SMOKE SFT checkpoint under SMOKE=1 (so
  # `SMOKE=1 sft_<model>.sh` followed by `SMOKE=1 grpo_<model>.sh` is a
  # self-contained 50-step plumbing check). Override with PRISM_PRECOMPUTED_DIR
  # / PRISM_SFT_INIT_FROM.
  SFT="${PRISM_SFT_INIT_FROM:-$(_sft_ckpt_dir)/best.pt}"
  if [ ! -f "$SFT" ]; then
    echo "ERROR: SFT checkpoint not found: $SFT" >&2
    echo "       Run the SFT recipe for this target model first (same SMOKE setting)," >&2
    echo "       or set PRISM_SFT_INIT_FROM to an SFT checkpoint (e.g. the released prism-<model>-sft.pt)." >&2
    exit 1
  fi
  local DATA_ARGS
  if [ "${PRISM_ON_THE_FLY:-0}" = "1" ]; then
    unset PRISM_PRECOMPUTED_DIR
    DATA_ARGS="$(_onthefly_args)"; PRE="on-the-fly ($DATASET_SRC/jsonl)"
  else
    _ensure_precompute
    PRE="${PRISM_PRECOMPUTED_DIR:-$(_precompute_dir)}"
    DATA_ARGS="--precomputed-dir $PRE"
  fi
  CKPT="$(_rl_ckpt_dir)"
  # Judge endpoint: optionally sourced from a file (useful when the judge node
  # IP changes per scheduling), else taken from the environment.
  [ -n "${PRISM_JUDGE_ENDPOINT_FILE:-}" ] && [ -f "$PRISM_JUDGE_ENDPOINT_FILE" ] && source "$PRISM_JUDGE_ENDPOINT_FILE"
  export PRISM_JUDGE_MODEL="${PRISM_JUDGE_MODEL:-gemma4-31B-it}"
  export PRISM_JUDGE_API_KEY="${PRISM_JUDGE_API_KEY:-not-needed}"
  : "${PRISM_JUDGE_BASE_URL:?set PRISM_JUDGE_BASE_URL to the judge endpoint (http://<host>:8088/v1— see scripts/serve_judge.sh)}"
  # Must be exported explicitly: the endpoint file may use plain assignments,
  # which set a shell var (enough to pass the guard above) that child python
  # processes never see — the judge client then silently falls back to
  # api.openai.com and every reward call 401s.
  export PRISM_JUDGE_BASE_URL
  if [ "${SMOKE:-0}" = "1" ]; then STEPS=50; EVAL=25; else STEPS=20000; EVAL=200; fi
  # Optional overrides (used by recipes/grpo_seed_qwen3.5-9b.sh): run name /
  # ckpt dir / step cap / group size.
  RUN="${RL_RUN_NAME:-grpo-${TAG}}"
  [ "${SMOKE:-0}" = "1" ] && RUN="SMOKE-${RUN#SMOKE-}"   # smoke runs never share a W&B name with a real run
  STEPS="${RL_STEPS:-$STEPS}"
  CKPT="${RL_CKPT_DIR:-$CKPT}"
  local NCAND="${RL_NCAND:-6}"
  # Prioritized sampling is part of every released recipe; RL_PRIO=0 disables it.
  local PRIO_ARG="--prioritized-sampling"; [ "${RL_PRIO:-1}" = "0" ] && PRIO_ARG=""
  local RESUME_ARG=""
  if [ -n "${RESUME_FROM:-}" ]; then
    if [ "${RESUME_STRIP_WANDB:-0}" = "1" ]; then
      # Continuations log to a FRESH W&B run: drop the embedded wandb_run_id
      # (train.py would otherwise reattach to the finished source run and
      # collide with its already-logged steps).
      mkdir -p "$CKPT"
      local STRIPPED="$CKPT/resume_init_$(basename "$RESUME_FROM")"
      RESUME_SRC="$RESUME_FROM" RESUME_DST="$STRIPPED" $PY - <<'PY'
import os, torch
src, dst = os.environ["RESUME_SRC"], os.environ["RESUME_DST"]
ck = torch.load(src, map_location="cpu", weights_only=False)
ck.pop("wandb_run_id", None)
torch.save(ck, dst)
print(f"[rl] stripped wandb_run_id: {src} -> {dst} (opt_step={ck.get('opt_step')})")
PY
      RESUME_FROM="$STRIPPED"
    fi
    RESUME_ARG="--resume $RESUME_FROM"
  fi
  echo "[rl] sft=$SFT pre=$PRE ckpt=$CKPT judge=$PRISM_JUDGE_MODEL @ $PRISM_JUDGE_BASE_URL resume=${RESUME_FROM:-none}"
  # Overrides vs config defaults = exactly the params the released runs changed
  # (lr, n_candidates, kl_estimator, gen_max_new_tokens); everything else
  # (instruction_weight 1.0, gen_temperature 1.2, projection_lr 5e-6, length +
  # prioritization knobs, kl_coef 0.05, eval_samples 500, save_every 50) is the
  # pinned config default. Reward/judge code untouched for comparability.
  $PY -m prism.rl.train \
    --sft-init-from "$SFT" $DATA_ARGS \
    --checkpoint-dir "$CKPT" --wandb-run-name "$RUN" \
    --lr 2e-5 --n-candidates "$NCAND" --kl-estimator k3 --kl-coef 0.05 \
    --gen-max-new-tokens 144 --max-opt-steps "$STEPS" --eval-every "$EVAL" \
    $PRIO_ARG $RESUME_ARG
  echo "RL DONE → $CKPT"
}
