# Judge calibration — runbook

`prism.calibration` validates that the LLM judge used for the GRPO reward
(`prism.rl.judge`) agrees with human annotators, following the protocol
in docs/RUBRIC.md: annotate a pilot → measure agreement (κ) → inspect
disagreements → revise the rubric → re-validate → regression-check.

**The human gold data is not shipped here** — it lives with
[prism-eval](https://github.com/Offensive-AI-Lab/prism-eval) under
`data/calibration/` (already converted to JSONL, e.g. `coverage_gold_v1.jsonl`,
`advdet_gold_v1.jsonl`) — usable directly as `--gold` below;
`gold_to_jsonl` is only needed to convert your **own** annotation workbooks.
Every tool takes explicit paths, so you can also run the loop on your own
annotations. Install analysis deps with `uv sync --extra calibration`.

## Core loop: judge vs gold

```bash
# 1. Sample calibration records from a precomputed activation dataset
uv run python -m prism.calibration.sample \
    --precomputed-dir $PRISM_DATA_DIR/precomputed/<dir> --out-dir calib/

# 2. Generate monitor outputs for those records (checkpoint + activations via env)
PRISM_SFT_INIT_FROM=$PRISM_CKPT_DIR/<run>/best.pt \
PRISM_PRECOMPUTED_DIR=$PRISM_DATA_DIR/precomputed/<dir> \
uv run python -m prism.calibration.generate_reports --in-dir calib/

# 3. Score the reports with the judge (needs PRISM_JUDGE_* env)
uv run python -m prism.calibration.score_judge --reports calib/sft_reports.jsonl

# 4. Convert an adjudicated gold workbook to JSONL and score judge vs human
uv run python -m prism.calibration.gold_to_jsonl --xlsx <gold.xlsx>
uv run python -m prism.calibration.score_against_gold \
    --judge calib/judge_scores.jsonl --gold gold.jsonl
```

`score_against_gold` reports per-axis agreement (unweighted / linear /
quadratic κ), bias vs human, and confusion matrices. The paper's gates:
coverage κ_quad ≥ 0.78 with |bias| ≤ 0.03; hallucination κ_quad ≥ 0.74.
**κ alone is insufficient** — quadratic weighting barely penalizes 1↔0.5
flips, so always check the bias and the 1→0.5 downgrade count too.

## Reward re-derivation

After changing reward *weights* (not judge outputs), recompute rewards from
an existing `judge_scores.jsonl` without any LLM calls:

```bash
uv run python -m prism.calibration.rederive_rewards --in calib/judge_scores.jsonl
```

## Running your own annotation rounds

This repo ships no annotation-queue tooling — annotate with whatever stack
you prefer (the paper used a W&B Weave annotation queue), adjudicate into a
gold workbook, convert with `gold_to_jsonl`, and score with
`score_against_gold`. Deeper agreement analyses (ICC, per-annotator κ) live
with the eval-side calibration in prism-eval.

## Canonical rubric

The prompt in `prism/rl/judge.py` (`SYSTEM_PROMPT`) produced the GRPO reward —
the same prompt prism-eval ships as its canonical scoring-judge prompt.
docs/RUBRIC.md is the full scoring rubric.
