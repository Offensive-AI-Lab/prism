"""score_against_gold.py — judge κ vs gold labels (no Weave required).

Use this when human labels live in ``gold_labels.jsonl`` (emitted by
``gold_to_jsonl.py``) instead of Weave.

Joins judge predictions (``judge_scores.jsonl``) to gold labels
(``gold_labels.jsonl``) by ``(record_id, bullet_index)`` and computes:

  - Cohen's κ (unweighted + linear-weighted)
  - Spearman rank correlation
  - Agreement %
  - 3×3 confusion matrix

…per axis (``instruction`` and ``hallucination``). Acceptance gate per the
calibration runbook: weighted κ ≥ 0.6 → ship the judge into GRPO.

Run::

    uv run python -m prism.calibration.score_against_gold \\
        --judge judge_scores.jsonl --gold gold_labels.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

try:
    from sklearn.metrics import cohen_kappa_score, confusion_matrix as _cm
    from scipy.stats import spearmanr
except ImportError as _e:  # pragma: no cover
    raise ImportError(
        f"{_e}. The calibration tools need the optional extra: `uv sync --extra calibration`.",
        name=_e.name,
    ) from _e

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def _discretize(score: float) -> str:
    """Map a continuous [0,1] score to one of three buckets matching how
    annotators see the rubric: not / partial / fully present. Bin edges
    chosen so the 0.0 / 0.5 / 1.0 rubric values land exactly in each bucket.
    """
    if score < 0.25:
        return "0"
    if score < 0.75:
        return "0.5"
    return "1"


def _pair_axis(
    judge_rows: list[dict],
    gold_rows: list[dict],
    axis: str,
) -> tuple[list[str], list[str], list[float], list[float], int]:
    """Pair judge ↔ gold scores by (record_id, bullet_index) for one axis.

    Returns (human_cats, judge_cats, human_floats, judge_floats, n_evals).
    """
    if axis == "instruction":
        gold_key = "instruction_scores"
        judge_key = "instruction_scores"
    elif axis == "hallucination":
        gold_key = "hallucination_scores"
        judge_key = "hallucination_scores"
    else:
        raise ValueError(f"Unknown axis: {axis!r}")

    judge_by_id = {r["record_id"]: r for r in judge_rows}

    h_cats: list[str] = []
    j_cats: list[str] = []
    h_floats: list[float] = []
    j_floats: list[float] = []
    n_evals = 0
    n_length_mismatch = 0

    for g in gold_rows:
        rid = g["record_id"]
        j = judge_by_id.get(rid)
        if j is None:
            continue
        g_scores = g.get(gold_key) or []
        j_scores = (j.get("judge") or {}).get(judge_key) or []
        if not g_scores or not j_scores:
            continue
        if len(g_scores) != len(j_scores):
            n_length_mismatch += 1
        n_eval_pairs = min(len(g_scores), len(j_scores))
        if n_eval_pairs == 0:
            continue
        for i in range(n_eval_pairs):
            h_floats.append(float(g_scores[i]))
            j_floats.append(float(j_scores[i]))
            h_cats.append(_discretize(float(g_scores[i])))
            j_cats.append(_discretize(float(j_scores[i])))
        n_evals += 1

    if n_length_mismatch:
        logger.warning(
            "[%s] %d records had judge/gold length mismatch — pairing to shorter side",
            axis, n_length_mismatch,
        )

    return h_cats, j_cats, h_floats, j_floats, n_evals


def _axis_report(
    judge_rows: list[dict],
    gold_rows: list[dict],
    axis: str,
) -> dict:
    h_cats, j_cats, h_floats, j_floats, n_evals = _pair_axis(judge_rows, gold_rows, axis)
    if not h_cats:
        return {"error": f"no matching {axis} claims between judge and gold"}

    labels = ["0", "0.5", "1"]
    kappa = cohen_kappa_score(h_cats, j_cats, labels=labels)
    kappa_linear = cohen_kappa_score(h_cats, j_cats, labels=labels, weights="linear")
    kappa_quadratic = cohen_kappa_score(h_cats, j_cats, labels=labels, weights="quadratic")
    if len(set(h_floats)) > 1 and len(set(j_floats)) > 1:
        rho, _ = spearmanr(h_floats, j_floats)
        rho = float(rho)
    else:
        rho = float("nan")  # spearman undefined when one side is constant
    cm = _cm(h_cats, j_cats, labels=labels).tolist()
    agreement = sum(1 for h, j in zip(h_cats, j_cats) if h == j) / len(h_cats)

    return {
        "n_claims": len(h_cats),
        "n_evals": n_evals,
        "cohens_kappa": round(float(kappa), 4),
        "cohens_kappa_linear_weighted": round(float(kappa_linear), 4),
        "cohens_kappa_quadratic_weighted": round(float(kappa_quadratic), 4),
        "spearman_correlation": round(rho, 4) if rho == rho else None,
        "agreement_pct": round(agreement * 100, 1),
        "confusion_matrix": {
            "labels": labels,
            "rows_are": "human (gold)",
            "cols_are": "judge",
            "matrix": cm,
        },
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--judge", required=True,
                   help="judge_scores.jsonl (from score_judge.py)")
    p.add_argument("--gold", required=True,
                   help="gold_labels.jsonl (from gold_to_jsonl.py; gold data ships with "
                        "the prism-eval repo's calibration data)")
    p.add_argument("--out", default="calibration_report.json")
    p.add_argument("--gate", type=float, default=0.6,
                   help="Minimum linear-weighted κ on instruction axis to pass (exits 2 otherwise).")
    args = p.parse_args()

    judge_rows = [json.loads(line) for line in Path(args.judge).read_text().splitlines() if line.strip()]
    gold_rows = [json.loads(line) for line in Path(args.gold).read_text().splitlines() if line.strip()]
    logger.info("Loaded %d judge rows, %d gold rows", len(judge_rows), len(gold_rows))

    report = {
        "n_judge_records": len(judge_rows),
        "n_gold_records": len(gold_rows),
        "axes": {
            "instruction": _axis_report(judge_rows, gold_rows, "instruction"),
            "hallucination": _axis_report(judge_rows, gold_rows, "hallucination"),
        },
    }

    out_path = Path(args.out)
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    logger.info("Wrote → %s", out_path.resolve())

    for axis, block in report["axes"].items():
        if "error" in block:
            logger.warning("[%s] %s", axis, block["error"])
            continue
        logger.info(
            "[%s] n_claims=%d n_evals=%d κ=%.3f κ_lin=%.3f κ_quad=%.3f agree=%.1f%% ρ=%s",
            axis, block["n_claims"], block["n_evals"],
            block["cohens_kappa"], block["cohens_kappa_linear_weighted"],
            block["cohens_kappa_quadratic_weighted"],
            block["agreement_pct"],
            f"{block['spearman_correlation']:.3f}" if block["spearman_correlation"] is not None else "n/a",
        )

    inst = report["axes"]["instruction"]
    if "cohens_kappa_linear_weighted" in inst and inst["cohens_kappa_linear_weighted"] < args.gate:
        logger.error("Instruction κ_linear=%.3f below gate %.2f", inst["cohens_kappa_linear_weighted"], args.gate)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
