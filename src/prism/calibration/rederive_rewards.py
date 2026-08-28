"""rederive_rewards.py — recompute rewards from cached per-bullet scores.

The judge produces two weight-independent lists per record:

    judge.instruction_scores    — one float per GT bullet (recall)
    judge.hallucination_scores  — one float per ITM bullet (precision; 1=halluc)

Only the *scalar* reward and length_penalty depend on the reward knobs
(`instruction_weight`, `hallucination_weight`, `length_penalty_*`). When
those knobs change, we don't need to re-call the LLM judge — we can
recompute the scalar fields from the cached lists in seconds.

Reads ``judge_scores.jsonl``, recomputes ``reward``, ``length_penalty``,
``mean_instruction_score``, and ``mean_hallucination_score`` for each row
under the current ``RL_CONFIG`` knobs, and atomically writes the
file back. The per-bullet lists, gt_instructions, itm_bullets, and
sft_report are preserved unchanged.

Run:
    uv run python -m prism.calibration.rederive_rewards --in judge_scores.jsonl
    uv run python -m prism.calibration.rederive_rewards \\
        --in judge_scores.jsonl --out judge_scores.new.jsonl
    uv run python -m prism.calibration.rederive_rewards \\
        --in judge_scores.jsonl --w-inst 0.7 --w-halluc 0.3 --no-length-penalty
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from prism.rl import judge
from prism.rl.config import RL_CONFIG

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _rederive_one(row: dict, knobs: dict) -> dict:
    """Recompute scalar reward fields on a single judge_scores row.

    Mutates the row's ``judge`` sub-dict in place and returns the row.
    """
    j = row["judge"]
    inst_scores = list(j.get("instruction_scores") or [])
    halluc_scores = list(j.get("hallucination_scores") or [])
    n_gt = len(row.get("gt_instructions") or [])
    n_itm = len(halluc_scores)

    mean_inst = _mean(inst_scores)
    mean_halluc = _mean(halluc_scores)
    length_pen = judge._length_penalty(
        n_report_bullets=n_itm,
        n_gt_bullets=n_gt,
        enabled=knobs["length_penalty_enabled"],
        k=knobs["length_penalty_k"],
        lam=knobs["length_penalty_lambda"],
    )
    reward = (
        knobs["instruction_weight"] * mean_inst
        - knobs["hallucination_weight"] * mean_halluc
        - length_pen
    )
    j["mean_instruction_score"] = mean_inst
    j["mean_hallucination_score"] = mean_halluc
    j["length_penalty"] = length_pen
    j["reward"] = reward
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in", dest="in_path", type=str, required=True,
                   help="judge_scores.jsonl (from score_judge.py)")
    p.add_argument("--out", type=str, default=None,
                   help="Output path (default: overwrite --in atomically)")
    p.add_argument("--w-inst", type=float, default=None)
    p.add_argument("--w-halluc", type=float, default=None)
    p.add_argument("--length-penalty-k", type=float, default=None)
    p.add_argument("--length-penalty-lambda", type=float, default=None)
    p.add_argument("--no-length-penalty", action="store_true",
                   help="Disable the length penalty regardless of config")
    args = p.parse_args()

    knobs = {
        "instruction_weight":
            args.w_inst if args.w_inst is not None else RL_CONFIG["instruction_weight"],
        "hallucination_weight":
            args.w_halluc if args.w_halluc is not None else RL_CONFIG["hallucination_weight"],
        "length_penalty_enabled":
            False if args.no_length_penalty else RL_CONFIG["length_penalty_enabled"],
        "length_penalty_k":
            args.length_penalty_k if args.length_penalty_k is not None
            else RL_CONFIG["length_penalty_k"],
        "length_penalty_lambda":
            args.length_penalty_lambda if args.length_penalty_lambda is not None
            else RL_CONFIG["length_penalty_lambda"],
    }
    logger.info("Knobs: %s", knobs)

    in_path = Path(args.in_path)
    rows = [json.loads(line) for line in in_path.read_text().splitlines() if line.strip()]
    logger.info("Loaded %d rows from %s", len(rows), in_path)

    # Sanity: rows must carry the current-spec fields. If they don't, we
    # cannot rederive — refuse rather than silently produce wrong rewards.
    sample = rows[0]["judge"] if rows else {}
    if "hallucination_scores" not in sample:
        raise SystemExit(
            f"{in_path} is missing per-bullet `hallucination_scores`. "
            "It was produced under the old judge spec — re-run score_judge.py first."
        )

    rewards_before = [r["judge"].get("reward", 0.0) for r in rows]
    for r in rows:
        _rederive_one(r, knobs)
    rewards_after = [r["judge"]["reward"] for r in rows]

    out_path = Path(args.out) if args.out else in_path
    # Atomic overwrite: write to .tmp, fsync, rename.
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp_path.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp_path, out_path)
    logger.info("Wrote %d rederived rows → %s", len(rows), out_path)

    def _summary(name: str, vals: list[float]) -> None:
        if not vals:
            return
        m = sum(vals) / len(vals)
        std = (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5
        logger.info("  %s: mean=%.4f std=%.4f min=%.4f max=%.4f",
                    name, m, std, min(vals), max(vals))

    _summary("reward (before)", rewards_before)
    _summary("reward (after) ", rewards_after)
    deltas = [a - b for a, b in zip(rewards_after, rewards_before)]
    _summary("Δ reward       ", deltas)


if __name__ == "__main__":
    main()
