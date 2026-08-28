"""score_judge.py — run the LLM judge on the SFT calibration reports.

Reads sft_reports.jsonl, scores each (prompt_a, response_a, gt_bullets,
sft_report) tuple with `judge.batch_score`, writes per-record
judge labels to judge_scores.jsonl.

Each output row:
    {"record_id", "global_idx", "source_dataset", "phase",
     "gt_instructions": [...],
     "itm_bullets": [...],               # pre-split ITM report bullets (same numbering judge saw)
     "judge": {"instruction_scores": [float, ...],       # one per GT bullet, higher=better
               "hallucination_scores": [float, ...],      # one per ITM bullet, higher=worse
               "mean_instruction_score": float,
               "mean_hallucination_score": float,
               "length_penalty": float,
               "reward": float}}

Run:
    uv run python -m prism.calibration.score_judge --reports sft_reports.jsonl
    uv run python -m prism.calibration.score_judge --reports sft_reports.jsonl --workers 16
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

from prism.common.env import load_env
from prism.rl import judge
from prism.rl.config import RL_CONFIG

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reports", type=str, required=True,
                   help="sft_reports.jsonl (from generate_reports.py / gold_to_jsonl.py / "
                        "generate_reports.py)")
    p.add_argument("--out", type=str, default=None,
                   help="Output JSONL path (default: judge_scores.jsonl next to --reports)")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--max-retries", type=int, default=None)
    p.add_argument("--judge-model", type=str, default=None)
    args = p.parse_args()

    load_env()
    cfg = dict(RL_CONFIG)
    if args.judge_model is not None:
        cfg["judge_model"] = args.judge_model
    elif cfg.get("judge_model", "MODEL_NOT_SET") == "MODEL_NOT_SET":
        cfg["judge_model"] = os.environ.get("PRISM_JUDGE_MODEL") or ""
    if not cfg["judge_model"]:
        raise SystemExit("judge_model not set. Export PRISM_JUDGE_MODEL or pass --judge-model.")
    workers = args.workers if args.workers is not None else cfg["judge_workers"]
    max_retries = args.max_retries if args.max_retries is not None else cfg["judge_max_retries"]

    reports_path = Path(args.reports)
    rows: list[dict] = [json.loads(line) for line in reports_path.read_text().splitlines() if line.strip()]
    logger.info("Loaded %d reports from %s", len(rows), reports_path)

    flat_p = [r["prompt_a"] for r in rows]
    flat_resp_a = [r["response_a"] for r in rows]
    flat_gt = [judge.split_instructions(r["response_b"]) for r in rows]
    flat_cand = [r["sft_report"] for r in rows]
    flat_itm = [judge.split_instructions(r["sft_report"]) for r in rows]
    n_with_gt = sum(1 for g in flat_gt if g)
    avg_gt = sum(len(g) for g in flat_gt) / max(1, len(flat_gt))
    avg_itm = sum(len(b) for b in flat_itm) / max(1, len(flat_itm))
    logger.info("Judge inputs: %d/%d have parsed GT bullets; avg %.1f GT claims, %.1f ITM bullets per record",
                n_with_gt, len(rows), avg_gt, avg_itm)
    logger.info(
        "Reward weights: w_inst=%.2f w_halluc=%.2f  | length penalty: enabled=%s k=%.2f λ=%.3f",
        cfg["instruction_weight"], cfg["hallucination_weight"],
        cfg["length_penalty_enabled"], cfg["length_penalty_k"], cfg["length_penalty_lambda"],
    )
    logger.info("Judging with model=%s workers=%d retries=%d", cfg["judge_model"], workers, max_retries)

    client = judge._make_client()
    t0 = time.time()
    scored = judge.batch_score(
        prompts_a=flat_p, responses_a=flat_resp_a,
        gt_instructions=flat_gt, candidates=flat_cand,
        model=cfg["judge_model"],
        instruction_weight=cfg["instruction_weight"],
        hallucination_weight=cfg["hallucination_weight"],
        length_penalty_enabled=cfg["length_penalty_enabled"],
        length_penalty_k=cfg["length_penalty_k"],
        length_penalty_lambda=cfg["length_penalty_lambda"],
        workers=workers, max_retries=max_retries, client=client,
    )
    dt = time.time() - t0
    logger.info("Judge took %.1fs (%.2fs/record, single LLM call per record)",
                dt, dt / max(1, len(rows)))

    out_path = Path(args.out) if args.out else reports_path.parent / "judge_scores.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        for row, gt, itm, s in zip(rows, flat_gt, flat_itm, scored):
            out = {
                "record_id": row["record_id"],
                "global_idx": row["global_idx"],
                "source_dataset": row["source_dataset"],
                "phase": row["phase"],
                "gt_instructions": gt,
                "itm_bullets": itm,                  # numbered list judge scored
                "sft_report": row["sft_report"],     # echo raw for traceability
                "judge": {
                    "instruction_scores": s.instruction_scores,
                    "hallucination_scores": s.hallucination_scores,
                    "mean_instruction_score": s.mean_instruction_score,
                    "mean_hallucination_score": s.mean_hallucination_score,
                    "length_penalty": s.length_penalty,
                    "reward": s.reward,
                },
            }
            fh.write(json.dumps(out) + "\n")
    logger.info("Wrote %d scored rows → %s", len(rows), out_path)

    # Summary
    rewards = [s.reward for s in scored]
    mean_inst = [s.mean_instruction_score for s in scored]
    mean_halluc = [s.mean_hallucination_score for s in scored]
    length_pens = [s.length_penalty for s in scored]
    r_mean = sum(rewards) / len(rewards)
    logger.info(
        "Reward: mean=%.3f std=%.3f min=%.3f max=%.3f",
        r_mean,
        (sum((r - r_mean) ** 2 for r in rewards) / len(rewards)) ** 0.5,
        min(rewards), max(rewards),
    )
    n_pen = sum(1 for lp in length_pens if lp > 0)
    logger.info(
        "Mean per-bullet:  instruction=%.3f  hallucination=%.3f  | length penalty: avg=%.3f hit=%d/%d",
        sum(mean_inst) / len(mean_inst),
        sum(mean_halluc) / len(mean_halluc),
        sum(length_pens) / len(length_pens),
        n_pen, len(scored),
    )


if __name__ == "__main__":
    main()
