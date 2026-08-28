"""gold_to_jsonl.py — convert the gold xlsx into the judge's input schema.

The 49-record gold set lives in ``pilot_gold.xlsx`` (sheet
``Gold Dataset(49)``) — shipped with the calibration data in the
prism-eval repo (https://github.com/Offensive-AI-Lab/prism-eval). Each
row has both the inputs the judge needs and the human-agreed scores for
κ comparison. This script splits those into:

  ``sft_reports.jsonl``   — judge input, same schema as
                            ``generate_reports.py`` output. Feeds
                            ``score_judge.py``.
  ``gold_labels.jsonl``   — human-agreed scores, keyed on record_id.
                            Feeds ``score_against_gold.py``.

Run::

    uv run --with openpyxl python -m prism.calibration.gold_to_jsonl \\
        --xlsx /path/to/pilot_gold.xlsx
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

GOLD_SHEET = "Gold Dataset(49)"


def _parse_csv_floats(s: object) -> list[float]:
    """Parse '1,0,0.5,1' → [1.0, 0.0, 0.5, 1.0]. Scores are 0/0.5/1."""
    if pd.isna(s):
        return []
    return [float(x.strip()) for x in str(s).split(",") if x.strip()]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--xlsx", required=True,
                   help="Gold workbook (pilot_gold.xlsx) — ships with the calibration "
                        "data in the prism-eval repo.")
    p.add_argument("--sheet", default=GOLD_SHEET)
    p.add_argument("--reports-out", default="sft_reports.jsonl")
    p.add_argument("--labels-out", default="gold_labels.jsonl")
    args = p.parse_args()

    df = pd.read_excel(args.xlsx, sheet_name=args.sheet)
    df = df.sort_values("record_id").reset_index(drop=True)
    logger.info("Loaded %d rows from %s :: %s", len(df), args.xlsx, args.sheet)

    reports_path = Path(args.reports_out)
    labels_path = Path(args.labels_out)
    reports_path.parent.mkdir(parents=True, exist_ok=True)

    n_empty_report = 0
    n_empty_gt = 0
    with reports_path.open("w") as rfh, labels_path.open("w") as lfh:
        for idx, row in df.iterrows():
            sft_report = str(row["itm_report"] or "")
            instructions = str(row["instructions"] or "")
            if not sft_report.strip():
                n_empty_report += 1
            if not instructions.strip():
                n_empty_gt += 1

            report = {
                "record_id": row["record_id"],
                "global_idx": int(idx),
                "source_dataset": row["source_dataset"],
                "phase": "pilot",
                "prompt_a": str(row["prompt"] or ""),
                "response_a": str(row["model_response"] or ""),
                # split_instructions accepts "1. foo\n2. bar" via _BULLET_RE
                "response_b": instructions,
                "sft_report": sft_report,
            }
            rfh.write(json.dumps(report) + "\n")

            inst_scores = _parse_csv_floats(row["instruction_scores"])
            halluc_scores = _parse_csv_floats(row["hallucination_scores"])
            label = {
                "record_id": row["record_id"],
                "global_idx": int(idx),
                "source_dataset": row["source_dataset"],
                "instruction_scores": inst_scores,
                "hallucination_scores": halluc_scores,
            }
            lfh.write(json.dumps(label) + "\n")

    logger.info("Wrote %d reports → %s", len(df), reports_path.resolve())
    logger.info("Wrote %d labels  → %s", len(df), labels_path.resolve())
    if n_empty_report:
        logger.warning("  %d rows had empty itm_report", n_empty_report)
    if n_empty_gt:
        logger.warning("  %d rows had empty instructions", n_empty_gt)


if __name__ == "__main__":
    main()
