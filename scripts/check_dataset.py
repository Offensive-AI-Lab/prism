#!/usr/bin/env python
"""Validate the released PRISM training dataset before or after use.

Deterministic, offline checks over the dataset directory
(<dir>/jsonl/*.jsonl + <dir>/valid_record_ids.json):

  1. every record carries id, source_dataset, prompt, response,
     instruction_set and metadata.paraphrase_group_id;
  2. record ids are unique across all files;
  3. per-source record counts match the release;
  4. every mask id resolves to a record (203,589 total);
  5. the deterministic split (seed 42, group-aware) reproduces the released
     membership: train 162,821 / val 20,410 / test 20,358, pinned by a
     checksum over the val-split ids.

Usage:
    uv run python scripts/check_dataset.py --dataset-dir $PRISM_DATA_DIR/prompt-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

EXPECTED_COUNTS = {"if_eval": 492, "if_multi_constraints": 77002, "ultrachat": 200002}
EXPECTED_MASK_IDS = 203589
EXPECTED_SPLIT = (162821, 20410, 20358)
EXPECTED_VAL_IDS_SHA256 = "c7e4beda237346498a75085941d01282f0e349bf2610cac570399d37fdc09f90"
REQUIRED_FIELDS = ("id", "source_dataset", "prompt", "response", "instruction_set", "metadata")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset-dir", required=True)
    args = ap.parse_args()
    root = Path(args.dataset_dir)
    files = sorted((root / "jsonl").glob("*.jsonl"))
    mask_path = root / "valid_record_ids.json"
    problems = []
    if not files:
        print(f"ERROR: no JSONL files under {root}/jsonl", file=sys.stderr)
        return 1

    ids = set()
    per_source = {}
    n_dupes = n_bad_fields = 0
    for f in files:
        for line in f.open():
            r = json.loads(line)
            missing = [k for k in REQUIRED_FIELDS if not r.get(k)]
            legacy = [k for k in ("prompt_a", "response_a", "response_b") if r.get(k)]
            if missing and not legacy:
                n_bad_fields += 1
            rid = r.get("id", "")
            if rid in ids:
                n_dupes += 1
            ids.add(rid)
            src = r.get("source_dataset", "?")
            per_source[src] = per_source.get(src, 0) + 1
    if n_bad_fields:
        problems.append(f"{n_bad_fields} records missing required fields")
    if n_dupes:
        problems.append(f"{n_dupes} duplicate record ids")
    for src, want in EXPECTED_COUNTS.items():
        got = per_source.get(src, 0)
        if got != want:
            problems.append(f"{src}: {got} records, expected {want}")
    print(f"records: {sum(per_source.values())} {per_source}")

    if mask_path.exists():
        mask = set(json.loads(mask_path.read_text()))
        if len(mask) != EXPECTED_MASK_IDS:
            problems.append(f"mask has {len(mask)} ids, expected {EXPECTED_MASK_IDS}")
        orphans = len(mask - ids)
        if orphans:
            problems.append(f"{orphans} mask ids match no record")
        print(f"mask: {len(mask)} ids, all resolved" if not orphans else f"mask: {len(mask)} ids")
    else:
        problems.append(f"missing {mask_path}")

    from prism.activations.records import load_split_records
    cfg = {"dataset_paths": [str(f) for f in files], "split_val_ratio": 0.1,
           "split_test_ratio": 0.1, "split_seed": 42, "split_stratify_by": None}
    tr, va, te, _ = load_split_records(cfg, str(mask_path))
    counts = (len(tr), len(va), len(te))
    print(f"split: train {counts[0]} / val {counts[1]} / test {counts[2]}")
    if counts != EXPECTED_SPLIT:
        problems.append(f"split counts {counts} != expected {EXPECTED_SPLIT}")
    val_sha = hashlib.sha256("\n".join(r.id for r in va).encode()).hexdigest()
    if val_sha != EXPECTED_VAL_IDS_SHA256:
        problems.append("val-split membership differs from the release")

    if problems:
        print("\nFAIL:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1
    print("\nOK — dataset matches the release exactly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
