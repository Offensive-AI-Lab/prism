#!/usr/bin/env python
"""Download the released PRISM training dataset into the recipes' layout.

Fetches the three source JSONLs and the valid_record_ids.json mask from
Hugging Face, verifies every SHA-256, and lays them out as the recipes
expect:

    <dest>/jsonl/{if_eval,if_multi_constraints,ultrachat}.jsonl
    <dest>/valid_record_ids.json

Usage:
    uv run python scripts/download_dataset.py                  # dest = $PRISM_DATA_DIR
    uv run python scripts/download_dataset.py --dest /data/prism-training-data

After this, every recipe trains directly (the activation cache is built
automatically on first run); scripts/check_dataset.py re-validates the
records, counts and split membership at any time.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

REPO_ID = os.environ.get("PRISM_DATASET_REPO", "Offensive-AI-Lab/prism-training-dataset")

FILES = {
    "if_eval.jsonl": ("jsonl", "8ff20c399072433afa04b13c82358f5f3eff3f784a0215b048233bdcd54417e1"),
    "if_multi_constraints.jsonl": ("jsonl", "d24ed6fa26852174d79d96d1cf968c8f4e2a0b0c47eef04d59f4061c48669a9f"),
    "ultrachat.jsonl": ("jsonl", "f74eef7afb15504996aaad073fb800dd36e1293f1b952e14009f8ef7a1e40248"),
    "valid_record_ids.json": ("", "66202284fffb28e2e51401ddbb4230da916dfda6033c677645de0fa78703314a"),
}


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dest", default=None,
                    help="Target dataset directory (default: $PRISM_DATA_DIR)")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"),
                    help="Hugging Face access token (default: $HF_TOKEN)")
    args = ap.parse_args()
    dest = Path(args.dest) if args.dest else None
    if dest is None:
        data_dir = os.environ.get("PRISM_DATA_DIR")
        if not data_dir:
            print("Set PRISM_DATA_DIR (e.g. export PRISM_DATA_DIR=./prism-data) or pass --dest.", file=sys.stderr)
            return 1
        dest = Path(data_dir)

    from huggingface_hub import hf_hub_download

    for fname, (subdir, digest) in FILES.items():
        target_dir = dest / subdir if subdir else dest
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / fname
        if not target.exists():
            print(f"downloading {REPO_ID}/{fname} -> {target}")
            hf_hub_download(repo_id=REPO_ID, filename=fname, repo_type="dataset", token=args.token,
                            local_dir=str(target_dir))
        got = sha256_of(target)
        if got != digest:
            print(f"ERROR: {target} has SHA-256 {got}, expected {digest}. "
                  f"Delete it and re-run.", file=sys.stderr)
            return 1
        print(f"verified {target}")
    print(f"\nDataset ready at {dest} — the recipes will pick it up via PRISM_DATA_DIR.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
