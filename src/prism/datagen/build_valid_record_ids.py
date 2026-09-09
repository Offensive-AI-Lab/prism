"""
build_valid_record_ids.py — produce a valid_record_ids.json mask for a
precomputed_dir.

Walks the filter's kept-output JSONLs and computes a set of record_ids that pass
the additional quality / shape rules layered on top of the LLM-judge filter:

  1. drop template-leak records — instruction_set contains signatures from
     RETRIEVAL_PROMPT itself (judge let some of these through)
  2. drop word-fragmentation records — instruction_set is a long list of
     1-word bullets (model split prompt into single-token bullets)
  3. drop records whose instruction_set has more than ``max_bullets`` bullets

The output is a JSON file containing a list of record_id strings. The
training data loader reads this file (if present) and pre-filters its
index, so neither the activation safetensors nor the meta.json files need
to be touched.

Usage:
    python -m prism.datagen.build_valid_record_ids \
        --input-glob '$PRISM_DATA_DIR/filtered/*.jsonl' \
        --precomputed-dir $PRISM_DATA_DIR \
        --max-bullets 6
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import re
from collections import Counter
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("build_valid_record_ids")


_BULLET_LINE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+(.*?)\s*$", re.MULTILINE)

_TEMPLATE_LEAK_PATTERNS = [
    re.compile(r"do not (carry out|demonstrate|comply with).{0,100}instruction", re.IGNORECASE),
    re.compile(r"enumerate the instructions and explicit constraints", re.IGNORECASE),
    re.compile(r"output only the bullet list", re.IGNORECASE),
    re.compile(r"begin (the reply|your reply) with", re.IGNORECASE),
    re.compile(r"<<<MESSAGE_(START|END)>>>"),
]


def extract_bullets(text: str) -> list[str]:
    return _BULLET_LINE.findall(text or "")


def is_template_leak(instruction_set: str) -> bool:
    return any(p.search(instruction_set) for p in _TEMPLATE_LEAK_PATTERNS)


def is_word_fragmentation(bullets: list[str]) -> bool:
    """Heuristic: ≥5 bullets where the vast majority are ≤2 words.

    Catches the failure where the model splits the prompt into per-word
    bullets ("- Test", "- Your", "- ENTIRE", ...)."""
    if len(bullets) < 5:
        return False
    word_counts = [len(b.split()) for b in bullets]
    short_count = sum(1 for w in word_counts if w <= 2)
    return short_count >= 0.7 * len(bullets)


def evaluate(rec: dict, max_bullets: int) -> tuple[bool, str]:
    """Return (kept, reason). kept=True means include in valid set."""
    instruction_set = rec.get("instruction_set") or ""
    bullets = extract_bullets(instruction_set)
    n = len(bullets)

    if is_template_leak(instruction_set):
        return False, "template_leak"
    if is_word_fragmentation(bullets):
        return False, "word_fragmentation"
    if n > max_bullets:
        return False, f"over_cap (n={n})"
    if n == 0:
        return False, "no_bullets"
    return True, "kept"


def main():
    p = argparse.ArgumentParser(description="Build valid_record_ids.json")
    p.add_argument("--input-glob", required=True,
                   help="Glob for filtered JSONL files (records have 'id' field).")
    p.add_argument("--precomputed-dir", required=True,
                   help="Target precomputed_dir; valid_record_ids.json is written here.")
    p.add_argument("--max-bullets", type=int, default=6,
                   help="Drop records with more than this many bullets (default 6).")
    p.add_argument("--output-name", default="valid_record_ids.json",
                   help="File name inside precomputed_dir (default valid_record_ids.json).")
    p.add_argument("--emit-clean-jsonl-dir", default=None,
                   help="Optionally also write fully-cleaned JSONL copies of the "
                        "inputs (kept records only) into this directory — for "
                        "on-the-fly training, which has no mask support.")
    args = p.parse_args()

    paths = sorted(glob.glob(args.input_glob))
    # Exclude filter sidecars — .removed.jsonl are records the filter already
    # rejected; .errors.jsonl are judge-error records held out of both sets.
    paths = [
        p for p in paths
        if not p.endswith(".removed.jsonl") and not p.endswith(".errors.jsonl")
    ]
    if not paths:
        raise SystemExit(f"No JSONL files matched: {args.input_glob}")
    logger.info("Inputs: %d kept file(s) (excluded .removed.jsonl / .errors.jsonl)",
                len(paths))

    kept_ids: list[str] = []
    counts = Counter()
    per_source = Counter()
    per_source_dropped: dict[str, Counter] = {}

    clean_dir = Path(args.emit_clean_jsonl_dir) if args.emit_clean_jsonl_dir else None
    if clean_dir is not None:
        clean_dir.mkdir(parents=True, exist_ok=True)

    for path in paths:
        n_in = n_kept = 0
        clean_out = open(clean_dir / Path(path).name, "w", encoding="utf-8") if clean_dir else None
        try:
            for line in open(path, "r", encoding="utf-8"):
                rec = json.loads(line)
                n_in += 1
                kept, reason = evaluate(rec, args.max_bullets)
                counts[reason] += 1
                src = rec.get("source_dataset", "unknown")
                per_source[src] += 1
                if kept:
                    rid = rec.get("id") or rec.get("record_id")
                    if not rid:
                        raise RuntimeError(f"Record without id in {path}: {rec}")
                    kept_ids.append(rid)
                    n_kept += 1
                    if clean_out is not None:
                        clean_out.write(line if line.endswith("\n") else line + "\n")
                else:
                    per_source_dropped.setdefault(src, Counter())[reason] += 1
        finally:
            if clean_out is not None:
                clean_out.close()
        logger.info("  %s: %d -> %d kept (%.1f%%)",
                    Path(path).name, n_in, n_kept, 100 * n_kept / max(n_in, 1))

    out_path = Path(args.precomputed_dir) / args.output_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(kept_ids), encoding="utf-8")

    total = sum(counts.values())
    kept_total = counts["kept"]
    logger.info("=" * 60)
    logger.info("Reason breakdown:")
    for reason, n in counts.most_common():
        logger.info("  %-25s %s (%.2f%%)", reason, f"{n:>8,}", 100 * n / total)
    logger.info("=" * 60)
    logger.info("Per-source drops:")
    for src, cs in per_source_dropped.items():
        total_src = per_source[src]
        dropped = sum(cs.values())
        logger.info("  %s: kept %s / %s (%.1f%%)",
                    src, f"{total_src - dropped:,}", f"{total_src:,}",
                    100 * (total_src - dropped) / max(total_src, 1))
        for reason, n in cs.most_common():
            logger.info("    %-22s %s", reason, f"{n:>8,}")
    logger.info("=" * 60)
    logger.info("Total kept: %s / %s (%.1f%%)",
                f"{kept_total:,}", f"{total:,}", 100 * kept_total / total)
    logger.info("Wrote %s (%d ids)", out_path, len(kept_ids))


if __name__ == "__main__":
    main()
