"""Build a stratified list of "hard" training record_ids from prior judge traces.

Aggregates per-record mean reward from one or more `judge_traces.jsonl`
files, joins with the train-split meta to recover `source_dataset`, then
keeps records that are below the reward threshold (i.e. the policy is
NOT already ceiling-hitting them).

Stratification: per-source. If `--target-size T` is set, each source S
gets `round(T * N_S / N_total)` slots filled by the hardest (lowest-mean)
records of S first. Without `--target-size`, all records under the
threshold are kept and per-source counts are reported as-is.

Records that never appeared in the supplied judge_traces are treated as
"unobserved" — by default they are included (no evidence that they're
easy) and sorted to the end of each source's pool so they only fill
remaining slots when `--target-size` caps the bucket. Pass
`--exclude-unobserved` to drop them entirely.

Output is a JSON dict:
    {
      "record_ids": [<id>, <id>, ...],   # flat list for the train hook
      "source_breakdown": {<src>: <count>, ...},
      "total": <int>,
      "config": {<argparse vars>},
    }

Run:
    uv run python -m prism.rl.build_hard_ids \\
        --judge-traces \\
            checkpoints/grpo-run-a/judge_traces.jsonl \\
            checkpoints/grpo-run-b/judge_traces.jsonl \\
        --precomputed-dir <precomputed-activations-dir> \\
        --threshold 0.85 \\
        --output hard_train_ids.json
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _load_per_record_rewards(
    trace_paths: list[Path],
    min_observations: int,
) -> dict[str, float]:
    """Aggregate per-record mean reward across all supplied traces.

    Skips rows with `judge_error` set (those rewards are forced to 0 by
    train.py and would falsely flag the record as hard).
    """
    rewards: dict[str, list[float]] = collections.defaultdict(list)
    n_rows = 0
    n_skipped = 0
    for path in trace_paths:
        with path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec.get("judge_error"):
                    n_skipped += 1
                    continue
                rewards[rec["record_id"]].append(float(rec["reward"]))
                n_rows += 1
    means: dict[str, float] = {}
    n_too_few = 0
    for rid, rs in rewards.items():
        if len(rs) < min_observations:
            n_too_few += 1
            continue
        means[rid] = sum(rs) / len(rs)
    logger.info(
        "Loaded %d trace rows (%d skipped as judge_error) → %d unique record_ids; "
        "%d dropped for fewer than %d observations",
        n_rows, n_skipped, len(rewards), n_too_few, min_observations,
    )
    return means


def _load_train_meta(precomputed_dir: Path, split: str) -> dict[str, str]:
    """Walk `<precomputed_dir>/<split>/shard-*.meta.json` → {record_id: source_dataset}."""
    split_dir = precomputed_dir / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"split dir not found: {split_dir}")
    rid_to_source: dict[str, str] = {}
    for meta_path in sorted(split_dir.glob("shard-*.meta.json")):
        for rec in json.loads(meta_path.read_text(encoding="utf-8")):
            rid_to_source[rec["record_id"]] = rec.get("source_dataset", "unknown")
    if not rid_to_source:
        raise RuntimeError(f"no shard-*.meta.json under {split_dir}")
    logger.info("Train split has %d records across %d sources",
                len(rid_to_source), len(set(rid_to_source.values())))
    return rid_to_source


def _stratified_pick(
    by_source: dict[str, list[tuple[str, float, bool]]],
    source_totals: dict[str, int],
    target_size: int | None,
) -> dict[str, list[str]]:
    """Pick records per source, sorted hardest-first.

    Each entry is (record_id, mean_reward_or_inf, was_observed). Sort key
    puts observed-and-low first, then unobserved (treated as +inf so they
    only fill remaining target slots).

    With `target_size`, each source S gets `round(target_size * N_S / total)`
    slots; the leftover from rounding is dropped silently rather than
    redistributed (keeps the proportion clean).
    """
    selected: dict[str, list[str]] = {}
    total_underlying = sum(source_totals.values())
    for src in sorted(by_source):
        rows = sorted(
            by_source[src],
            key=lambda t: (t[1], t[0]),  # (mean asc, record_id for stable tiebreak)
        )
        if target_size is not None:
            cap = max(1, round(target_size * source_totals[src] / total_underlying))
            rows = rows[:cap]
        selected[src] = [rid for rid, _, _ in rows]
    return selected


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--judge-traces", nargs="+", required=True, type=Path,
                   help="One or more judge_traces.jsonl files from prior runs.")
    p.add_argument("--precomputed-dir", required=True, type=Path,
                   help="Same value as cfg['precomputed_dir'] for the upcoming run; "
                        "used to read train-split meta for source_dataset stratification.")
    p.add_argument("--split", default="train", choices=["train", "val", "test"],
                   help="Which split's meta to walk (default: train).")
    p.add_argument("--threshold", type=float, default=0.85,
                   help="Records whose observed mean reward is BELOW this are 'hard' "
                        "and eligible for inclusion. Default 0.85.")
    p.add_argument("--min-observations", type=int, default=1,
                   help="Require at least this many observed (judge_error=False) trace "
                        "rows per record before trusting its mean. Default 1.")
    p.add_argument("--exclude-unobserved", action="store_true",
                   help="Drop records that never appeared in the supplied judge_traces. "
                        "Default keeps them (treated as unknown-difficulty, ranked last).")
    p.add_argument("--target-size", type=int, default=None,
                   help="If set, cap total output at ~this many records, allocating per "
                        "source in proportion to train-split source counts.")
    p.add_argument("--output", required=True, type=Path,
                   help="Output JSON file path.")
    args = p.parse_args()

    rewards = _load_per_record_rewards(args.judge_traces, args.min_observations)
    rid_to_source = _load_train_meta(args.precomputed_dir, args.split)

    source_totals: dict[str, int] = collections.Counter(rid_to_source.values())

    by_source: dict[str, list[tuple[str, float, bool]]] = collections.defaultdict(list)
    n_observed_hard = 0
    n_observed_easy = 0
    n_unobserved = 0
    for rid, src in rid_to_source.items():
        if rid in rewards:
            m = rewards[rid]
            if m < args.threshold:
                by_source[src].append((rid, m, True))
                n_observed_hard += 1
            else:
                n_observed_easy += 1
        else:
            if not args.exclude_unobserved:
                # Unobserved → push to end of each source's pool via +inf key.
                by_source[src].append((rid, float("inf"), False))
            n_unobserved += 1

    logger.info(
        "Eligible pool: observed-hard=%d  observed-easy(dropped)=%d  unobserved=%d (%s)",
        n_observed_hard, n_observed_easy, n_unobserved,
        "kept" if not args.exclude_unobserved else "dropped",
    )

    selected = _stratified_pick(by_source, source_totals, args.target_size)
    flat_ids = [rid for src in sorted(selected) for rid in selected[src]]

    breakdown = {src: len(ids) for src, ids in selected.items()}
    underlying_pct = {
        src: f"{100 * source_totals[src] / sum(source_totals.values()):.1f}%"
        for src in source_totals
    }
    selected_pct = {
        src: f"{100 * len(ids) / max(1, len(flat_ids)):.1f}%"
        for src, ids in selected.items()
    }

    logger.info("Selected %d records — per-source breakdown:", len(flat_ids))
    for src in sorted(set(list(breakdown) + list(source_totals))):
        logger.info(
            "  %-25s  selected=%4d (%5s)  vs underlying=%4d (%5s)",
            src, breakdown.get(src, 0), selected_pct.get(src, "0.0%"),
            source_totals.get(src, 0), underlying_pct.get(src, "0.0%"),
        )

    out = {
        "record_ids": flat_ids,
        "source_breakdown": breakdown,
        "underlying_source_totals": dict(source_totals),
        "total": len(flat_ids),
        "config": {
            "judge_traces": [str(p) for p in args.judge_traces],
            "precomputed_dir": str(args.precomputed_dir),
            "split": args.split,
            "threshold": args.threshold,
            "min_observations": args.min_observations,
            "exclude_unobserved": args.exclude_unobserved,
            "target_size": args.target_size,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote %d ids → %s", len(flat_ids), args.output)


if __name__ == "__main__":
    main()
