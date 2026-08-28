"""sample.py — stratified pilot + main sampling for judge calibration.

Draws disjoint subsets from the *test* split of the precomputed activations,
stratified by ``source_dataset`` according to the ``--ratios`` argument
(default: ultrachat=0.62, if_multi_constraints=0.33, if_eval=0.05).

The default DEVIATES from natural proportionality (which would give if_eval
only ~1 record) in order to give the small `if_eval` source enough
records to compute a defensible per-source κ in the calibration report.
Pass ``--ratios proportional`` to fall back to natural test-split
proportions if you want κ to reflect the deployment distribution exactly.

Outputs (under ``--out-dir``):
  pilot_ids.json   (50 records — dual-annotated)
  main_ids.json    (400 records — single-annotated)

Each entry: {"record_id": str, "global_idx": int, "source_dataset": str}.
The ``global_idx`` is the dataset index needed by downstream stages so
they can pull activations + text without re-scanning metadata.

Run:
    uv run python -m prism.calibration.sample --out-dir calibration_out
    uv run python -m prism.calibration.sample --out-dir calibration_out --pilot 3 --main 5
    uv run python -m prism.calibration.sample --out-dir calibration_out \\
        --ratios "ultrachat=0.5,if_multi_constraints=0.25,if_eval=0.25"
    uv run python -m prism.calibration.sample --out-dir calibration_out --ratios proportional
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import Counter, defaultdict
from pathlib import Path

from prism.activations import PrecomputedActivationDataset
from prism.rl.config import RL_CONFIG

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def _iter_metadata(ds: PrecomputedActivationDataset):
    """Yield (global_idx, record_id, source_dataset) for every record in the dataset.

    Reads only the per-shard meta JSON files — never touches the safetensors,
    so iterating 27k records takes seconds, not minutes.
    """
    for global_idx, (_, meta_path, offset, _) in enumerate(ds._index):
        meta_list = ds._get_meta(meta_path)
        meta = meta_list[offset]
        yield global_idx, meta.get("record_id", ""), meta.get("source_dataset", "unknown")


# Default per-source ratios. Deviates from natural pool proportionality
# (which is 72/28/0.17) so the tiny `if_eval` source gets enough records
# to support a defensible per-source κ in the calibration report.
DEFAULT_RATIOS: dict[str, float] = {
    "ultrachat": 0.62,
    "if_multi_constraints": 0.33,
    "if_eval": 0.05,
}


def _resolve_ratios(
    by_source: dict[str, list[tuple[int, str]]],
    spec: str | dict[str, float] | None,
) -> dict[str, float]:
    """Turn a CLI spec into a {source: fraction} dict that sums to 1.0.

    Recognised forms:
      - None         → DEFAULT_RATIOS (filtered to sources present in the pool)
      - "proportional" → natural test-split proportions (|source| / |total|)
      - dict         → used directly (validated)
      - "k1=v1,k2=v2,…" string → parsed to dict
    """
    if spec is None:
        ratios = {s: r for s, r in DEFAULT_RATIOS.items() if s in by_source}
    elif spec == "proportional":
        total = sum(len(v) for v in by_source.values())
        ratios = {s: len(v) / total for s, v in by_source.items()}
    elif isinstance(spec, dict):
        ratios = dict(spec)
    else:
        ratios = {}
        for piece in str(spec).split(","):
            piece = piece.strip()
            if not piece:
                continue
            k, _, v = piece.partition("=")
            ratios[k.strip()] = float(v.strip())

    # Validate: cover all sources, sum to ~1.0.
    missing = set(by_source) - set(ratios)
    extra = set(ratios) - set(by_source)
    if missing:
        raise ValueError(f"--ratios is missing sources present in test split: {sorted(missing)}")
    if extra:
        raise ValueError(f"--ratios mentions sources not in test split: {sorted(extra)}")
    total = sum(ratios.values())
    if not 0.99 <= total <= 1.01:
        raise ValueError(f"--ratios must sum to 1.0 (got {total:.4f})")

    # Normalise to exactly 1.0 to avoid largest-remainder drift.
    return {s: r / total for s, r in ratios.items()}


def _stratified_split(
    by_source: dict[str, list[tuple[int, str]]],
    n_pilot: int,
    n_main: int,
    rng: random.Random,
    ratios: dict[str, float] | None = None,
) -> tuple[list[tuple[int, str, str]], list[tuple[int, str, str]]]:
    """Draw `n_pilot` + `n_main` disjoint records, stratified by ``ratios``.

    If ``ratios`` is None, falls back to natural pool proportions (preserves
    historical behaviour for callers that don't pass it).
    """
    if ratios is None:
        total = sum(len(v) for v in by_source.values())
        ratios = {s: len(v) / total for s, v in by_source.items()}

    sources = sorted(by_source.keys())

    # Largest-remainder allocation so per-source counts sum to exactly n.
    def _allocate(n: int) -> dict[str, int]:
        raw = {s: n * ratios.get(s, 0.0) for s in sources}
        floor = {s: int(v) for s, v in raw.items()}
        remainder = n - sum(floor.values())
        frac_sorted = sorted(sources, key=lambda s: raw[s] - floor[s], reverse=True)
        for s in frac_sorted[:remainder]:
            floor[s] += 1
        return floor

    n_pilot_per = _allocate(n_pilot)
    n_main_per = _allocate(n_main)

    pilot: list[tuple[int, str, str]] = []
    main: list[tuple[int, str, str]] = []
    for s in sources:
        pool = list(by_source[s])
        rng.shuffle(pool)
        need = n_pilot_per[s] + n_main_per[s]
        if need > len(pool):
            raise ValueError(
                f"Source {s!r} has only {len(pool)} records; "
                f"need {need} (={n_pilot_per[s]} pilot + {n_main_per[s]} main) "
                "for stratified sample. Lower the ratio for this source or reduce n_pilot/n_main."
            )
        for idx, rid in pool[:n_pilot_per[s]]:
            pilot.append((idx, rid, s))
        for idx, rid in pool[n_pilot_per[s] : n_pilot_per[s] + n_main_per[s]]:
            main.append((idx, rid, s))

    return pilot, main


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pilot", type=int, default=50, help="Number of pilot records (dual-annotated)")
    p.add_argument("--main", type=int, default=400, help="Number of main records (single-annotated)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ratios", type=str, default=None,
                   help='Per-source fractions, e.g. "ultrachat=0.62,if_multi_constraints=0.33,if_eval=0.05" '
                        'or the keyword "proportional" for natural pool proportions. '
                        'Defaults to DEFAULT_RATIOS (62/33/5).')
    p.add_argument("--precomputed-dir", type=str, default=None,
                   help="Override RL_CONFIG['precomputed_dir']")
    p.add_argument("--out-dir", type=str, required=True,
                   help="Where to write pilot_ids.json + main_ids.json")
    args = p.parse_args()

    precomputed_dir = args.precomputed_dir or RL_CONFIG["precomputed_dir"]
    hook_layer = RL_CONFIG["hook_layer"]
    logger.info("Loading test split: %s (layer=%d)", precomputed_dir, hook_layer)
    ds = PrecomputedActivationDataset(precomputed_dir, split="test", layers=[hook_layer])
    logger.info("Test split: %d records, sources=%s", len(ds), list(ds.source_map))

    by_source: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for gi, rid, src in _iter_metadata(ds):
        by_source[src].append((gi, rid))
    logger.info("Per-source pool: %s", {s: len(v) for s, v in by_source.items()})

    ratios = _resolve_ratios(by_source, args.ratios)
    logger.info("Per-source target ratios: %s", {s: round(r, 4) for s, r in ratios.items()})

    rng = random.Random(args.seed)
    pilot, main = _stratified_split(by_source, args.pilot, args.main, rng, ratios=ratios)

    # Disjoint check + per-source distribution
    pilot_ids = {p[1] for p in pilot}
    main_ids = {m[1] for m in main}
    assert pilot_ids.isdisjoint(main_ids), "BUG: pilot and main share records"
    pilot_dist = Counter(p[2] for p in pilot)
    main_dist = Counter(m[2] for m in main)
    logger.info("Pilot per-source: %s", dict(pilot_dist))
    logger.info("Main  per-source: %s", dict(main_dist))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _write(records: list[tuple[int, str, str]], path: Path) -> None:
        path.write_text(
            json.dumps(
                [{"global_idx": gi, "record_id": rid, "source_dataset": src}
                 for (gi, rid, src) in records],
                indent=2,
            )
        )
        logger.info("Wrote %d records → %s", len(records), path)

    _write(pilot, out_dir / "pilot_ids.json")
    _write(main, out_dir / "main_ids.json")
    logger.info("Done. Seed=%d  Pilot=%d  Main=%d  Total=%d (test=%d)",
                args.seed, len(pilot), len(main), len(pilot) + len(main), len(ds))


if __name__ == "__main__":
    main()
