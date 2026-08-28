"""Deterministic train/val/test splitting shared by activation extraction and
the on-the-fly trainers.

The cache written by ``prism.activations.extract`` freezes its split at
extraction time with these exact functions. The on-the-fly paths of
``prism.sft.train`` and ``prism.rl.train`` call the same functions with the
same parameters, so a run that extracts activations in-loop sees the same
train/val/test membership as a run that reads the cache (given the same
input files, in the same order, and the same seed / ratios).

Records are any objects supporting ``rec.get(key, default)`` — the plain
dicts used by the extractor or :class:`prism.activations.records.Record`.
"""

from __future__ import annotations

import logging
import random

logger = logging.getLogger(__name__)


def group_units(records) -> list:
    """Group records into split units by ``paraphrase_group_id``.

    Records sharing a non-null group id form one unit (the whole unit is
    assigned to a single split); records without a group id stay
    singleton units. Unit order follows first appearance in ``records``, so
    the result is deterministic for a given record order.
    """
    units: list[list] = []
    by_group: dict = {}
    for rec in records:
        gid = rec.get("paraphrase_group_id")
        if gid is None:
            units.append([rec])
        else:
            unit = by_group.get(gid)
            if unit is None:
                unit = []
                by_group[gid] = unit
                units.append(unit)
            unit.append(rec)
    return units


def split_grouped(records, val_ratio: float, test_ratio: float,
                  rng: random.Random):
    """Split records into (train, val, test) keeping paraphrase groups intact.

    Mirrors the plain record-level split semantics — shuffle with the same
    rng, then carve off ``max(1, int(N * ratio))`` records for test and val —
    but shuffles and assigns whole units, so paraphrases of one base example
    never straddle a split boundary. Only used when at least one record
    carries a ``paraphrase_group_id``.
    """
    units = group_units(records)
    rng.shuffle(units)
    n = len(records)
    test_target = max(1, int(n * test_ratio))
    val_target = max(1, int(n * val_ratio))
    train_records: list = []
    val_records: list = []
    test_records: list = []
    for unit in units:
        if len(test_records) < test_target:
            test_records.extend(unit)
        elif len(val_records) < val_target:
            val_records.extend(unit)
        else:
            train_records.extend(unit)
    return train_records, val_records, test_records


def split_records(records, val_ratio: float, test_ratio: float, seed: int,
                  stratify_by: str | None = None):
    """Shuffle and split into (train, val, test).

    If ``stratify_by`` is given (e.g. ``"source_dataset"``), each value of
    that field is split independently with the same ratios and the
    per-bucket train/val/test are concatenated. Guarantees proportional
    representation in every split.

    Records carrying a non-null ``paraphrase_group_id`` are assigned as whole
    groups so paraphrases of the same base example never land in different
    splits. Paraphrase-free data takes the record-level path (identical
    output for the same seed).
    """
    has_groups = any(rec.get("paraphrase_group_id") is not None for rec in records)

    if stratify_by:
        buckets: dict[str, list] = {}
        if has_groups:
            # Bucket whole paraphrase units by their first record's stratify
            # value so a group that spans stratify values is still assigned
            # to a single bucket (and therefore a single split) — leakage
            # prevention takes precedence over exact stratification.
            for unit in group_units(records):
                key = unit[0].get(stratify_by, "_unknown")
                buckets.setdefault(key, []).extend(unit)
        else:
            for rec in records:
                buckets.setdefault(rec.get(stratify_by, "_unknown"), []).append(rec)
        train_all: list = []
        val_all: list = []
        test_all: list = []
        for key in sorted(buckets):
            sub = buckets[key]
            rng = random.Random(seed)
            if has_groups:
                sub_train, sub_val, sub_test = split_grouped(sub, val_ratio, test_ratio, rng)
                t_size, v_size = len(sub_test), len(sub_val)
                test_all.extend(sub_test)
                val_all.extend(sub_val)
                train_all.extend(sub_train)
            else:
                rng.shuffle(sub)
                t_size = max(1, int(len(sub) * test_ratio))
                v_size = max(1, int(len(sub) * val_ratio))
                test_all.extend(sub[:t_size])
                val_all.extend(sub[t_size:t_size + v_size])
                train_all.extend(sub[t_size + v_size:])
            logger.info(
                "  stratify[%s=%s]: train=%d val=%d test=%d (of %d)",
                stratify_by, key, len(sub) - t_size - v_size, v_size, t_size, len(sub),
            )
        rng_post = random.Random(seed + 1)
        rng_post.shuffle(train_all)
        rng_post.shuffle(val_all)
        rng_post.shuffle(test_all)
        return train_all, val_all, test_all

    if has_groups:
        return split_grouped(records, val_ratio, test_ratio, random.Random(seed))

    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    test_size = max(1, int(len(shuffled) * test_ratio))
    val_size = max(1, int(len(shuffled) * val_ratio))
    test_records = shuffled[:test_size]
    val_records = shuffled[test_size:test_size + val_size]
    train_records = shuffled[test_size + val_size:]
    return train_records, val_records, test_records
