"""Oracle-dataset records for ON-THE-FLY activation extraction.

This is the in-memory counterpart of the activation cache: the same JSONL
files, the same record filter and ``valid_record_ids.json`` mask, and the
same deterministic split (:mod:`prism.activations.split`) — minus the
activations, which the trainers extract from the resident target model
per batch instead of reading from disk.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from torch.utils.data import Dataset

from prism.activations.split import split_records

logger = logging.getLogger(__name__)

VALID_IDS_FILENAME = "valid_record_ids.json"


@dataclass
class Record:
    id: str
    source_dataset: str
    prompt_a: str
    response_a: str
    prompt_b: str
    response_b: str
    paraphrase_group_id: Optional[str] = None

    # Duck-type as a mapping for the split helpers (they use ``rec.get``).
    def get(self, key: str, default=None):
        return getattr(self, key, default)


def load_records(paths: Sequence[str], require_response_b: bool = True) -> Tuple[List[Record], dict]:
    """Load JSONL oracle records in the given file order.

    Mirrors the extractor's record filter (``prompt_a`` and ``response_a``
    required; ``metadata.paraphrase_group_id`` carried for the split).
    Trainers additionally need ``response_b`` (the ground-truth instruction
    list), so it is required by default.

    Returns ``(records, source_map)`` with ``source_map`` = ``{source_dataset:
    idx}`` over the sorted source names, as in the cache manifest.
    """
    records: List[Record] = []
    source_names: set = set()
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_absolute():
            path = (Path.cwd() / raw_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"dataset file not found: {path}")
        count = 0
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                prompt_a = obj.get("prompt_a", "")
                response_a = obj.get("response_a", "")
                response_b = obj.get("response_b", "")
                if not (prompt_a and response_a):
                    continue
                if require_response_b and not response_b:
                    continue
                metadata = obj.get("metadata")
                gid = metadata.get("paraphrase_group_id") if isinstance(metadata, dict) else None
                rec = Record(
                    id=obj.get("id", ""),
                    source_dataset=obj.get("source_dataset", "unknown"),
                    prompt_a=prompt_a,
                    response_a=response_a,
                    prompt_b=obj.get("prompt_b", ""),
                    response_b=response_b,
                    paraphrase_group_id=gid,
                )
                records.append(rec)
                source_names.add(rec.source_dataset)
                count += 1
        logger.info("Loaded %s records from %s", f"{count:,}", path.name)
    source_map = {name: idx for idx, name in enumerate(sorted(source_names))}
    logger.info("Total records: %s across %d sources", f"{len(records):,}", len(source_map))
    return records, source_map


def find_valid_ids_file(paths: Iterable[str]) -> Optional[Path]:
    """Locate ``valid_record_ids.json`` for a set of JSONL files.

    The recipes keep the mask next to the ``jsonl/`` directory
    (``<dataset>/jsonl/*.jsonl`` + ``<dataset>/valid_record_ids.json``); a
    mask inside the JSONL directory itself is accepted too. Returns the first
    match or ``None``.
    """
    seen: list[Path] = []
    for raw in paths:
        d = Path(raw).resolve().parent
        for cand in (d / VALID_IDS_FILENAME, d.parent / VALID_IDS_FILENAME):
            if cand not in seen:
                seen.append(cand)
                if cand.is_file():
                    return cand
    return None


def apply_valid_ids(records: List[Record], valid_ids_path) -> List[Record]:
    """Keep only records whose ``id`` is in the JSON list at ``valid_ids_path``.

    Same semantics as the cache loader: the mask applies to every split.
    """
    valid = set(json.loads(Path(valid_ids_path).read_text(encoding="utf-8")))
    kept = [r for r in records if r.id in valid]
    logger.info(
        "Applied %s: kept %s / %s records", Path(valid_ids_path).name,
        f"{len(kept):,}", f"{len(records):,}",
    )
    if not kept:
        raise ValueError(f"{valid_ids_path} left 0 records — wrong mask for these files?")
    return kept


def split_dataset(records: List[Record], cfg: dict):
    """Split with the cache's parameters (``split_*`` config keys)."""
    return split_records(
        records,
        val_ratio=float(cfg["split_val_ratio"]),
        test_ratio=float(cfg.get("split_test_ratio", 0.0)),
        seed=int(cfg["split_seed"]),
        stratify_by=cfg.get("split_stratify_by"),
    )


class RecordDataset(Dataset):
    """List-backed dataset of :class:`Record`; ``.records`` exposes ids in order."""

    def __init__(self, records: List[Record], source_map: dict):
        self.records = records
        self.source_map = source_map

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Record:
        return self.records[idx]

    def record_ids(self) -> List[str]:
        return [r.id for r in self.records]


def load_split_records(cfg: dict, valid_ids_path=None):
    """Load ``cfg['dataset_paths']``, split, then apply the mask — in the
    cache's order of operations.

    The extractor splits the *full* record list (its filter needs only
    ``prompt_a``/``response_a``) and the ``valid_record_ids.json`` mask is
    applied later, at load time, per split. Doing the same here — split the
    unmasked population, then mask each split, then drop label-less
    records — is what makes on-the-fly train/val membership identical to
    the cache built from the same files.

    ``valid_ids_path``: explicit mask path, or ``None`` to auto-detect next
    to the JSONL files (``""``/``"none"`` disables the auto-detection).
    Returns ``(train, val, test, source_map)`` as lists of :class:`Record`.
    """
    paths = list(cfg["dataset_paths"])
    records, source_map = load_records(paths, require_response_b=False)
    train, val, test = split_dataset(records, cfg)
    logger.info(
        "Split (seed=%s val=%.2f test=%.2f stratify=%s): train=%s val=%s test=%s",
        cfg["split_seed"], float(cfg["split_val_ratio"]), float(cfg.get("split_test_ratio", 0.0)),
        cfg.get("split_stratify_by"), f"{len(train):,}", f"{len(val):,}", f"{len(test):,}",
    )
    if valid_ids_path is None:
        valid_ids_path = find_valid_ids_file(paths)
    elif str(valid_ids_path).lower() in ("", "none"):
        valid_ids_path = None
    if valid_ids_path is not None:
        valid = set(json.loads(Path(valid_ids_path).read_text(encoding="utf-8")))
        train, val, test = ([r for r in part if r.id in valid] for part in (train, val, test))
        logger.info(
            "Applied %s: train=%s val=%s test=%s", Path(valid_ids_path).name,
            f"{len(train):,}", f"{len(val):,}", f"{len(test):,}",
        )
        if not train:
            raise ValueError(f"{valid_ids_path} left 0 training records — wrong mask for these files?")
    else:
        logger.warning(
            "No %s found next to the dataset files — training on UNFILTERED records "
            "(pass --valid-record-ids to apply the mask, as the cache path does).",
            VALID_IDS_FILENAME,
        )
    # Trainers need the label; the extractor keeps label-less records in the
    # cache but the training loaders never see them.
    n_before = len(train) + len(val) + len(test)
    train, val, test = ([r for r in part if r.response_b] for part in (train, val, test))
    dropped = n_before - (len(train) + len(val) + len(test))
    if dropped:
        logger.info("Dropped %d records without response_b", dropped)
    return train, val, test, source_map
