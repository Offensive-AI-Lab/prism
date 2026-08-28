"""Manifest management for precomputed activation shards."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def _atomic_write_text(path: Path, text: str) -> None:
    """Write text via a temp file + rename so a crash mid-write never leaves
    a truncated manifest behind."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


@dataclass
class ShardInfo:
    file: str
    meta_file: str
    records: int


@dataclass
class SplitManifest:
    """Manifest for a single split (train or val)."""

    num_records: int = 0
    num_shards: int = 0
    shards: List[ShardInfo] = field(default_factory=list)

    def save(self, path: Path) -> None:
        data = {
            "num_records": self.num_records,
            "num_shards": self.num_shards,
            "shards": [asdict(s) for s in self.shards],
        }
        _atomic_write_text(path, json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path) -> "SplitManifest":
        data = json.loads(path.read_text(encoding="utf-8"))
        shards = [ShardInfo(**s) for s in data.get("shards", [])]
        return cls(
            num_records=data["num_records"],
            num_shards=data["num_shards"],
            shards=shards,
        )


@dataclass
class ExtractionManifest:
    """Top-level manifest for a precomputed activation dataset."""

    status: str = "in_progress"
    model_id: str = ""
    layers: List[int] = field(default_factory=list)
    hidden_size: int = 0
    num_tokens: int = 128
    token_position: str = "last"
    dtype: str = "bfloat16"
    val_ratio: float = 0.05
    test_ratio: float = 0.05
    split_seed: int = 42
    random_seed: Optional[int] = None
    dataset_paths: List[str] = field(default_factory=list)
    total_records: int = 0
    train_records: int = 0
    val_records: int = 0
    test_records: int = 0
    records_per_shard: int = 256
    source_map: Dict[str, int] = field(default_factory=dict)
    # Resume tracking
    completed_train_shards: int = 0
    completed_val_shards: int = 0
    completed_test_shards: int = 0
    completed_records: int = 0

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        _atomic_write_text(path, json.dumps(data, indent=2))
        logger.debug("Manifest saved: %s", path)

    @classmethod
    def load(cls, path: Path) -> "ExtractionManifest":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# Parameters that determine split membership, record ordering, and shard
# contents. A resumed run must match every one of these — a silently accepted
# mismatch (e.g. a different --seed or --layers) would mix incompatible
# shards in one output directory.
RESUME_CRITICAL_FIELDS = (
    "model_id",
    "layers",
    "num_tokens",
    "token_position",
    "dtype",
    "val_ratio",
    "test_ratio",
    "split_seed",
    "random_seed",
    "dataset_paths",
    "total_records",
    "train_records",
    "val_records",
    "test_records",
    "records_per_shard",
)


def load_or_create_manifest(
    output_dir: Path,
    **kwargs,
) -> ExtractionManifest:
    """Load an existing manifest for resume, or create a new one.

    When an in-progress manifest exists, the current arguments are validated
    against it and a hard error is raised on any mismatch of the
    split/extraction-defining parameters (no silent proceed).
    """
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        manifest = ExtractionManifest.load(manifest_path)
        if manifest.status == "complete":
            logger.info("Extraction already complete at %s", output_dir)
            return manifest

        mismatches = [
            f"  {name}: manifest={getattr(manifest, name)!r} vs current={kwargs[name]!r}"
            for name in RESUME_CRITICAL_FIELDS
            if name in kwargs and getattr(manifest, name) != kwargs[name]
        ]
        if mismatches:
            raise RuntimeError(
                f"Refusing to resume: {manifest_path} was created with different "
                "extraction parameters:\n"
                + "\n".join(mismatches)
                + "\nUse a fresh --output-dir (or rerun with the original arguments)."
            )

        logger.info(
            "Resuming extraction from %s (completed %d records)",
            output_dir,
            manifest.completed_records,
        )
        return manifest

    manifest = ExtractionManifest(**kwargs)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "train").mkdir(exist_ok=True)
    (output_dir / "val").mkdir(exist_ok=True)
    (output_dir / "test").mkdir(exist_ok=True)
    manifest.save(manifest_path)
    return manifest


def finalize_manifest(
    output_dir: Path,
    manifest: ExtractionManifest,
    train_manifest: SplitManifest,
    val_manifest: SplitManifest,
    test_manifest: SplitManifest,
) -> None:
    """Mark extraction as complete and write all manifests."""
    manifest.status = "complete"
    manifest.completed_records = manifest.total_records
    manifest.completed_train_shards = train_manifest.num_shards
    manifest.completed_val_shards = val_manifest.num_shards
    manifest.completed_test_shards = test_manifest.num_shards
    manifest.save(output_dir / "manifest.json")
    train_manifest.save(output_dir / "train" / "manifest.json")
    val_manifest.save(output_dir / "val" / "manifest.json")
    test_manifest.save(output_dir / "test" / "manifest.json")
    logger.info(
        "Extraction complete: %d train (%d shards), %d val (%d shards), %d test (%d shards)",
        train_manifest.num_records, train_manifest.num_shards,
        val_manifest.num_records, val_manifest.num_shards,
        test_manifest.num_records, test_manifest.num_shards,
    )
