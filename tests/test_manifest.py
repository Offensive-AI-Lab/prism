"""Manifest round-trip + resume-safety guarantees (DATA-3 regression tests)."""

import json

import pytest

from prism.activations.manifest import (
    ExtractionManifest,
    ShardInfo,
    SplitManifest,
    load_or_create_manifest,
)

_ARGS = dict(
    model_id="Qwen/Qwen3.5-9B",
    layers=[16],
    num_tokens=128,
    token_position="last",
    dtype="bfloat16",
    val_ratio=0.1,
    test_ratio=0.1,
    split_seed=42,
    random_seed=None,
    dataset_paths=["/data/a.jsonl"],
    total_records=1000,
    train_records=800,
    val_records=100,
    test_records=100,
    records_per_shard=256,
)


def test_manifest_roundtrip(tmp_path):
    m = load_or_create_manifest(tmp_path, **_ARGS)
    assert m.status == "in_progress"
    assert (tmp_path / "manifest.json").is_file()
    for split in ("train", "val", "test"):
        assert (tmp_path / split).is_dir()
    loaded = ExtractionManifest.load(tmp_path / "manifest.json")
    assert loaded == m


def test_resume_with_same_args_is_accepted(tmp_path):
    load_or_create_manifest(tmp_path, **_ARGS)
    resumed = load_or_create_manifest(tmp_path, **_ARGS)
    assert resumed.status == "in_progress"


def test_resume_with_different_params_hard_errors(tmp_path):
    load_or_create_manifest(tmp_path, **_ARGS)
    for key, bad in [("split_seed", 7), ("layers", [8]), ("num_tokens", 64),
                     ("token_position", "first"), ("records_per_shard", 128)]:
        args = dict(_ARGS, **{key: bad})
        with pytest.raises(RuntimeError, match="Refusing to resume"):
            load_or_create_manifest(tmp_path, **args)


def test_completed_manifest_skips_validation(tmp_path):
    m = load_or_create_manifest(tmp_path, **_ARGS)
    m.status = "complete"
    m.save(tmp_path / "manifest.json")
    # Different args, but the run is finished — load succeeds (extract exits early).
    out = load_or_create_manifest(tmp_path, **dict(_ARGS, split_seed=7))
    assert out.status == "complete"


def test_atomic_save_leaves_no_tmp(tmp_path):
    m = ExtractionManifest(**_ARGS)
    m.save(tmp_path / "manifest.json")
    assert not list(tmp_path.glob("*.tmp"))
    assert json.loads((tmp_path / "manifest.json").read_text())["model_id"] == _ARGS["model_id"]


def test_split_manifest_roundtrip(tmp_path):
    sm = SplitManifest(num_records=10, num_shards=1,
                       shards=[ShardInfo(file="s0.safetensors", meta_file="s0.meta.json", records=10)])
    sm.save(tmp_path / "m.json")
    loaded = SplitManifest.load(tmp_path / "m.json")
    assert loaded == sm
