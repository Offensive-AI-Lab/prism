"""On-the-fly data path: shared split, record loader/mask, collate contract,
RL data-mode validation, sampler fast path. CPU-only, no network."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import pytest
import torch

from prism.activations import split as split_mod
from prism.activations.records import (
    Record, RecordDataset, apply_valid_ids, find_valid_ids_file, load_records,
    load_split_records,
)


# ── 1. split.py must reproduce the extractor's historical implementation ──────
# Reference copy of the split logic as it shipped inside extract.py before the
# refactor (verbatim). Guards against drift between cache and on-the-fly splits.
def _ref_group_units(records):
    units, by_group = [], {}
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


def _ref_split_grouped(records, val_ratio, test_ratio, rng):
    units = _ref_group_units(records)
    rng.shuffle(units)
    n = len(records)
    test_target = max(1, int(n * test_ratio))
    val_target = max(1, int(n * val_ratio))
    tr, va, te = [], [], []
    for unit in units:
        if len(te) < test_target:
            te.extend(unit)
        elif len(va) < val_target:
            va.extend(unit)
        else:
            tr.extend(unit)
    return tr, va, te


def _ref_split_records(records, val_ratio, test_ratio, seed, stratify_by=None):
    has_groups = any(rec.get("paraphrase_group_id") is not None for rec in records)
    if stratify_by:
        buckets = {}
        if has_groups:
            for unit in _ref_group_units(records):
                buckets.setdefault(unit[0].get(stratify_by, "_unknown"), []).extend(unit)
        else:
            for rec in records:
                buckets.setdefault(rec.get(stratify_by, "_unknown"), []).append(rec)
        tr_all, va_all, te_all = [], [], []
        for key in sorted(buckets):
            sub = buckets[key]
            rng = random.Random(seed)
            if has_groups:
                s_tr, s_va, s_te = _ref_split_grouped(sub, val_ratio, test_ratio, rng)
                te_all.extend(s_te); va_all.extend(s_va); tr_all.extend(s_tr)
            else:
                rng.shuffle(sub)
                t = max(1, int(len(sub) * test_ratio)); v = max(1, int(len(sub) * val_ratio))
                te_all.extend(sub[:t]); va_all.extend(sub[t:t + v]); tr_all.extend(sub[t + v:])
        rng_post = random.Random(seed + 1)
        rng_post.shuffle(tr_all); rng_post.shuffle(va_all); rng_post.shuffle(te_all)
        return tr_all, va_all, te_all
    if has_groups:
        return _ref_split_grouped(records, val_ratio, test_ratio, random.Random(seed))
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    t = max(1, int(len(shuffled) * test_ratio)); v = max(1, int(len(shuffled) * val_ratio))
    return shuffled[t + v:], shuffled[t:t + v], shuffled[:t]


def _synth(n, groups: bool, seed=0):
    rng = random.Random(seed)
    recs = []
    for i in range(n):
        gid = f"g{i // 3}" if groups and rng.random() < 0.7 else None
        recs.append({"id": f"r{i}", "source_dataset": rng.choice(["a", "b", "c"]),
                     "paraphrase_group_id": gid})
    return recs


@pytest.mark.parametrize("groups", [False, True])
@pytest.mark.parametrize("stratify", [None, "source_dataset"])
@pytest.mark.parametrize("seed", [42, 7])
def test_split_matches_extractor_reference(groups, stratify, seed):
    recs = _synth(500, groups, seed)
    got = split_mod.split_records(list(recs), 0.1, 0.1, seed, stratify)
    ref = _ref_split_records(list(recs), 0.1, 0.1, seed, stratify)
    assert [[r["id"] for r in part] for part in got] == [[r["id"] for r in part] for part in ref]
    ids = [r["id"] for part in got for r in part]
    assert sorted(ids) == sorted(r["id"] for r in recs)  # partition, nothing lost


def test_split_keeps_paraphrase_groups_together():
    recs = _synth(300, True, 1)
    tr, va, te = split_mod.split_records(recs, 0.1, 0.1, 42)
    where = {}
    for name, part in (("tr", tr), ("va", va), ("te", te)):
        for r in part:
            if r["paraphrase_group_id"] is not None:
                where.setdefault(r["paraphrase_group_id"], set()).add(name)
    assert all(len(v) == 1 for v in where.values())


def test_extractor_uses_shared_split():
    from prism.activations import extract
    assert extract._split_records is split_mod.split_records


# ── 2. record loader / mask ───────────────────────────────────────────────────
def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


@pytest.fixture
def dataset_dir(tmp_path):
    rows_a = [
        {"id": "a1", "source_dataset": "if_eval", "prompt_a": "p1", "response_a": "r1",
         "prompt_b": "q", "response_b": "- x", "metadata": {"paraphrase_group_id": "g1"}},
        {"id": "a2", "source_dataset": "if_eval", "prompt_a": "p2", "response_a": "r2",
         "prompt_b": "q", "response_b": "- y", "metadata": {"paraphrase_group_id": "g1"}},
        {"id": "bad", "source_dataset": "if_eval", "prompt_a": "", "response_a": "r"},   # no prompt
        {"id": "nolabel", "source_dataset": "if_eval", "prompt_a": "p", "response_a": "r"},  # no response_b
    ]
    rows_b = [{"id": f"b{i}", "source_dataset": "ultrachat", "prompt_a": f"p{i}", "response_a": f"r{i}",
               "prompt_b": "q", "response_b": "- z"} for i in range(30)]
    _write_jsonl(tmp_path / "jsonl" / "ds_a.jsonl", rows_a)
    _write_jsonl(tmp_path / "jsonl" / "ds_b.jsonl", rows_b)
    (tmp_path / "valid_record_ids.json").write_text(json.dumps(["a1", "a2"] + [f"b{i}" for i in range(20)]))
    return tmp_path


def test_load_records_filter_and_source_map(dataset_dir):
    paths = [str(dataset_dir / "jsonl" / "ds_a.jsonl"), str(dataset_dir / "jsonl" / "ds_b.jsonl")]
    recs, smap = load_records(paths)
    assert [r.id for r in recs[:2]] == ["a1", "a2"] and len(recs) == 32
    assert smap == {"if_eval": 0, "ultrachat": 1}
    assert recs[0].paraphrase_group_id == "g1" and recs[2].paraphrase_group_id is None
    recs_pre, _ = load_records(paths, require_response_b=False)
    assert len(recs_pre) == 33  # extractor semantics: response_b optional


def test_load_records_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_records([str(tmp_path / "nope.jsonl")])


def test_valid_ids_autodetect_and_mask(dataset_dir):
    paths = [str(dataset_dir / "jsonl" / "ds_b.jsonl")]
    assert find_valid_ids_file(paths) == dataset_dir / "valid_record_ids.json"
    recs, _ = load_records(paths)
    kept = apply_valid_ids(recs, dataset_dir / "valid_record_ids.json")
    assert [r.id for r in kept] == [f"b{i}" for i in range(20)]
    with pytest.raises(ValueError):
        apply_valid_ids([Record("zzz", "s", "p", "r", "q", "b")], dataset_dir / "valid_record_ids.json")


def test_load_split_records_uses_split_keys(dataset_dir):
    paths = sorted(str(p) for p in (dataset_dir / "jsonl").glob("*.jsonl"))
    cfg = {"dataset_paths": paths, "split_val_ratio": 0.1, "split_test_ratio": 0.1,
           "split_seed": 42, "split_stratify_by": None}
    tr, va, te, smap = load_split_records(cfg)          # auto-detected mask → 22 records
    assert len(tr) + len(va) + len(te) == 22
    # cache order of operations: split the FULL population (extractor filter,
    # response_b optional), then mask each split, then drop label-less records
    full, _ = load_records(paths, require_response_b=False)
    assert len(full) == 33
    ref = split_mod.split_records(full, 0.1, 0.1, 42, None)
    valid = set(json.loads((dataset_dir / "valid_record_ids.json").read_text()))
    ref = [[r for r in p if r.id in valid and r.response_b] for p in ref]
    assert [[r.id for r in p] for p in (tr, va, te)] == [[r.id for r in p] for p in ref]
    tr2, va2, te2, _ = load_split_records(cfg, "none")   # mask disabled → only response_b filter
    assert len(tr2) + len(va2) + len(te2) == 32
    ds = RecordDataset(tr, smap)
    assert ds.record_ids() == [r.id for r in tr] and ds[0] is tr[0]


# ── 3. sampler fast path ─────────────────────────────────────────────────────
def test_enumerate_record_ids_fast_path(tmp_path):
    from prism.rl.sampler import enumerate_record_ids
    ds = RecordDataset([Record(f"id{i}", "s", "p", "r", "q", "b") for i in range(5)], {"s": 0})
    cache = tmp_path / "never_written.json"
    assert enumerate_record_ids(ds, cache_path=cache) == [f"id{i}" for i in range(5)]
    assert not cache.exists()


# ── 4. on-the-fly collate contract ────────────────────────────────────────────
class _FakeTokenizer:
    """Whitespace tokenizer without a chat template (exercises the fallback)."""
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": [hash(t) % 1000 + 1 for t in text.split()]}


def test_onthefly_collate_contract(monkeypatch):
    monkeypatch.delenv("PRISM_TARGET_MODEL", raising=False)
    from prism.activations import build_onthefly_collate_fn
    cfg = {"max_act_tokens": 4, "max_target_len": 64, "skip_prompt_b": True}
    recs = [
        Record("r1", "if_eval", "one two", "a b c d e f", "q", "- gt1"),
        Record("r2", "ultrachat", "one two three four", "a b", "q", "- gt2"),
    ]
    batch = build_onthefly_collate_fn(_FakeTokenizer(), cfg, {"if_eval": 0, "ultrachat": 1})(recs)
    for k in ("a_input_ids", "a_attention_mask", "a_prompt_only_lens", "a_response_token_counts",
              "chat_prefix_input_ids", "chat_prefix_attention_mask", "source_ids",
              "record_ids", "prompts_a", "responses_a", "ground_truth_b"):
        assert k in batch, k
    assert "precomputed_acts" not in batch and "act_masks" not in batch
    assert batch["record_ids"] == ["r1", "r2"] and batch["ground_truth_b"] == ["- gt1", "- gt2"]
    assert batch["source_ids"].tolist() == [0, 1]
    B, W = batch["a_input_ids"].shape
    assert B == 2 and batch["a_attention_mask"].shape == (B, W)
    # right-padded contexts; response token counts capped at max_act_tokens
    lens = batch["a_attention_mask"].sum(1).tolist()
    assert all(batch["a_attention_mask"][i, :lens[i]].all() and not batch["a_attention_mask"][i, lens[i]:].any() for i in range(B))
    assert batch["a_response_token_counts"].tolist() == [4, 2]
    assert (batch["a_prompt_only_lens"] <= batch["a_attention_mask"].sum(1)).all()
    # decoder prefix identical for every row (skip_prompt_b → empty user turn)
    assert torch.equal(batch["chat_prefix_input_ids"][0], batch["chat_prefix_input_ids"][1])
    assert batch["chat_prefix_attention_mask"].all()


# ── 5. RL data-mode validation ────────────────────────────────────────────────
def test_resolve_data_mode(tmp_path, monkeypatch):
    for k in list(os.environ):
        if k.startswith("PRISM_"):
            monkeypatch.delenv(k, raising=False)
    from prism.rl.train import resolve_data_mode
    jsonl = tmp_path / "x.jsonl"; jsonl.write_text("")
    cache = tmp_path / "cache"; cache.mkdir(); (cache / "manifest.json").write_text("{}")
    with pytest.raises(SystemExit, match="No training data"):
        resolve_data_mode({"precomputed_dir": None, "dataset_paths": []})
    with pytest.raises(SystemExit, match="choose one"):
        resolve_data_mode({"precomputed_dir": str(cache), "dataset_paths": [str(jsonl)]})
    with pytest.raises(SystemExit, match="not found"):
        resolve_data_mode({"precomputed_dir": None, "dataset_paths": [str(tmp_path / "missing.jsonl")]})
    with pytest.raises(SystemExit, match="manifest"):
        resolve_data_mode({"precomputed_dir": str(tmp_path / "nocache"), "dataset_paths": []})
    assert resolve_data_mode({"precomputed_dir": str(cache), "dataset_paths": []}) is False
    assert resolve_data_mode({"precomputed_dir": None, "dataset_paths": [str(jsonl)]}) is True


def test_env_override_dataset_paths(monkeypatch):
    monkeypatch.setenv("PRISM_DATASET_PATHS", os.pathsep.join(["/a.jsonl", "/b.jsonl"]))
    monkeypatch.setenv("PRISM_VALID_RECORD_IDS", "/mask.json")
    from prism.target_models import apply_profile_overlay as apply_profile_overrides
    cfg = {"dataset_paths": [], "valid_record_ids": None}
    apply_profile_overrides(cfg)
    assert cfg["dataset_paths"] == ["/a.jsonl", "/b.jsonl"] and cfg["valid_record_ids"] == "/mask.json"
