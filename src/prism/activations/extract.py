"""CLI for precomputing Qwen activations into sharded safetensors files.

Usage::

    uv run python -m prism.activations.extract \
        --dataset-paths data/oracle.jsonl data/other.jsonl \
        --model-id Qwen/Qwen3.5-9B \
        --layers 8,16,24,30 \
        --num-tokens 128 \
        --token-position last \
        --dtype bfloat16 \
        --records-per-shard 256 \
        --batch-size 32 \
        --val-ratio 0.05 \
        --seed 42 \
        --output-dir /persistent/precomputed/qwen3.5-9b-L8_16_24_30-last128/

Optimizations:
    - Length-sorted batching: records sorted by token count, batched together
      to minimize padding waste and maximize GPU utilization.
    - Prefetch tokenization: a background thread tokenizes the next batch
      while the GPU processes the current one.
    - Bulk GPU slicing: selected tokens gathered on GPU in one op per layer,
      then transferred to CPU in a single bulk copy.
    - Async shard writing: shards written to disk in a background thread
      so the GPU never waits on I/O.
    - Default batch_size=32: Qwen3.5-9B bf16 ~19GB + activations ~15GB
      leaves ~60GB headroom on a 96GB RTX 6000 Pro.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import random
import threading
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from safetensors.torch import save_file

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_records(dataset_paths: List[str]):
    """Load JSONL records, return (records, source_map)."""
    records = []
    source_names = set()

    for raw_path in dataset_paths:
        path = Path(raw_path)
        if not path.is_absolute():
            path = (Path.cwd() / raw_path).resolve()
        if not path.exists():
            logger.warning("Dataset file not found, skipping: %s", path)
            continue

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

                # Legacy field names (prompt_a/response_a/response_b) still load.
                prompt = obj.get("prompt") or obj.get("prompt_a", "")
                response = obj.get("response") or obj.get("response_a", "")
                # instruction_set is optional — pretraining-style records may omit it
                instruction_set = obj.get("instruction_set") or obj.get("response_b", "")
                if not all([prompt, response]):
                    continue

                # Paraphrase provenance: paraphrases of the same base example
                # must land in the same train/val/test split (DATA-9).
                metadata = obj.get("metadata")
                paraphrase_group_id = (
                    metadata.get("paraphrase_group_id") if isinstance(metadata, dict) else None
                )

                records.append({
                    "id": obj.get("id", ""),
                    "source_dataset": obj.get("source_dataset", "unknown"),
                    "prompt": prompt,
                    "response": response,
                    "instruction_set": instruction_set,
                    "paraphrase_group_id": paraphrase_group_id,
                })
                source_names.add(records[-1]["source_dataset"])
                count += 1

        logger.info("Loaded %s records from %s", f"{count:,}", path.name)

    source_map = {name: idx for idx, name in enumerate(sorted(source_names))}
    logger.info("Total records: %s across %d sources", f"{len(records):,}", len(source_map))
    return records, source_map


# Split helpers live in prism.activations.split so the on-the-fly trainers
# produce exactly the same train/val/test membership as this extractor.
from prism.activations.split import group_units as _group_units  # noqa: E402
from prism.activations.split import split_grouped as _split_grouped  # noqa: E402
from prism.activations.split import split_records as _split_records  # noqa: E402


# ---------------------------------------------------------------------------
# Model loading and hook registration
# ---------------------------------------------------------------------------

def _find_layers(model: nn.Module) -> nn.ModuleList:
    """Find the transformer layer stack in a HuggingFace model."""
    best = None

    def _walk(module, prefix=""):
        nonlocal best
        for name, child in module._modules.items():
            if child is None:
                continue
            path = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.ModuleList) and len(child) > 1:
                types = {type(c).__name__ for c in child}
                if len(types) == 1:
                    if best is None or len(child) > len(best[1]):
                        best = (path, child)
            _walk(child, path)

    _walk(model)
    if best is None:
        raise RuntimeError("Cannot locate transformer layers in model.")
    logger.info("Found %d transformer layers at: %s", len(best[1]), best[0])
    return best[1]


class _EarlyExit(Exception):
    """Raised inside a forward hook to abort the forward pass early."""


def _register_hooks(
    model: nn.Module,
    layer_indices: List[int],
    early_exit: bool = True,
) -> Tuple[dict, List]:
    """Register forward hooks on specified layers.

    Returns:
        activation_store: {layer_idx: tensor} populated after each forward pass.
        handles: List of hook handles for cleanup.
    """
    layers = _find_layers(model)
    sorted_layers = sorted(set(layer_indices))
    max_layer = sorted_layers[-1]

    for idx in sorted_layers:
        if idx < 0 or idx >= len(layers):
            raise ValueError(f"Layer {idx} out of range for model with {len(layers)} layers")

    activation_store: Dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(layer_idx: int):
        def hook(_module, _inputs, outputs):
            hidden = outputs[0] if isinstance(outputs, tuple) else outputs
            activation_store[layer_idx] = hidden.detach()
            if early_exit and layer_idx == max_layer:
                raise _EarlyExit()
        return hook

    for idx in sorted_layers:
        handles.append(layers[idx].register_forward_hook(make_hook(idx)))

    skipped = len(layers) - max_layer - 1 if early_exit else 0
    logger.info(
        "Hooks registered on layers %s (early_exit=%s, skipping %d layers + lm_head)",
        sorted_layers, early_exit, skipped,
    )
    return activation_store, handles


# ---------------------------------------------------------------------------
# Tokenization helpers
# ---------------------------------------------------------------------------

def _load_tokenizer(model_id: str):
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    tokenizer = processor.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _normalize_template_output(output):
    if isinstance(output, dict):
        return output["input_ids"]
    if hasattr(output, "input_ids"):
        return output.input_ids
    return output


def _apply_chat_template(tokenizer, messages, add_generation_prompt: bool):
    template_fn = getattr(tokenizer, "apply_chat_template", None)
    if template_fn is not None:
        try:
            return _normalize_template_output(
                template_fn(messages, tokenize=True, add_generation_prompt=add_generation_prompt, enable_thinking=False)
            )
        except TypeError:
            return _normalize_template_output(
                template_fn(messages, tokenize=True, add_generation_prompt=add_generation_prompt)
            )
    text_parts = [f"{m['role'].title()}: {m['content']}" for m in messages]
    if add_generation_prompt:
        text_parts.append("Assistant:")
    return tokenizer("\n".join(text_parts), add_special_tokens=True)["input_ids"]


def _tokenize_context(tokenizer, record: dict) -> Tuple[List[int], int]:
    """Tokenize prompt + response and return (token_ids, response_start)."""
    from prism.target_models import get_profile, wrap_messages
    _profile = get_profile()
    prompt_only_ids = _apply_chat_template(
        tokenizer,
        wrap_messages([{"role": "user", "content": record["prompt"]}], _profile),
        add_generation_prompt=True,
    )
    full_ids = _apply_chat_template(
        tokenizer,
        wrap_messages([
            {"role": "user", "content": record["prompt"]},
            {"role": "assistant", "content": record["response"]},
        ], _profile),
        add_generation_prompt=False,
    )
    response_start = min(len(prompt_only_ids), len(full_ids))
    return full_ids, response_start


# ---------------------------------------------------------------------------
# Pre-tokenization: sort by length + prefetch in background thread
# ---------------------------------------------------------------------------

def _pretokenize_records(tokenizer, records: List[dict]) -> List[dict]:
    """Tokenize all records, attach token IDs and response_start, sort by length.

    This runs once per split and enables:
      - Length-sorted batching (minimize padding waste)
      - No tokenization during GPU loop (prefetch is instant)
    """
    t0 = time.time()
    tokenized = []
    for record in records:
        full_ids, resp_start = _tokenize_context(tokenizer, record)
        tokenized.append({
            **record,
            "_token_ids": full_ids,
            "_response_start": resp_start,
            "_token_len": len(full_ids),
        })
    # Sort by token length so similarly-sized sequences are batched together
    tokenized.sort(key=lambda r: r["_token_len"])
    elapsed = time.time() - t0
    logger.info("Pre-tokenized %d records in %.1fs (sorted by length)", len(tokenized), elapsed)
    return tokenized


def _make_batches(records: List[dict], batch_size: int) -> List[List[dict]]:
    """Split pre-tokenized, length-sorted records into batches."""
    return [records[i:i + batch_size] for i in range(0, len(records), batch_size)]


def _prepare_batch_tensors(batch_records: List[dict], pad_id: int):
    """Build padded input_ids and attention_mask tensors from pre-tokenized records.

    Returns:
        input_ids: [B, max_len] (long, CPU)
        attention_mask: [B, max_len] (long, CPU)
        response_starts: list of ints
    """
    all_ids = [torch.tensor(r["_token_ids"], dtype=torch.long) for r in batch_records]
    response_starts = [r["_response_start"] for r in batch_records]

    max_len = max(t.size(0) for t in all_ids)
    B = len(batch_records)
    input_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((B, max_len), dtype=torch.long)
    for i, ids in enumerate(all_ids):
        input_ids[i, :ids.size(0)] = ids
        attention_mask[i, :ids.size(0)] = 1

    return input_ids, attention_mask, response_starts


class _BatchPrefetcher:
    """Prefetches the next batch's CPU tensors in a background thread.

    While the GPU processes batch N, thread prepares tensors for batch N+1.
    """

    def __init__(self, batches: List[List[dict]], pad_id: int, prefetch_depth: int = 2):
        self._batches = batches
        self._pad_id = pad_id
        self._queue: queue.Queue = queue.Queue(maxsize=prefetch_depth)
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self):
        for batch_records in self._batches:
            input_ids, attn_mask, resp_starts = _prepare_batch_tensors(batch_records, self._pad_id)
            self._queue.put((batch_records, input_ids, attn_mask, resp_starts))
        self._queue.put(None)  # sentinel

    def __iter__(self):
        while True:
            item = self._queue.get()
            if item is None:
                break
            yield item


# ---------------------------------------------------------------------------
# Shard writing (async)
# ---------------------------------------------------------------------------

def _write_shard(
    shard_path: Path,
    layer_activations: Dict[int, torch.Tensor],
    masks: torch.Tensor,
    meta_records: List[dict],
) -> None:
    """Write one shard (safetensors + meta.json) atomically.

    Expects pre-stacked tensors:
        layer_activations: {layer_idx: [S, N, H]} already padded
        masks: [S, N] uint8

    Both files land via a ``.tmp`` sibling + ``os.replace`` so a crash
    mid-write never leaves a truncated shard behind (DATA-3). The meta.json
    is moved into place first, so a visible ``.safetensors`` file always has
    its metadata alongside it.
    """
    tensors = {f"activations_L{l}": act for l, act in layer_activations.items()}
    tensors["act_masks"] = masks

    meta_path = shard_path.with_suffix(".meta.json")
    shard_tmp = shard_path.with_name(shard_path.name + ".tmp")
    meta_tmp = meta_path.with_name(meta_path.name + ".tmp")

    save_file(tensors, str(shard_tmp))
    meta_tmp.write_text(json.dumps(meta_records, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(meta_tmp, meta_path)
    os.replace(shard_tmp, shard_path)


class _AsyncShardWriter:
    """Writes shards in a background thread so GPU processing continues.

    Completion is observable: ``submit`` accepts an ``on_written`` callback
    that runs (on the writer thread) only after the shard is durably on
    disk, and ``drain()`` blocks until every queued shard has been written.
    This lets callers mark shards complete only once they exist on disk
    instead of when they are queued (DATA-3).
    """

    def __init__(self, max_pending: int = 3):
        self._queue: queue.Queue = queue.Queue(maxsize=max_pending)
        self._error: Exception | None = None
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self):
        while True:
            item = self._queue.get()
            if item is None:
                self._queue.task_done()
                break
            args, on_written = item
            try:
                _write_shard(*args)
                if on_written is not None:
                    on_written()
            except Exception as e:
                self._error = e
                logger.error("Async shard write failed: %s", e)
            finally:
                self._queue.task_done()

    def submit(self, shard_path, layer_activations, masks, meta_records, on_written=None):
        if self._error is not None:
            raise self._error
        self._queue.put(((shard_path, layer_activations, masks, meta_records), on_written))

    def drain(self):
        """Block until every submitted shard has been written to disk."""
        self._queue.join()
        if self._error is not None:
            raise self._error

    def shutdown(self):
        self._queue.put(None)
        self._thread.join()
        if self._error is not None:
            raise self._error


# ---------------------------------------------------------------------------
# Bulk GPU extraction
# ---------------------------------------------------------------------------

def _extract_batch_bulk(
    model: nn.Module,
    activation_store: dict,
    batch_records: List[dict],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    response_starts: List[int],
    layers: List[int],
    num_tokens: int,
    token_position: str,
    rng: random.Random | None,
    device: str,
) -> Tuple[Dict[int, torch.Tensor], torch.Tensor, List[dict]]:
    """Run one batch through frozen Qwen and extract selected activations.

    Returns pre-stacked tensors ready for shard accumulation:
        layer_acts: {layer_idx: Tensor[B, num_tokens, H]} (CPU, padded)
        masks: Tensor[B, num_tokens] (CPU, uint8)
        meta_list: [dict, ...] per record
    """
    from prism.activations.token_select import select_token_range

    B = input_ids.size(0)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    # Forward pass (hooks capture activations, _EarlyExit aborts after last layer)
    activation_store.clear()
    try:
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    except _EarlyExit:
        pass

    # Compute selection ranges on CPU (fast — just arithmetic)
    sel_ranges = []
    for i in range(B):
        total_len = int(attention_mask[i].sum().item())
        resp_start = response_starts[i]
        resp_len = max(total_len - resp_start, 0)
        sel_start, sel_end = select_token_range(resp_len, num_tokens, token_position, rng)
        sel_ranges.append((resp_start, sel_start, sel_end))

    # Bulk gather on GPU, then single D2H transfer per layer
    hidden_size = activation_store[layers[0]].shape[-1]
    layer_acts: Dict[int, torch.Tensor] = {}

    for layer_idx in layers:
        hidden = activation_store[layer_idx]  # [B, seq_len, H] on GPU
        padded = torch.zeros((B, num_tokens, hidden_size), dtype=hidden.dtype, device=device)

        for i, (resp_start, sel_start, sel_end) in enumerate(sel_ranges):
            n_sel = sel_end - sel_start
            if n_sel > 0:
                abs_start = resp_start + sel_start
                abs_end = resp_start + sel_end
                padded[i, :n_sel, :] = hidden[i, abs_start:abs_end, :]

        layer_acts[layer_idx] = padded.cpu()  # single bulk D2H transfer

    # Build masks on CPU
    masks = torch.zeros((B, num_tokens), dtype=torch.uint8)
    for i, (_, sel_start, sel_end) in enumerate(sel_ranges):
        n_sel = sel_end - sel_start
        if n_sel > 0:
            masks[i, :n_sel] = 1

    # Build metadata
    meta_list = []
    for i, record in enumerate(batch_records):
        resp_start, sel_start, sel_end = sel_ranges[i]
        total_len = int(attention_mask[i].sum().item())
        resp_len = max(total_len - resp_start, 0)
        meta_list.append({
            "record_id": record["id"],
            "source_dataset": record["source_dataset"],
            "prompt": record["prompt"],
            "response": record["response"],
            "instruction_set": record["instruction_set"],
            "response_a_total_tokens": resp_len,
            "selected_token_count": sel_end - sel_start,
            "selected_range": [sel_start, sel_end],
        })

    # Free GPU activation memory immediately
    activation_store.clear()

    return layer_acts, masks, meta_list


# ---------------------------------------------------------------------------
# Shard accumulator — stacks tensors, flushes when full
# ---------------------------------------------------------------------------

class _ShardAccumulator:
    """Accumulates batch results and flushes complete shards via async writer."""

    def __init__(
        self,
        layers: List[int],
        num_tokens: int,
        records_per_shard: int,
        dtype: torch.dtype,
        split_dir: Path,
        shard_writer: _AsyncShardWriter,
        start_shard_idx: int = 0,
    ):
        self.layers = layers
        self.num_tokens = num_tokens
        self.records_per_shard = records_per_shard
        self.dtype = dtype
        self.split_dir = split_dir
        self.writer = shard_writer
        self.shard_idx = start_shard_idx

        # Buffers: pre-stacked tensors from batches
        self._act_chunks: Dict[int, List[torch.Tensor]] = {l: [] for l in layers}
        self._mask_chunks: List[torch.Tensor] = []
        self._meta: List[dict] = []
        self.flushed_shards: list = []
        self.total_records = 0

        # Number of shards durably written to disk for this split (updated by
        # the async writer only after os.replace lands the files — DATA-3).
        self._durable_lock = threading.Lock()
        self._durable_count = start_shard_idx

    @property
    def buffered_count(self) -> int:
        return len(self._meta)

    @property
    def durable_shard_count(self) -> int:
        """Shards of this split confirmed on disk (safe to record as complete)."""
        with self._durable_lock:
            return self._durable_count

    def add_batch(self, layer_acts: Dict[int, torch.Tensor], masks: torch.Tensor, meta: List[dict]):
        """Add a batch result. Tensors are already [B, N, H] and [B, N]."""
        for l in self.layers:
            self._act_chunks[l].append(layer_acts[l])
        self._mask_chunks.append(masks)
        self._meta.extend(meta)

        # Flush complete shards
        while len(self._meta) >= self.records_per_shard:
            self._flush(self.records_per_shard)

    def flush_remaining(self):
        """Flush any leftover records as a partial shard."""
        if self._meta:
            self._flush(len(self._meta))

    def _flush(self, count: int):
        """Flush `count` records as one shard."""
        flush_acts = {}
        for l in self.layers:
            cat = torch.cat(self._act_chunks[l], dim=0)
            flush_acts[l] = cat[:count].to(self.dtype)
            remainder = cat[count:]
            self._act_chunks[l] = [remainder] if remainder.size(0) > 0 else []

        cat_masks = torch.cat(self._mask_chunks, dim=0)
        flush_masks = cat_masks[:count]
        remainder_masks = cat_masks[count:]
        self._mask_chunks = [remainder_masks] if remainder_masks.size(0) > 0 else []

        flush_meta = self._meta[:count]
        self._meta = self._meta[count:]

        shard_name = f"shard-{self.shard_idx:05d}.safetensors"
        meta_name = f"shard-{self.shard_idx:05d}.meta.json"
        shard_path = self.split_dir / shard_name

        completed_idx = self.shard_idx

        def _mark_durable(idx: int = completed_idx) -> None:
            # Runs on the writer thread after the shard hits disk. Writes
            # complete in FIFO submit order, so shards 0..idx are all durable.
            with self._durable_lock:
                if idx + 1 > self._durable_count:
                    self._durable_count = idx + 1

        from prism.activations.manifest import ShardInfo
        self.writer.submit(shard_path, flush_acts, flush_masks, flush_meta,
                           on_written=_mark_durable)
        self.flushed_shards.append(ShardInfo(file=shard_name, meta_file=meta_name, records=count))
        self.total_records += count
        self.shard_idx += 1


# ---------------------------------------------------------------------------
# Main extraction loop
# ---------------------------------------------------------------------------

def extract(
    dataset_paths: List[str],
    model_id: str,
    layers: List[int],
    num_tokens: int,
    token_position: str,
    dtype_str: str,
    records_per_shard: int,
    batch_size: int,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    output_dir: str,
    hf_cache_dir: str | None = None,
    stratify_by: str | None = None,
    allow_resume: bool = False,
) -> None:
    """Main extraction entry point."""
    from prism.activations.manifest import (
        ExtractionManifest,
        ShardInfo,
        SplitManifest,
        finalize_manifest,
        load_or_create_manifest,
    )

    output_path = Path(output_dir)
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map[dtype_str]

    # Target-model override (PRISM_TARGET_MODEL): gemma2-9b / ministral3-8b rewrite
    # the model id + loader class. No-op for Qwen (legacy path preserved verbatim).
    from prism.target_models import is_active as _is_active, get_profile as _get_profile
    if _is_active():
        model_id = _get_profile()["model_id"]

    # Load records and split
    records, source_map = _load_records(dataset_paths)
    if not records:
        logger.error("No records loaded. Exiting.")
        return

    if stratify_by:
        logger.info("Stratifying split by %r", stratify_by)
    train_records, val_records, test_records = _split_records(
        records, val_ratio, test_ratio, seed, stratify_by=stratify_by,
    )
    logger.info("Split: %d train, %d val, %d test", len(train_records), len(val_records), len(test_records))

    # Load or create manifest (for resume)
    manifest = load_or_create_manifest(
        output_dir=output_path,
        model_id=model_id,
        layers=layers,
        num_tokens=num_tokens,
        token_position=token_position,
        dtype=dtype_str,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        split_seed=seed,
        random_seed=seed if token_position == "random" else None,
        dataset_paths=[str(Path(p).resolve()) for p in dataset_paths],
        total_records=len(records),
        train_records=len(train_records),
        val_records=len(val_records),
        test_records=len(test_records),
        records_per_shard=records_per_shard,
        source_map=source_map,
    )

    if manifest.status == "complete":
        logger.info("Extraction already complete. Use a different output directory to re-extract.")
        return

    # Determine resume point (whole completed shards only — see the split
    # loop below for how the skip is applied in shard/write order, DATA-3)
    skip_train_records = manifest.completed_train_shards * records_per_shard
    skip_val_records = manifest.completed_val_shards * records_per_shard
    skip_test_records = manifest.completed_test_shards * records_per_shard
    if skip_train_records + skip_val_records + skip_test_records > 0:
        if not allow_resume:
            raise RuntimeError(
                f"{output_path} holds a PARTIAL extraction "
                f"({manifest.completed_train_shards} train / "
                f"{manifest.completed_val_shards} val / "
                f"{manifest.completed_test_shards} test shards done). Pass "
                "--allow-resume to continue from the last durably written shard "
                "(same arguments enforced via the manifest), or use a fresh "
                "--output-dir to re-extract from scratch."
            )
        if token_position == "random":
            raise RuntimeError(
                "Cannot resume an extraction that used --token-position random: "
                "the token-sampling RNG stream cannot be restored mid-run, so "
                "resumed shards would select different token positions than a "
                "single-pass run. Re-extract from scratch into a fresh --output-dir."
            )
        if stratify_by:
            raise RuntimeError(
                "Cannot resume a --stratify-by extraction: the manifest does not "
                "record the stratification setting of the original run, so split "
                "compatibility cannot be verified. Re-extract from scratch into a "
                "fresh --output-dir."
            )
        logger.info(
            "Resuming: skipping %d train (%d shards), %d val (%d shards), %d test (%d shards)",
            skip_train_records, manifest.completed_train_shards,
            skip_val_records, manifest.completed_val_shards,
            skip_test_records, manifest.completed_test_shards,
        )

    # Set up HF cache
    if hf_cache_dir:
        Path(hf_cache_dir).mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", hf_cache_dir)

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading model %s on %s", model_id, device)

    if _is_active():
        # gemma-2 (CausalLM/eager) / Ministral-3 (ImageTextToText/sdpa) via profile.
        from prism.target_models import load_target_model
        model, tokenizer, _profile = load_target_model(
            device, dtype=dtype, for_extraction=True
        )
    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map=device,
        )
        tokenizer = _load_tokenizer(model_id)

    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    pad_id = tokenizer.pad_token_id

    # Register hooks with early exit
    activation_store, handles = _register_hooks(model, layers, early_exit=True)

    rng = random.Random(seed) if token_position == "random" else None
    shard_writer = _AsyncShardWriter(max_pending=3)

    try:
        for split_name, split_records_list, skip_records, completed_shards_attr in [
            ("train", train_records, skip_train_records, "completed_train_shards"),
            ("val", val_records, skip_val_records, "completed_val_shards"),
            ("test", test_records, skip_test_records, "completed_test_shards"),
        ]:
            split_dir = output_path / split_name
            split_dir.mkdir(parents=True, exist_ok=True)

            if skip_records >= len(split_records_list):
                logger.info("Split %s: all records already processed", split_name)
            else:
                # Pre-tokenize and length-sort the FULL split, then drop the
                # records already covered by completed shards. Shards are
                # written in this length-sorted order, so a resume must skip
                # in the same order — skipping in shuffle order (the old
                # behaviour) duplicated and dropped records (DATA-3).
                tokenized = _pretokenize_records(tokenizer, split_records_list)
                if skip_records > 0:
                    tokenized = tokenized[skip_records:]
                num_todo = len(tokenized)
                batches = _make_batches(tokenized, batch_size)
                logger.info(
                    "Processing %s split: %d records, %d batches (batch_size=%d)",
                    split_name, num_todo, len(batches), batch_size,
                )

                # Set up accumulator
                start_shard = getattr(manifest, completed_shards_attr)
                accumulator = _ShardAccumulator(
                    layers=layers,
                    num_tokens=num_tokens,
                    records_per_shard=records_per_shard,
                    dtype=dtype,
                    split_dir=split_dir,
                    shard_writer=shard_writer,
                    start_shard_idx=start_shard,
                )

                # Prefetched batch iteration
                prefetcher = _BatchPrefetcher(batches, pad_id, prefetch_depth=2)
                t_split_start = time.time()

                for batch_idx, (batch_records, input_ids, attn_mask, resp_starts) in enumerate(prefetcher):
                    layer_acts, masks, meta = _extract_batch_bulk(
                        model=model,
                        activation_store=activation_store,
                        batch_records=batch_records,
                        input_ids=input_ids,
                        attention_mask=attn_mask,
                        response_starts=resp_starts,
                        layers=layers,
                        num_tokens=num_tokens,
                        token_position=token_position,
                        rng=rng,
                        device=device,
                    )

                    if manifest.hidden_size == 0:
                        manifest.hidden_size = layer_acts[layers[0]].shape[-1]

                    accumulator.add_batch(layer_acts, masks, meta)

                    # Update manifest for resume — record only shards the
                    # async writer has durably written to disk. Marking
                    # queued-but-unwritten shards complete corrupted resume
                    # after a crash (DATA-3).
                    durable = accumulator.durable_shard_count
                    if durable > getattr(manifest, completed_shards_attr):
                        setattr(manifest, completed_shards_attr, durable)
                        manifest.completed_records = (
                            manifest.completed_train_shards * records_per_shard
                            + manifest.completed_val_shards * records_per_shard
                            + manifest.completed_test_shards * records_per_shard
                        )
                        manifest.save(output_path / "manifest.json")

                    # Progress logging
                    processed = (batch_idx + 1) * batch_size
                    if (batch_idx + 1) % 10 == 0 or batch_idx == len(batches) - 1:
                        elapsed = time.time() - t_split_start
                        recs_per_sec = min(processed, num_todo) / max(elapsed, 0.1)
                        eta = (num_todo - min(processed, num_todo)) / max(recs_per_sec, 0.1)
                        logger.info(
                            "%s progress: %d/%d records (%.1f%%) | %.1f recs/s | ETA %.0fs | shards: %d",
                            split_name,
                            min(processed, num_todo), num_todo,
                            100.0 * min(processed, num_todo) / num_todo,
                            recs_per_sec, eta,
                            accumulator.shard_idx,
                        )

                # Flush remaining, then wait until every queued shard of this
                # split is durably on disk before recording completion and
                # scanning the directory (DATA-3 durable-write barrier).
                accumulator.flush_remaining()
                shard_writer.drain()
                setattr(manifest, completed_shards_attr, accumulator.shard_idx)
                manifest.completed_records = (
                    manifest.completed_train_shards * records_per_shard
                    + manifest.completed_val_shards * records_per_shard
                    + manifest.completed_test_shards * records_per_shard
                )
                manifest.save(output_path / "manifest.json")

                elapsed = time.time() - t_split_start
                logger.info(
                    "%s complete: %d records, %d shards in %.1fs (%.1f recs/s)",
                    split_name, accumulator.total_records, len(accumulator.flushed_shards),
                    elapsed, accumulator.total_records / max(elapsed, 0.1),
                )

            # Write split manifest by scanning the directory — robust to
            # resume, and runs even when this session had nothing left to
            # process for the split, so finalize always finds a manifest.
            all_shards: list[ShardInfo] = []
            shard_files = sorted(
                p for p in split_dir.iterdir()
                if p.name.startswith("shard-") and p.suffix == ".safetensors"
            )
            for shard_path in shard_files:
                meta_path = shard_path.with_suffix(".meta.json")
                records = len(json.loads(meta_path.read_text(encoding="utf-8")))
                all_shards.append(ShardInfo(
                    file=shard_path.name,
                    meta_file=meta_path.name,
                    records=records,
                ))
            split_manifest = SplitManifest(
                num_records=sum(s.records for s in all_shards),
                num_shards=len(all_shards),
                shards=all_shards,
            )
            split_manifest.save(split_dir / "manifest.json")

        # Finalize
        train_manifest = SplitManifest.load(output_path / "train" / "manifest.json")
        val_manifest = SplitManifest.load(output_path / "val" / "manifest.json")
        test_manifest = SplitManifest.load(output_path / "test" / "manifest.json")
        finalize_manifest(output_path, manifest, train_manifest, val_manifest, test_manifest)

    finally:
        shard_writer.shutdown()
        for handle in handles:
            handle.remove()
        logger.info("Hooks removed, shard writer shut down.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Precompute target-model activations (hook-layer residual stream) into sharded safetensors files.",
    )
    parser.add_argument("--dataset-paths", nargs="+", required=True, help="JSONL dataset files")
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-9B", help="HuggingFace model ID")
    parser.add_argument("--layers", default="16", help="Comma-separated layer indices to hook (the recipes pass the profile's hook layer: 16 / 21 / 17)")
    parser.add_argument("--num-tokens", type=int, default=128, help="Number of response tokens to select")
    parser.add_argument("--token-position", default="last", choices=["last", "first", "middle", "random"],
                        help="Token selection strategy")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--records-per-shard", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", required=True, help="Output directory for shards")
    parser.add_argument("--hf-cache-dir", default=None, help="HuggingFace cache directory")
    parser.add_argument("--stratify-by", default=None,
                        help="Record field to stratify train/val/test splits over "
                             "(e.g. 'source_dataset'). Default: random split.")
    parser.add_argument("--model-profile", default=None,
                        help="target_models profile (qwen3.5-9b / gemma2-9b / ministral3-8b). "
                             "Sets PRISM_TARGET_MODEL; overrides --model-id + loader class.")
    parser.add_argument("--allow-resume", action="store_true",
                        help="Resume a partial extraction from the last durably "
                             "written shard. Requires identical arguments to the "
                             "original run (enforced via the manifest); refused "
                             "for --token-position random or --stratify-by.")

    args = parser.parse_args()
    if args.model_profile:
        os.environ["PRISM_TARGET_MODEL"] = args.model_profile
    layers = [int(x.strip()) for x in args.layers.split(",")]

    extract(
        dataset_paths=args.dataset_paths,
        model_id=args.model_id,
        layers=layers,
        num_tokens=args.num_tokens,
        token_position=args.token_position,
        dtype_str=args.dtype,
        records_per_shard=args.records_per_shard,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        output_dir=args.output_dir,
        hf_cache_dir=args.hf_cache_dir,
        stratify_by=args.stratify_by,
        allow_resume=args.allow_resume,
    )


if __name__ == "__main__":
    main()
