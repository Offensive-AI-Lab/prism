"""PyTorch Dataset for loading precomputed activations from safetensors shards."""

from __future__ import annotations

import json
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from safetensors import safe_open
from torch.utils.data import Dataset, Sampler

from prism.activations.manifest import ExtractionManifest, ShardInfo, SplitManifest

logger = logging.getLogger(__name__)

# Bound the per-process caches to avoid host-RAM blow-up across an epoch.
# ShardGroupedSampler iterates sequentially within a shard, so an LRU of
# size 2 covers the active shard plus one for the next. Slightly larger
# values give headroom for prefetch / multi-worker overlap.
_DEFAULT_SHARD_CACHE_SIZE = 4
_DEFAULT_META_CACHE_SIZE = 4


class PrecomputedActivationDataset(Dataset):
    """Loads precomputed activations from sharded safetensors files.

    Each sample returns a dict with:
        - ``activations``: ``{layer_idx: Tensor[N, H]}``
        - ``act_mask``: ``Tensor[N]`` (uint8, 1=valid)
        - ``record_id``, ``source_dataset``, ``prompt_a``, ``response_a``,
          ``prompt_b``, ``response_b`` (strings for collate/logging)
        - ``source_id``: int (from source_map)
    """

    def __init__(
        self,
        precomputed_dir: str,
        split: str = "train",
        layers: Optional[List[int]] = None,
        shard_cache_size: int = _DEFAULT_SHARD_CACHE_SIZE,
        meta_cache_size: int = _DEFAULT_META_CACHE_SIZE,
    ):
        """
        Args:
            precomputed_dir: Root directory containing manifest.json and train/val subdirs.
            split: ``"train"`` or ``"val"``.
            layers: Subset of layers to load.  If None, loads all available layers.
            shard_cache_size: Max number of safetensors handles kept open at once.
            meta_cache_size: Max number of parsed shard meta.json blobs cached.
        """
        root = Path(precomputed_dir)
        self.root = root
        self.split = split

        # Load top-level manifest for source_map and extraction params
        self.manifest = ExtractionManifest.load(root / "manifest.json")
        if self.manifest.status != "complete":
            raise RuntimeError(
                f"Extraction at {root} is not complete (status={self.manifest.status}). "
                "Run the extraction script first."
            )

        self.source_map = self.manifest.source_map
        self.all_layers = self.manifest.layers
        self.layers = layers if layers is not None else list(self.all_layers)

        # Validate requested layers are available
        available = set(self.all_layers)
        for l in self.layers:
            if l not in available:
                raise ValueError(
                    f"Layer {l} not in precomputed layers {self.all_layers}. "
                    "Re-extract with the needed layers."
                )

        # Load split manifest
        split_dir = root / split
        split_manifest_path = split_dir / "manifest.json"
        if not split_manifest_path.exists():
            raise FileNotFoundError(f"Split manifest not found: {split_manifest_path}")
        self.split_manifest = SplitManifest.load(split_manifest_path)

        # Optional record-id mask. When valid_record_ids.json is present in
        # the precomputed_dir, only those record_ids are exposed by the
        # dataset. Used to honour post-extraction quality filters (e.g. the
        # dataset cleaning applied a ≤6 bullet cap + template-leak
        # drops without re-sharding the activations).
        valid_ids_path = root / "valid_record_ids.json"
        valid_ids: Optional[set] = None
        if valid_ids_path.exists():
            try:
                valid_ids = set(json.loads(valid_ids_path.read_text(encoding="utf-8")))
                logger.info(
                    "Loaded valid_record_ids.json: %d ids — pre-filtering index",
                    len(valid_ids),
                )
            except Exception as e:
                logger.warning(
                    "Failed to parse %s, ignoring: %s", valid_ids_path, e
                )
                valid_ids = None

        # Build global index: [(shard_file, meta_file, offset_in_shard, shard_size)]
        self._index: List[Tuple[Path, Path, int, int]] = []
        self._shard_boundaries: List[int] = []  # cumulative record counts for shard-aware sampling
        cumulative = 0
        skipped_by_mask = 0
        for shard_info in self.split_manifest.shards:
            shard_path = split_dir / shard_info.file
            meta_path = split_dir / shard_info.meta_file

            if valid_ids is None:
                # Fast path — no mask, include every record in the shard.
                for offset in range(shard_info.records):
                    self._index.append((shard_path, meta_path, offset, shard_info.records))
                cumulative += shard_info.records
            else:
                # Masked path — load this shard's meta once to read record_ids,
                # then include only the offsets whose id is in the mask.
                meta_recs = json.loads(meta_path.read_text(encoding="utf-8"))
                kept_this_shard = 0
                for offset, rec in enumerate(meta_recs):
                    rid = rec.get("record_id", "")
                    if rid in valid_ids:
                        self._index.append((shard_path, meta_path, offset, shard_info.records))
                        kept_this_shard += 1
                    else:
                        skipped_by_mask += 1
                cumulative += kept_this_shard
            self._shard_boundaries.append(cumulative)
        if valid_ids is not None:
            logger.info(
                "valid_record_ids mask applied: kept %d records, skipped %d",
                len(self._index), skipped_by_mask,
            )

        # Bounded LRU caches. Without a cap these grow to one entry per
        # shard, and the safetensors mmaps alone (one per shard, ~256 MB
        # of mapped address space each at the typical 256-record shard
        # size) push the cgroup over its limit on long epochs.
        self._meta_cache: "OrderedDict[str, List[dict]]" = OrderedDict()
        self._shard_handles: "OrderedDict[str, object]" = OrderedDict()
        self._shard_cache_size = max(1, int(shard_cache_size))
        self._meta_cache_size = max(1, int(meta_cache_size))

        logger.info(
            "PrecomputedActivationDataset(%s/%s): %d records, %d shards, layers=%s",
            root.name, split, len(self._index), len(self.split_manifest.shards), self.layers,
        )

    def __len__(self) -> int:
        return len(self._index)

    def _get_meta(self, meta_path: Path) -> List[dict]:
        key = str(meta_path)
        cached = self._meta_cache.get(key)
        if cached is not None:
            self._meta_cache.move_to_end(key)
            return cached
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self._meta_cache[key] = meta
        while len(self._meta_cache) > self._meta_cache_size:
            self._meta_cache.popitem(last=False)
        return meta

    def _get_shard_handle(self, shard_path: Path):
        key = str(shard_path)
        handle = self._shard_handles.get(key)
        if handle is not None:
            self._shard_handles.move_to_end(key)
            return handle
        handle = safe_open(key, framework="pt", device="cpu")
        self._shard_handles[key] = handle
        while len(self._shard_handles) > self._shard_cache_size:
            _, evicted = self._shard_handles.popitem(last=False)
            # Drop the safetensors handle so the underlying mmap is
            # released. Without this, evicted shards keep their mapped
            # pages around for the page cache to chew on.
            close = getattr(evicted, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
            del evicted
        return handle

    def __getitem__(self, idx: int) -> dict:
        shard_path, meta_path, offset, _ = self._index[idx]

        # Load metadata
        meta_list = self._get_meta(meta_path)
        meta = meta_list[offset]

        # Load activations via memory-mapped safetensors
        f = self._get_shard_handle(shard_path)

        activations = {}
        for layer_idx in self.layers:
            tensor_name = f"activations_L{layer_idx}"
            # safe_open returns the full shard tensor; slice the record
            full = f.get_slice(tensor_name)
            activations[layer_idx] = full[offset]  # [N, H]

        mask_full = f.get_slice("act_masks")
        act_mask = mask_full[offset]  # [N]

        return {
            "activations": activations,
            "act_mask": act_mask,
            "record_id": meta.get("record_id", ""),
            "source_dataset": meta.get("source_dataset", "unknown"),
            "source_id": self.source_map.get(meta.get("source_dataset", "unknown"), -1),
            "prompt_a": meta.get("prompt_a", ""),
            "response_a": meta.get("response_a", ""),
            "response_b": meta.get("response_b", ""),
        }

    @property
    def shard_boundaries(self) -> List[int]:
        """Cumulative record counts per shard, for shard-aware sampling."""
        return self._shard_boundaries


class ShardGroupedSampler(Sampler):
    """Shuffles shard order but iterates sequentially within each shard.

    This maximizes sequential SSD reads and minimizes random seeks
    across different shard files.
    """

    def __init__(self, dataset: PrecomputedActivationDataset, shuffle: bool = True, seed: int = 0):
        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        boundaries = self.dataset.shard_boundaries
        num_shards = len(boundaries)

        # Build shard ranges: [(start_idx, end_idx), ...]
        shard_ranges = []
        prev = 0
        for end in boundaries:
            shard_ranges.append((prev, end))
            prev = end

        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            shard_order = torch.randperm(num_shards, generator=g).tolist()
        else:
            shard_order = list(range(num_shards))

        for shard_idx in shard_order:
            start, end = shard_ranges[shard_idx]
            yield from range(start, end)

    def __len__(self) -> int:
        return len(self.dataset)


def build_precomputed_collate_fn(
    tokenizer,
    cfg: dict,
    source_map: dict,
    layers: List[int],
    training: bool = True,
):
    """Build a collate function for PrecomputedActivationDataset.

    This collate function:
      1. Stacks precomputed activations and masks (already selected, fixed size).
      2. Tokenizes prompt_b + response_b for the decoder target.
      3. Passes through raw text fields for eval logging.

    The returned batch dict is compatible with the prism.sft and prism.rl
    training loops (with minor key mapping).
    """
    max_target_len = cfg.get("max_target_len", 1024)
    decoder_user_prompt = cfg.get("decoder_user_prompt", "")
    skip_prompt_b = cfg.get("skip_prompt_b", False)

    # Target-model chat quirks (e.g. Ministral's default system prompt): wrap the
    # decoder-prefix messages per the active profile. No-op for Qwen/gemma.
    from prism.target_models import get_profile, wrap_messages
    _profile = get_profile()

    def _normalize(output):
        if isinstance(output, dict):
            return output["input_ids"]
        if hasattr(output, "input_ids"):
            return output.input_ids
        return output

    def _apply_template(messages, add_generation_prompt: bool):
        messages = wrap_messages(messages, _profile)
        template_fn = getattr(tokenizer, "apply_chat_template", None)
        if template_fn is not None:
            try:
                return _normalize(
                    template_fn(messages, tokenize=True, add_generation_prompt=add_generation_prompt, enable_thinking=False)
                )
            except TypeError:
                return _normalize(
                    template_fn(messages, tokenize=True, add_generation_prompt=add_generation_prompt)
                )
        text_parts = [f"{m['role'].title()}: {m['content']}" for m in messages]
        if add_generation_prompt:
            text_parts.append("Assistant:")
        return tokenizer("\n".join(text_parts), add_special_tokens=True)["input_ids"]

    pad_id = tokenizer.pad_token_id

    def collate_fn(batch: List[dict]) -> dict:
        batch_size = len(batch)

        # --- Stack precomputed activations ---
        stacked_acts = {}
        for layer_idx in layers:
            stacked_acts[layer_idx] = torch.stack([sample["activations"][layer_idx] for sample in batch])
        act_masks = torch.stack([sample["act_mask"] for sample in batch])

        # --- Tokenize decoder target (prompt_b + response_b) ---
        chat_ids_list = []
        chat_prefix_ids_list = []
        answer_starts = []

        for sample in batch:
            if skip_prompt_b:
                prompt_b_text = ""
            else:
                prompt_b_text = decoder_user_prompt

            prefix_ids = _apply_template(
                [{"role": "user", "content": prompt_b_text}],
                add_generation_prompt=True,
            )
            full_ids = _apply_template(
                [
                    {"role": "user", "content": prompt_b_text},
                    {"role": "assistant", "content": sample["response_b"]},
                ],
                add_generation_prompt=False,
            )
            if len(full_ids) > max_target_len:
                full_ids = full_ids[:max_target_len]

            chat_ids_list.append(torch.tensor(full_ids, dtype=torch.long))
            chat_prefix_ids_list.append(torch.tensor(prefix_ids[:max_target_len], dtype=torch.long))
            answer_starts.append(min(len(prefix_ids), len(full_ids)))

        # Pad chat sequences
        max_chat_len = max(t.size(0) for t in chat_ids_list)
        chat_input_ids = torch.full((batch_size, max_chat_len), pad_id, dtype=torch.long)
        chat_attention_mask = torch.zeros((batch_size, max_chat_len), dtype=torch.long)
        labels = torch.full((batch_size, max_chat_len), -100, dtype=torch.long)

        max_prefix_len = max(t.size(0) for t in chat_prefix_ids_list)
        chat_prefix_input_ids = torch.full((batch_size, max_prefix_len), pad_id, dtype=torch.long)
        chat_prefix_attention_mask = torch.zeros((batch_size, max_prefix_len), dtype=torch.long)

        chat_lengths = []
        for i, ids in enumerate(chat_ids_list):
            length = ids.size(0)
            chat_input_ids[i, :length] = ids
            chat_attention_mask[i, :length] = 1
            chat_lengths.append(length)
            ans_start = answer_starts[i]
            if ans_start < length:
                labels[i, ans_start:length] = ids[ans_start:length]

        for i, ids in enumerate(chat_prefix_ids_list):
            length = ids.size(0)
            chat_prefix_input_ids[i, :length] = ids
            chat_prefix_attention_mask[i, :length] = 1

        # Source IDs
        source_ids = torch.tensor(
            [sample["source_id"] for sample in batch],
            dtype=torch.long,
        )

        return {
            # Precomputed activations (replaces the Qwen forward pass)
            "precomputed_acts": stacked_acts,  # {layer: [B, N, H]}
            "act_masks": act_masks,            # [B, N]

            # Decoder target
            "chat_input_ids": chat_input_ids,
            "chat_attention_mask": chat_attention_mask,
            "chat_prefix_input_ids": chat_prefix_input_ids,
            "chat_prefix_attention_mask": chat_prefix_attention_mask,
            "chat_lengths": torch.tensor(chat_lengths, dtype=torch.long),
            "answer_starts": torch.tensor(answer_starts, dtype=torch.long),
            "labels": labels,
            "source_ids": source_ids,

            # Raw text for eval logging (both key conventions for compatibility)
            "record_ids": [sample["record_id"] for sample in batch],
            "prompt_a_texts": [sample["prompt_a"] for sample in batch],
            "response_a_texts": [sample["response_a"] for sample in batch],
            "response_b_texts": [sample["response_b"] for sample in batch],
            # prism.sft compat aliases
            "prompts_a": [sample["prompt_a"] for sample in batch],
            "responses_a": [sample["response_a"] for sample in batch],
            "ground_truth_b": [sample["response_b"] for sample in batch],
        }

    return collate_fn


def build_onthefly_collate_fn(tokenizer, cfg: dict, source_map: dict):
    """Collate for :class:`prism.activations.records.RecordDataset` — the
    ON-THE-FLY counterpart of :func:`build_precomputed_collate_fn`.

    Emits the tokenised ``prompt_a + response_a`` context the trainer feeds
    to the resident target model (``prism.sft.model.extract_activations``
    selects the last ``max_act_tokens`` response tokens, exactly like the
    extractor's ``token_position=last``), plus the decoder prefix and the raw
    text fields the RL loop consumes. Keys match the precomputed batch except
    that ``precomputed_acts`` / ``act_masks`` are replaced by the ``a_*``
    fields.
    """
    max_act_tokens = int(cfg["max_act_tokens"])
    max_target_len = cfg.get("max_target_len", 1024)
    decoder_user_prompt = cfg.get("decoder_user_prompt", "")
    skip_prompt_b = cfg.get("skip_prompt_b", False)

    from prism.target_models import get_profile, wrap_messages
    _profile = get_profile(cfg.get("_target_profile"))

    def _normalize(output):
        if isinstance(output, dict):
            return output["input_ids"]
        if hasattr(output, "input_ids"):
            return output.input_ids
        return output

    def _apply_template(messages, add_generation_prompt: bool):
        messages = wrap_messages(messages, _profile)
        template_fn = getattr(tokenizer, "apply_chat_template", None)
        if template_fn is not None:
            try:
                return _normalize(
                    template_fn(messages, tokenize=True, add_generation_prompt=add_generation_prompt, enable_thinking=False)
                )
            except TypeError:
                return _normalize(
                    template_fn(messages, tokenize=True, add_generation_prompt=add_generation_prompt)
                )
        text_parts = [f"{m['role'].title()}: {m['content']}" for m in messages]
        if add_generation_prompt:
            text_parts.append("Assistant:")
        return tokenizer("\n".join(text_parts), add_special_tokens=True)["input_ids"]

    pad_id = tokenizer.pad_token_id

    def _pad(seqs: List[torch.Tensor]):
        width = max(t.size(0) for t in seqs)
        ids = torch.full((len(seqs), width), pad_id, dtype=torch.long)
        mask = torch.zeros((len(seqs), width), dtype=torch.long)
        for i, t in enumerate(seqs):
            ids[i, : t.size(0)] = t
            mask[i, : t.size(0)] = 1
        return ids, mask

    def collate_fn(batch: list) -> dict:
        # --- context for extraction: prompt_a + response_a (right-padded) ---
        a_ids, a_prompt_lens, a_resp_counts = [], [], []
        for rec in batch:
            prompt_only = _apply_template(
                [{"role": "user", "content": rec.prompt_a}], add_generation_prompt=True,
            )
            combined = _apply_template(
                [{"role": "user", "content": rec.prompt_a},
                 {"role": "assistant", "content": rec.response_a}],
                add_generation_prompt=False,
            )
            a_ids.append(torch.tensor(combined, dtype=torch.long))
            a_prompt_lens.append(len(prompt_only))
            a_resp_counts.append(min(len(combined) - len(prompt_only), max_act_tokens))
        a_input_ids, a_attention_mask = _pad(a_ids)

        # --- decoder prefix (same construction as the precomputed collate) ---
        prompt_b_text = "" if skip_prompt_b else decoder_user_prompt
        prefix_ids = _apply_template(
            [{"role": "user", "content": prompt_b_text}], add_generation_prompt=True,
        )[:max_target_len]
        prefix_t = torch.tensor(prefix_ids, dtype=torch.long)
        chat_prefix_input_ids, chat_prefix_attention_mask = _pad([prefix_t] * len(batch))

        source_ids = torch.tensor(
            [source_map.get(rec.source_dataset, -1) for rec in batch], dtype=torch.long,
        )
        return {
            "a_input_ids": a_input_ids,
            "a_attention_mask": a_attention_mask,
            "a_prompt_only_lens": torch.tensor(a_prompt_lens, dtype=torch.long),
            "a_response_token_counts": torch.tensor(a_resp_counts, dtype=torch.long),
            "chat_prefix_input_ids": chat_prefix_input_ids,
            "chat_prefix_attention_mask": chat_prefix_attention_mask,
            "source_ids": source_ids,
            "record_ids": [rec.id for rec in batch],
            "prompts_a": [rec.prompt_a for rec in batch],
            "responses_a": [rec.response_a for rec in batch],
            "ground_truth_b": [rec.response_b for rec in batch],
        }

    return collate_fn
