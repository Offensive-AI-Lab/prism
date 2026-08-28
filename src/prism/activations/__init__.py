"""Precompute Qwen activations into sharded safetensors for offline training."""

from prism.activations.dataset import (
    PrecomputedActivationDataset,
    ShardGroupedSampler,
    build_onthefly_collate_fn,
    build_precomputed_collate_fn,
)
from prism.activations.records import RecordDataset, load_split_records
from prism.activations.manifest import ExtractionManifest, SplitManifest
from prism.activations.token_select import select_positions, select_token_range

__all__ = [
    "ExtractionManifest",
    "PrecomputedActivationDataset",
    "RecordDataset",
    "ShardGroupedSampler",
    "build_onthefly_collate_fn",
    "load_split_records",
    "SplitManifest",
    "build_precomputed_collate_fn",
    "select_positions",
    "select_token_range",
]
