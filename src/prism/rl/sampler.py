"""Prioritized sampler for the RL trainer (historical notes §1.8 "Fix B").

Each prompt has a sampling weight inversely proportional to its current
running-mean reward, so the policy spends more rollout budget on prompts
where it's currently weak — *without* fully excluding the easy ones (which
keeps forgetting bounded).

Pieces:
    PrioritizedSampler — torch Sampler. Weighted-multinomial-with-replacement
        over a mutable weight tensor. The tensor lives in a shared list-of-one
        container so the trainer can swap it in between dataloader epochs
        without recreating the DataLoader.

    RunningRewardTracker — EMA per-record-id reward. Updated from each
        successful batch's rewards in train.py. Skips updates when the
        judge returned errors (avoids poisoning the running mean with
        forced-zero rewards from a dead vLLM endpoint).

    build_weights(record_ids_in_order, tracker, …) — converts tracker state
        to a 1-D weight tensor indexed by global dataset position. Clamped at
        `max_weight` so a single bad judge call can't make one prompt swallow
        the sampling budget.

    enumerate_record_ids(dataset, cache_path=…) — walks the precomputed
        dataset's `_get_meta` cache (NOT `__getitem__` — that loads activations)
        and returns `[record_id_at_global_idx_0, record_id_at_global_idx_1, …]`.
        Cached to JSON on first call so resume launches skip the scan.

The sampler is opt-in via cfg["prioritized_sampling"]; if False, the trainer
keeps using the existing ShardGroupedSampler and nothing here is touched.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import Sampler

logger = logging.getLogger(__name__)


class PrioritizedSampler(Sampler[int]):
    """Weighted-multinomial-with-replacement sampler with mutable weights.

    The trainer swaps a new weight tensor into `weights_ref[0]` at val-pass
    cadence. The sampler reads `weights_ref[0]` at the start of each
    `__iter__` call (i.e., each "dataloader epoch"), so updates take effect
    on the next epoch boundary.

    Each `__iter__` yields `samples_per_iter` indices, where
    `samples_per_iter` is sized so an epoch covers `eval_every` opt steps
    (= `eval_every * batch_size` records). At the end of each epoch the
    main training for-loop naturally calls `__iter__` again, picking up the
    new weights.

    Sampled with REPLACEMENT — hard prompts can appear multiple times in
    one epoch, and over a long run an easy prompt may go un-sampled for
    stretches. That's the intended behavior of priority sampling.
    """

    def __init__(
        self,
        num_records: int,
        weights_ref: list[torch.Tensor],
        samples_per_iter: int,
        seed: int = 0,
    ):
        self.num_records = int(num_records)
        self.weights_ref = weights_ref
        self.samples_per_iter = int(samples_per_iter)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        weights = self.weights_ref[0]
        if weights.numel() != self.num_records:
            raise RuntimeError(
                f"PrioritizedSampler: weights length {weights.numel()} != "
                f"num_records {self.num_records}. Trainer swapped a "
                f"mismatched tensor."
            )
        # Defensive: torch.multinomial requires positive total mass.
        # Easy to hit if a misconfigured (eps=0, default_mean=0) build_weights
        # produces all zeros, or if the trainer accidentally swaps in a
        # zeroed-out snapshot. Fall back to uniform so training continues.
        total = float(weights.sum().item())
        if not (total > 0.0):
            logger.warning(
                "PrioritizedSampler: weight sum is %s (<=0) — falling back to "
                "uniform sampling for this epoch.", total,
            )
            weights = torch.ones_like(weights)
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        # torch.multinomial requires a CPU tensor for large num_samples paths.
        weights_cpu = weights.detach().cpu().float()
        indices = torch.multinomial(
            weights_cpu, self.samples_per_iter, replacement=True, generator=g,
        )
        # NOTE: auto-increment so each dataloader epoch gets a fresh seed.
        # On `--resume`, set this explicitly via `set_epoch(opt_step // rebuild_every)`
        # before training starts — otherwise the first epoch post-resume
        # replays the indices of the original run's first epoch.
        self.epoch += 1
        return iter(indices.tolist())

    def __len__(self) -> int:
        return self.samples_per_iter


class RunningRewardTracker:
    """EMA-smoothed per-record reward.

    `alpha` is the weight on the newest observation (alpha=1 → no smoothing;
    alpha=0 → never update). Default 0.3 ≈ effective half-life of 2 visits.
    """

    def __init__(self, alpha: float = 0.3):
        self.alpha = float(alpha)
        self.values: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    def update(self, record_id: str, reward: float) -> None:
        r = float(reward)
        if record_id in self.values:
            self.values[record_id] = (
                self.alpha * r + (1.0 - self.alpha) * self.values[record_id]
            )
        else:
            self.values[record_id] = r
        self.counts[record_id] = self.counts.get(record_id, 0) + 1

    def update_batch(
        self,
        record_ids: list[str],
        rewards: list[list[float]],
        skip_if_all_zero: bool = False,
    ) -> int:
        """Updates the tracker from a `_complete_deferred` batch.

        Returns the number of records actually updated.
        `skip_if_all_zero` should be True when n_judge_errors > 0 for the
        batch — protects the tracker from a dead-judge train round.
        """
        if len(record_ids) != len(rewards):
            raise ValueError(
                f"record_ids ({len(record_ids)}) and rewards ({len(rewards)}) "
                f"have different lengths."
            )
        n = 0
        for rid, r_per_candidate in zip(record_ids, rewards):
            if not r_per_candidate:
                continue
            mean_r = float(sum(r_per_candidate)) / float(len(r_per_candidate))
            if skip_if_all_zero and mean_r == 0.0 and all(
                r == 0.0 for r in r_per_candidate
            ):
                continue
            self.update(rid, mean_r)
            n += 1
        return n

    def get(self, record_id: str, default: float = 0.5) -> float:
        return self.values.get(record_id, default)

    def state_dict(self) -> dict:
        return {
            "values": dict(self.values),
            "counts": dict(self.counts),
            "alpha": self.alpha,
        }

    def load_state_dict(self, state: dict) -> None:
        self.values = dict(state.get("values", {}))
        self.counts = dict(state.get("counts", {}))
        self.alpha = float(state.get("alpha", self.alpha))

    @classmethod
    def from_judge_traces(
        cls,
        trace_path: str | Path,
        alpha: float = 0.3,
    ) -> "RunningRewardTracker":
        """Rebuild a tracker by replaying a judge_traces.jsonl file.

        Each line is one (step, record_id, candidate_idx, reward) record.
        We group by (step, record_id), compute the per-group mean reward,
        then replay updates in step order so the EMA is well-formed.

        Used on `--resume` to recover the tracker state of a killed run
        instead of starting from a blank slate.
        """
        tracker = cls(alpha=alpha)
        path = Path(trace_path)
        if not path.exists():
            logger.warning("judge_traces not found at %s — empty tracker", trace_path)
            return tracker

        by_step_rec: dict[tuple[int, str], list[float]] = defaultdict(list)
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    by_step_rec[(int(d["step"]), str(d["record_id"]))].append(
                        float(d["reward"])
                    )
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    continue

        # Replay in step order so EMA reflects the actual training trajectory.
        for (_step, rid), rewards in sorted(by_step_rec.items(), key=lambda x: x[0][0]):
            if not rewards:
                continue
            # Also screen out all-zero groups — they were judge errors,
            # not real signal (see update_batch's skip_if_all_zero).
            if all(r == 0.0 for r in rewards):
                continue
            mean_r = sum(rewards) / len(rewards)
            tracker.update(rid, mean_r)
        logger.info(
            "RunningRewardTracker rebuilt from %s: %d unique records, alpha=%.2f",
            trace_path, len(tracker.values), tracker.alpha,
        )
        return tracker


def build_weights(
    record_ids_in_order: list[str],
    tracker: RunningRewardTracker,
    epsilon: float,
    max_weight: float,
    default_mean: float,
) -> torch.Tensor:
    """Convert tracker state into a per-global-index weight tensor.

    weights[i] = min(1 / (running_mean(record_ids_in_order[i]) + ε), max_weight)

    Unvisited records get `default_mean` (neutral, e.g. 0.5) → weight ≈ 1.67.
    """
    n = len(record_ids_in_order)
    weights = torch.empty(n, dtype=torch.float32)
    for i, rid in enumerate(record_ids_in_order):
        r = tracker.get(rid, default=default_mean)
        w = 1.0 / (max(r, 0.0) + epsilon)
        weights[i] = min(w, max_weight)
    return weights


def enumerate_record_ids(
    dataset,
    cache_path: str | Path | None = None,
) -> list[str]:
    """Walk dataset._index, reading shard meta files only (no activations).

    The result is `[record_id_at_global_idx_0, …]`, which is the column index
    of the weight tensor used by PrioritizedSampler.

    Cached to disk (JSON) at `cache_path`. On a cache hit with matching length
    we skip the walk entirely — saves ~minutes on resume launches.
    """
    n = len(dataset)
    if hasattr(dataset, "record_ids"):
        # RecordDataset (on-the-fly path): ids are already in memory.
        record_ids = list(dataset.record_ids())
        assert len(record_ids) == n
        return record_ids
    if cache_path is not None:
        cache = Path(cache_path)
        if cache.exists():
            try:
                cached = json.loads(cache.read_text())
                if isinstance(cached, list) and len(cached) == n:
                    logger.info(
                        "enumerate_record_ids: loaded %d ids from cache %s",
                        n, cache,
                    )
                    return cached
                logger.warning(
                    "enumerate_record_ids: cache length mismatch "
                    "(%d vs dataset %d) — rebuilding",
                    len(cached) if isinstance(cached, list) else -1, n,
                )
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("enumerate_record_ids: cache read failed (%s) — rebuilding", e)

    record_ids: list[str] = []
    last_logged = 0
    n_missing = 0
    for i in range(n):
        shard_path, meta_path, offset, _shard_size = dataset._index[i]
        meta_list = dataset._get_meta(meta_path)
        rid = meta_list[offset].get("record_id", "")
        if not rid:
            n_missing += 1
        record_ids.append(rid)
        if i - last_logged >= 50000:
            logger.info("enumerate_record_ids: %d / %d", i, n)
            last_logged = i
    if n_missing > 0:
        # Multiple records colliding on the empty-string key would corrupt
        # the per-prompt running mean. Flag loudly — but proceed so the run
        # at least gets uniform-ish weights, since the alternative is a hard
        # crash on an otherwise-launched run.
        logger.warning(
            "enumerate_record_ids: %d/%d records have empty record_id — "
            "they will all share the same tracker key. Check dataset meta.",
            n_missing, n,
        )

    if cache_path is not None:
        try:
            Path(cache_path).write_text(json.dumps(record_ids))
            logger.info(
                "enumerate_record_ids: cached %d ids → %s",
                len(record_ids), cache_path,
            )
        except OSError as e:
            logger.warning("enumerate_record_ids: cache write failed: %s", e)
    return record_ids
