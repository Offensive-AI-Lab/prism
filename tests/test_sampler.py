"""Tests for prism/rl/sampler.py — PrioritizedSampler + RunningRewardTracker.

Smoke-coverage: weights are read at iter time, hard prompts dominate the
sample stream, the tracker computes the right EMA, build_weights clamps
correctly, and the judge_traces.jsonl replay rebuilds the right state.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from prism.rl.sampler import (
    PrioritizedSampler,
    RunningRewardTracker,
    build_weights,
)


# ─── RunningRewardTracker ────────────────────────────────────────────────────


def test_tracker_first_update_takes_observation_directly():
    t = RunningRewardTracker(alpha=0.3)
    t.update("a", 0.5)
    assert t.values["a"] == 0.5
    assert t.counts["a"] == 1


def test_tracker_ema_after_second_update():
    t = RunningRewardTracker(alpha=0.3)
    t.update("a", 0.5)
    t.update("a", 1.0)
    # 0.3 * 1.0 + 0.7 * 0.5 = 0.65
    assert t.values["a"] == pytest.approx(0.65)
    assert t.counts["a"] == 2


def test_tracker_get_default_for_unknown_record():
    t = RunningRewardTracker(alpha=0.3)
    assert t.get("never_seen", default=0.42) == 0.42


def test_tracker_update_batch_skips_judge_errors():
    """All-zero rewards from a dead judge endpoint should NOT poison tracker."""
    t = RunningRewardTracker(alpha=0.3)
    rewards = [[0.0, 0.0, 0.0], [0.5, 0.7, 0.6]]
    record_ids = ["error_prompt", "good_prompt"]
    n_updated = t.update_batch(record_ids, rewards, skip_if_all_zero=True)
    assert n_updated == 1
    assert "error_prompt" not in t.values
    assert "good_prompt" in t.values
    assert t.values["good_prompt"] == pytest.approx(0.6)


def test_tracker_update_batch_includes_zeros_when_not_skipped():
    t = RunningRewardTracker(alpha=0.3)
    rewards = [[0.0, 0.0]]
    record_ids = ["a"]
    n_updated = t.update_batch(record_ids, rewards, skip_if_all_zero=False)
    assert n_updated == 1
    assert t.values["a"] == 0.0


def test_tracker_state_dict_roundtrip():
    t1 = RunningRewardTracker(alpha=0.25)
    t1.update("a", 0.5)
    t1.update("b", 0.9)
    state = t1.state_dict()

    t2 = RunningRewardTracker(alpha=0.99)  # different alpha, will be overwritten
    t2.load_state_dict(state)
    assert t2.values == t1.values
    assert t2.counts == t1.counts
    assert t2.alpha == 0.25


# ─── build_weights ────────────────────────────────────────────────────────────


def test_build_weights_unvisited_uses_default_mean():
    t = RunningRewardTracker(alpha=0.3)
    w = build_weights(
        ["x", "y", "z"], t, epsilon=0.1, max_weight=5.0, default_mean=0.5,
    )
    # All unvisited → all default. 1 / (0.5 + 0.1) = 1.667
    assert torch.allclose(w, torch.full_like(w, 1.0 / 0.6), atol=1e-5)


def test_build_weights_hard_prompts_get_higher_weight():
    t = RunningRewardTracker(alpha=0.3)
    t.update("hard", 0.1)   # very low reward
    t.update("easy", 0.9)   # high reward
    w = build_weights(
        ["hard", "easy"], t, epsilon=0.1, max_weight=10.0, default_mean=0.5,
    )
    # 1/(0.1+0.1) = 5.0;  1/(0.9+0.1) = 1.0
    assert w[0] == pytest.approx(5.0)
    assert w[1] == pytest.approx(1.0)
    assert w[0] > w[1]


def test_build_weights_clamps_at_max():
    t = RunningRewardTracker(alpha=0.3)
    t.update("very_hard", 0.0)
    w = build_weights(
        ["very_hard"], t, epsilon=0.1, max_weight=5.0, default_mean=0.5,
    )
    # 1/(0+0.1) = 10.0, clamped to 5.0
    assert w[0] == pytest.approx(5.0)


def test_build_weights_handles_negative_reward_floored_at_zero():
    """Length penalty can push rewards below 0; treat as 0 for weight calc."""
    t = RunningRewardTracker(alpha=0.3)
    t.update("penalized", -0.5)
    w = build_weights(
        ["penalized"], t, epsilon=0.1, max_weight=5.0, default_mean=0.5,
    )
    # max(0, -0.5) + 0.1 = 0.1 → weight = 10 → clamped to 5
    assert w[0] == pytest.approx(5.0)


# ─── PrioritizedSampler ──────────────────────────────────────────────────────


def test_sampler_yields_exactly_samples_per_iter():
    weights = torch.ones(100)
    sampler = PrioritizedSampler(
        num_records=100, weights_ref=[weights], samples_per_iter=42, seed=0,
    )
    indices = list(iter(sampler))
    assert len(indices) == 42
    assert len(sampler) == 42


def test_sampler_concentrates_on_high_weight_records():
    """If record 0 has weight 1000x the others, it should dominate the samples."""
    weights = torch.ones(10)
    weights[0] = 1000.0
    sampler = PrioritizedSampler(
        num_records=10, weights_ref=[weights], samples_per_iter=1000, seed=0,
    )
    indices = list(iter(sampler))
    n_zero = sum(1 for i in indices if i == 0)
    # Expected fraction: 1000 / (1000 + 9) ≈ 0.99 → >900 of 1000 draws
    assert n_zero > 900, f"expected >900 hits on record 0, got {n_zero}"


def test_sampler_picks_up_weight_swap_between_iters():
    """Mutating weights_ref[0] between __iter__ calls changes the sample stream."""
    weights_a = torch.zeros(10)
    weights_a[0] = 1.0   # all mass on record 0
    weights_ref = [weights_a]
    sampler = PrioritizedSampler(
        num_records=10, weights_ref=weights_ref, samples_per_iter=100, seed=0,
    )
    indices_a = list(iter(sampler))
    assert all(i == 0 for i in indices_a)

    weights_b = torch.zeros(10)
    weights_b[7] = 1.0   # all mass on record 7
    weights_ref[0] = weights_b
    indices_b = list(iter(sampler))
    assert all(i == 7 for i in indices_b)


def test_sampler_rejects_mismatched_weight_length():
    weights = torch.ones(5)
    sampler = PrioritizedSampler(
        num_records=10, weights_ref=[weights], samples_per_iter=10, seed=0,
    )
    with pytest.raises(RuntimeError, match="weights length"):
        list(iter(sampler))


def test_sampler_falls_back_to_uniform_when_all_weights_zero(caplog):
    """A misconfig (eps=0, default_mean=0, all-zero rewards) → all-zero
    weights → multinomial would crash. We should fall back to uniform
    and warn, not bring down the run."""
    import logging
    weights = torch.zeros(10)
    sampler = PrioritizedSampler(
        num_records=10, weights_ref=[weights], samples_per_iter=50, seed=0,
    )
    with caplog.at_level(logging.WARNING, logger="prism.rl.sampler"):
        indices = list(iter(sampler))
    assert len(indices) == 50
    # Uniform over 10 records → roughly 5 of each, but with 50 samples and
    # n=10 we'd expect ~5±2.5. The point is we got *something*, not crashed.
    assert min(indices) >= 0 and max(indices) < 10
    assert any("weight sum" in rec.message for rec in caplog.records)


def test_sampler_seed_determines_output():
    weights = torch.rand(50)
    s0 = PrioritizedSampler(
        num_records=50, weights_ref=[weights], samples_per_iter=20, seed=42,
    )
    s1 = PrioritizedSampler(
        num_records=50, weights_ref=[weights], samples_per_iter=20, seed=42,
    )
    assert list(iter(s0)) == list(iter(s1))


# ─── Resume from judge traces ────────────────────────────────────────────────


def test_from_judge_traces_replays_in_step_order(tmp_path: Path):
    """The tracker rebuilt from traces should reflect the same EMA path as
    a tracker that received those same updates live."""
    trace_path = tmp_path / "judge_traces.jsonl"
    # Two prompts, three "steps" of judge calls each
    records = []
    for step, rewards_a, rewards_b in [
        (10, [0.5, 0.7, 0.6], [0.9, 1.0, 0.8]),
        (20, [0.4, 0.5, 0.4], [0.95, 1.0, 0.9]),
        (30, [0.6, 0.5, 0.6], [0.85, 0.9, 0.95]),
    ]:
        for r in rewards_a:
            records.append({"step": step, "record_id": "A", "reward": r})
        for r in rewards_b:
            records.append({"step": step, "record_id": "B", "reward": r})

    with open(trace_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    t = RunningRewardTracker.from_judge_traces(trace_path, alpha=0.3)
    # Reference: same updates applied live
    ref = RunningRewardTracker(alpha=0.3)
    ref.update("A", (0.5 + 0.7 + 0.6) / 3)
    ref.update("B", (0.9 + 1.0 + 0.8) / 3)
    ref.update("A", (0.4 + 0.5 + 0.4) / 3)
    ref.update("B", (0.95 + 1.0 + 0.9) / 3)
    ref.update("A", (0.6 + 0.5 + 0.6) / 3)
    ref.update("B", (0.85 + 0.9 + 0.95) / 3)
    assert t.values["A"] == pytest.approx(ref.values["A"])
    assert t.values["B"] == pytest.approx(ref.values["B"])


def test_from_judge_traces_skips_all_zero_groups(tmp_path: Path):
    """Judge-error groups (all-zero rewards) should NOT update the tracker."""
    trace_path = tmp_path / "judge_traces.jsonl"
    records = [
        # Real signal
        {"step": 10, "record_id": "good", "reward": 0.7},
        {"step": 10, "record_id": "good", "reward": 0.8},
        # Judge-error step — all forced-zero
        {"step": 20, "record_id": "bad", "reward": 0.0},
        {"step": 20, "record_id": "bad", "reward": 0.0},
    ]
    with open(trace_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    t = RunningRewardTracker.from_judge_traces(trace_path, alpha=0.3)
    assert "good" in t.values
    assert "bad" not in t.values


def test_from_judge_traces_handles_missing_file(tmp_path: Path):
    t = RunningRewardTracker.from_judge_traces(
        tmp_path / "doesnt_exist.jsonl", alpha=0.3,
    )
    assert len(t.values) == 0
