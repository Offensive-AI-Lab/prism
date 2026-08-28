"""Reward length-penalty math — pinned to the released behavior."""

import pytest

from prism.rl.judge import _length_penalty


def _lp(n_report, n_gt, **kw):
    defaults = dict(enabled=True, k=1.5, lam=0.15)
    defaults.update(kw)
    return _length_penalty(n_report, n_gt, **defaults)


def test_no_penalty_within_budget():
    # report ≤ k·gt → no overrun
    assert _lp(6, 4) == 0.0
    assert _lp(7, 5) == 0.0


def test_overrun_penalty_is_linear():
    # overrun = n_report − 1.5·n_gt
    assert _lp(9, 4) == pytest.approx(0.15 * (9 - 6))
    assert _lp(10, 4) == pytest.approx(0.15 * (10 - 6))


def test_disabled_means_zero():
    assert _lp(20, 2, enabled=False) == 0.0


def test_zero_gt_guard():
    # No GT bullets → no penalty regardless of report length (the released
    # judge-error path relies on this returning 0 only for n_gt == 0).
    assert _lp(10, 0) == 0.0
    assert _lp(0, 0) == 0.0


def test_under_length_collapse_penalty():
    # shortfall = under_k·n_gt − n_report, gated by under_enabled
    val = _lp(1, 5, under_enabled=True, under_k=0.5, under_lam=0.15)
    assert val == pytest.approx(0.15 * (0.5 * 5 - 1))
    # exactly half the GT count → no shortfall
    assert _lp(3, 6, under_enabled=True, under_k=0.5, under_lam=0.15) == 0.0
    # under-penalty off (the released qwen run's behavior)
    assert _lp(1, 5, under_enabled=False) == 0.0


def test_both_sides_can_stack():
    # (degenerate config, but the terms must gate independently)
    val = _length_penalty(
        0, 4, enabled=True, k=1.5, lam=0.15,
        under_enabled=True, under_k=0.5, under_lam=0.15,
    )
    assert val == pytest.approx(0.15 * (0.5 * 4 - 0))
