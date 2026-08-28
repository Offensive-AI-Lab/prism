"""Pure-function tests for activation token selection."""

import random

import pytest

from prism.activations.token_select import select_positions, select_token_range


def test_last_takes_tail():
    assert select_token_range(200, 128, "last") == (72, 200)


def test_first_takes_head():
    assert select_token_range(200, 128, "first") == (0, 128)


def test_short_response_returns_everything():
    for strategy in ("last", "first", "middle"):
        assert select_token_range(50, 128, strategy) == (0, 50)


def test_empty_response():
    assert select_token_range(0, 128, "last") == (0, 0)
    assert select_positions(0, 128, "last") == []


def test_middle_is_centered_and_bounded():
    start, end = select_token_range(100, 10, "middle")
    assert end - start == 10
    assert 0 <= start and end <= 100
    assert abs((start + end) / 2 - 50) <= 1


def test_random_is_seeded_and_in_bounds():
    a = select_token_range(100, 10, "random", rng=random.Random(0))
    b = select_token_range(100, 10, "random", rng=random.Random(0))
    assert a == b
    start, end = a
    assert end - start == 10 and 0 <= start and end <= 100


def test_unknown_strategy_raises():
    with pytest.raises(ValueError):
        select_token_range(10, 5, "nope")


def test_positions_match_range():
    assert select_positions(10, 4, "last") == [6, 7, 8, 9]
