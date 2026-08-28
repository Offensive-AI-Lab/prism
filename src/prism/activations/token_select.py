"""Token selection strategies for slicing response activations."""

from __future__ import annotations

import random as _random
from typing import List, Tuple


def select_token_range(
    response_length: int,
    num_tokens: int,
    strategy: str,
    rng: _random.Random | None = None,
) -> Tuple[int, int]:
    """Return (start, end) indices relative to the response start.

    The returned range is a half-open interval [start, end) within the
    response portion of the sequence.  ``end - start <= num_tokens``.

    Args:
        response_length: Total number of response tokens available.
        num_tokens: Maximum number of tokens to select.
        strategy: One of ``"last"``, ``"first"``, ``"middle"``, ``"random"``.
        rng: Random instance used when ``strategy="random"``.

    Returns:
        (start, end) relative to the first response token.
    """
    if response_length <= 0:
        return (0, 0)

    n = min(num_tokens, response_length)

    if strategy == "last":
        return (response_length - n, response_length)

    if strategy == "first":
        return (0, n)

    if strategy == "middle":
        center = response_length // 2
        start = max(center - n // 2, 0)
        start = min(start, response_length - n)
        return (start, start + n)

    if strategy == "random":
        if rng is None:
            rng = _random.Random()
        max_start = response_length - n
        start = rng.randint(0, max_start) if max_start > 0 else 0
        return (start, start + n)

    raise ValueError(f"Unknown token selection strategy: {strategy!r}")


def select_positions(
    response_length: int,
    num_tokens: int,
    strategy: str,
    rng: _random.Random | None = None,
) -> List[int]:
    """Return a sorted list of token positions (relative to response start).

    This is a convenience wrapper around :func:`select_token_range` that
    returns an explicit list of indices.  For contiguous strategies (last,
    first, middle) this is equivalent to ``list(range(start, end))``.
    """
    start, end = select_token_range(response_length, num_tokens, strategy, rng)
    return list(range(start, end))
