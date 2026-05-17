"""Shared counter formatting for evaluation feedback and synthesis memory."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping


def format_named_counter(
    counter: Counter[str] | Mapping[str, int],
    denominator: int,
    *,
    max_items: int = 5,
    min_denominator: int = 1,
) -> str:
    """Format ``key count/denominator`` pairs from a counter."""
    if not counter:
        return "none"
    denom = max(min_denominator, denominator)
    ranked = (
        counter.most_common(max_items)
        if isinstance(counter, Counter)
        else sorted(counter.items(), key=lambda item: (-item[1], item[0]))[:max_items]
    )
    return ", ".join(f"{key} {count}/{denom}" for key, count in ranked)


def format_counter_delta(
    current_counter: Mapping[str, int],
    baseline_counter: Mapping[str, int],
    *,
    max_items: int = 6,
) -> str:
    """Format signed count deltas between two counters."""
    keys = set(current_counter) | set(baseline_counter)
    if not keys:
        return "none"
    ranked = sorted(
        keys,
        key=lambda key: (-abs(current_counter.get(key, 0) - baseline_counter.get(key, 0)), key),
    )
    return ", ".join(
        f"{key} {current_counter.get(key, 0) - baseline_counter.get(key, 0):+d}"
        for key in ranked[:max_items]
    )
