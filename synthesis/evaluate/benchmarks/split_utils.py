"""Shared helpers for proportional benchmark train/eval splits."""

from __future__ import annotations

import random
from collections import Counter
from typing import Any, Mapping, Sequence


def proportional_allocations(
    total: int,
    population: Mapping[str, int],
    *,
    order: Sequence[str],
) -> dict[str, int]:
    """Allocate ``total`` slots across buckets in proportion to ``population`` counts."""
    if total < 0:
        raise ValueError("total must be non-negative")
    if total == 0:
        return {key: 0 for key in order}

    pop_total = sum(int(population.get(key, 0)) for key in order)
    if pop_total <= 0:
        raise ValueError("population must contain at least one example")

    raw = {key: total * int(population.get(key, 0)) / pop_total for key in order}
    counts = {key: int(raw[key]) for key in order}
    assigned = sum(counts.values())
    if assigned > total:
        raise RuntimeError("proportional allocation assigned more than total")

    fractional = sorted(
        ((raw[key] - counts[key], key) for key in order),
        reverse=True,
    )
    idx = 0
    while assigned < total:
        _, key = fractional[idx % len(order)]
        counts[key] += 1
        assigned += 1
        idx += 1
    return counts


def stratified_sample_indices(
    labels_by_index: Mapping[int, str],
    *,
    split_sizes: Mapping[str, int],
    difficulties: Sequence[str],
    seed: int,
) -> dict[str, list[int]]:
    """Sample disjoint index lists with per-split proportional difficulty composition."""
    if not split_sizes:
        raise ValueError("split_sizes must name at least one split")
    for name, size in split_sizes.items():
        if size < 0:
            raise ValueError(f"split_sizes[{name}] must be non-negative")

    by_difficulty: dict[str, list[int]] = {difficulty: [] for difficulty in difficulties}
    for idx, label in labels_by_index.items():
        if label not in by_difficulty:
            raise ValueError(f"Unknown difficulty label {label!r} for index {idx}")
        by_difficulty[label].append(idx)

    population = {difficulty: len(by_difficulty[difficulty]) for difficulty in difficulties}
    allocations = {
        split_name: proportional_allocations(size, population, order=difficulties)
        for split_name, size in split_sizes.items()
    }

    needed_by_difficulty = {difficulty: 0 for difficulty in difficulties}
    for split_alloc in allocations.values():
        for difficulty in difficulties:
            needed_by_difficulty[difficulty] += split_alloc[difficulty]

    for difficulty in difficulties:
        available = len(by_difficulty[difficulty])
        needed = needed_by_difficulty[difficulty]
        if available < needed:
            raise ValueError(
                f"Not enough {difficulty} examples for stratified split: "
                f"need {needed}, have {available}"
            )

    rng = random.Random(seed)
    pools = {
        difficulty: rng.sample(by_difficulty[difficulty], len(by_difficulty[difficulty]))
        for difficulty in difficulties
    }
    offsets = {difficulty: 0 for difficulty in difficulties}

    selected: dict[str, list[int]] = {split_name: [] for split_name in split_sizes}
    for split_name, split_alloc in allocations.items():
        for difficulty in difficulties:
            start = offsets[difficulty]
            count = split_alloc[difficulty]
            selected[split_name].extend(pools[difficulty][start: start + count])
            offsets[difficulty] = start + count

    return {name: sorted(indices) for name, indices in selected.items()}


def composition(
    indices: Sequence[int],
    labels_by_index: Mapping[int, str],
) -> dict[str, int]:
    return dict(sorted(Counter(labels_by_index[idx] for idx in indices).items()))


def population_composition(labels_by_index: Mapping[int, str]) -> dict[str, int]:
    return composition(list(labels_by_index), labels_by_index)


def split_manifest_metadata(
    *,
    seed: int,
    split_strategy: str,
    labels_by_index: Mapping[int, str],
    split_sizes: Mapping[str, int],
    selected: Mapping[str, list[int]],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "seed": seed,
        "split_strategy": split_strategy,
        "total_examples": len(labels_by_index),
        "population_composition": population_composition(labels_by_index),
        **{f"{name}_size": len(selected[name]) for name in selected},
        **{f"{name}_composition": composition(selected[name], labels_by_index) for name in selected},
        **{f"{name}_indices": selected[name] for name in selected},
    }
    if extra:
        manifest.update(extra)
    return manifest
