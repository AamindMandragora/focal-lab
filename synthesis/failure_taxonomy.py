"""Cluster-organized failure taxonomy for synthesis-loop feedback.

Replaces five flat aggregate blocks (Primary Failure Modes, Diagnostic
Error Decomposition, Output Provenance, Correct-vs-Wrong Contrast,
Structural Generation Metrics) with a single cluster-organized block:

  1. Each wrong sample is converted to a 17-axis fingerprint.
  2. Fingerprints are clustered greedily with Hamming distance ≤ 1.
  3. Each cluster is rendered with its size, axis profile, and a meta-stat
     summary across clusters (which axes vary across clusters vs which
     are global to every cluster).
  4. Cross-attempt persistence: clusters get stable IDs across attempts so
     the author can see "mode_A appeared in attempts 1, 3, 5, 7, 9".
"""

from __future__ import annotations

from collections import Counter
from statistics import median
from typing import Any, Dict, Iterable, List, Optional, Tuple


# ---------------------------------------------------------------------------
# 17-axis fingerprint definition
# ---------------------------------------------------------------------------
#
# Each axis returns a small-cardinality value (bool or one of ≤3 categorical
# tokens). The fingerprint is a tuple in this exact axis order so that two
# tuples can be compared element-wise for Hamming distance.

FINGERPRINT_AXES: Tuple[str, ...] = (
    "syntax_valid",          # bool
    "contains_delimiters",   # bool
    "visible_delimiters",    # bool
    "used_constrained",      # bool: passed through the CSD path at all
    "answer_source",         # {constrained, text_fallback, none}
    "has_extracted_answer",  # bool
    "hit_max_steps",         # bool
    "runtime_exceeded",      # bool
    "error_present",         # bool
    "zero_visible_spans",    # bool: num_visible_spans == 0
    "zero_valid_spans",      # bool: num_valid_visible_spans == 0
    "partial_valid_spans",   # bool: 0 < valid < visible
    "tiny_span_dominant",    # bool: median visible span tok len <= 2
    "long_span_dominant",    # bool: median visible span tok len >= 20
    "token_budget_band",     # {low, mid, high}
    "gen_time_band",         # {fast, mid, slow}
    "accuracy_applicable",   # bool
)


def _normalize_answer_source(value: Any) -> str:
    if value is None:
        return "none"
    s = str(value).lower()
    if "constrain" in s:
        return "constrained"
    if "fallback" in s or "text" in s:
        return "text_fallback"
    return "none"


def _band_token_budget(token_count: int, max_steps: int) -> str:
    if max_steps <= 0:
        return "mid"
    ratio = token_count / max_steps
    if ratio < 0.25:
        return "low"
    if ratio > 0.75:
        return "high"
    return "mid"


def _band_gen_time(time_seconds: float, slow_threshold: float) -> str:
    if slow_threshold <= 0:
        return "mid"
    if time_seconds < 0.2 * slow_threshold:
        return "fast"
    if time_seconds > 0.8 * slow_threshold:
        return "slow"
    return "mid"


def fingerprint(
    sample: Dict[str, Any],
    *,
    max_steps: int = 512,
    slow_threshold_seconds: float = 30.0,
) -> Tuple[Any, ...]:
    """Reduce a per-sample dict to a 17-axis fingerprint tuple."""
    visible_lengths = sample.get("visible_span_token_lengths") or []
    median_visible = int(median(visible_lengths)) if visible_lengths else 0
    num_visible = sample.get("num_visible_spans", 0) or 0
    num_valid = sample.get("num_valid_visible_spans", 0) or 0
    return (
        bool(sample.get("is_syntax_valid", False)),
        bool(sample.get("contains_delimiters", False)),
        bool(sample.get("visible_delimiters", False)),
        bool(sample.get("used_constrained_chunk", False)),
        _normalize_answer_source(sample.get("answer_source")),
        bool(sample.get("has_extracted_answer", False)),
        bool(sample.get("hit_max_steps", False)),
        bool(sample.get("runtime_budget_exceeded", False)),
        bool(sample.get("error")),
        num_visible == 0,
        num_valid == 0,
        (0 < num_valid < num_visible),
        bool(visible_lengths) and median_visible <= 2,
        bool(visible_lengths) and median_visible >= 20,
        _band_token_budget(sample.get("token_count", 0) or 0, max_steps),
        _band_gen_time(sample.get("time_seconds", 0.0) or 0.0, slow_threshold_seconds),
        bool(sample.get("accuracy_applicable", True)),
    )


def _hamming(a: Tuple[Any, ...], b: Tuple[Any, ...]) -> int:
    return sum(1 for x, y in zip(a, b) if x != y)


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

Cluster = Dict[str, Any]  # {"medoid": tuple, "members": List[Tuple]}


def cluster(
    fingerprints: List[Tuple[Any, ...]],
    *,
    max_hamming: int = 1,
) -> List[Cluster]:
    """Greedy single-pass clustering with Hamming distance ≤ max_hamming.

    Each fingerprint joins the first existing cluster whose medoid is within
    max_hamming; otherwise it starts a new cluster. Stable and order-preserving.
    """
    clusters: List[Cluster] = []
    for fp in fingerprints:
        joined = False
        for c in clusters:
            if _hamming(fp, c["medoid"]) <= max_hamming:
                c["members"].append(fp)
                joined = True
                break
        if not joined:
            clusters.append({"medoid": fp, "members": [fp]})
    # Sort by size, largest first
    clusters.sort(key=lambda c: -len(c["members"]))
    return clusters


def cluster_axis_profile(c: Cluster) -> Dict[str, Any]:
    """Per-axis: dominant value + share within cluster."""
    profile: Dict[str, Any] = {}
    members = c["members"]
    n = len(members)
    for idx, axis in enumerate(FINGERPRINT_AXES):
        values = [m[idx] for m in members]
        counter = Counter(values)
        top_val, top_count = counter.most_common(1)[0]
        profile[axis] = (top_val, top_count / n)
    return profile


def cluster_meta_stats(clusters: List[Cluster]) -> Dict[str, Any]:
    """Marginal counts across clusters: which axes vary, which are constant."""
    n_clusters = len(clusters)
    varying: List[str] = []
    constant: Dict[str, Any] = {}
    for idx, axis in enumerate(FINGERPRINT_AXES):
        distinct = {c["medoid"][idx] for c in clusters}
        if len(distinct) == 1 and n_clusters > 1:
            constant[axis] = next(iter(distinct))
        elif len(distinct) > 1:
            varying.append(axis)
    return {
        "n_clusters": n_clusters,
        "n_wrong_samples": sum(len(c["members"]) for c in clusters),
        "axes_varying_across_clusters": varying,
        "axes_constant_across_clusters": constant,
    }


# ---------------------------------------------------------------------------
# Cross-attempt persistence
# ---------------------------------------------------------------------------

PersistentLedger = Dict[str, Any]
# Schema: {
#   "next_id": int,
#   "modes": [
#       {"id": "mode_A", "medoid": tuple, "attempts": [int, ...]}
#   ]
# }


def make_persistent_ledger() -> PersistentLedger:
    return {"next_id": 0, "modes": []}


def _mode_label(idx: int) -> str:
    # mode_A, mode_B, ..., mode_Z, mode_AA, mode_AB, ...
    label = ""
    n = idx
    while True:
        label = chr(ord("A") + (n % 26)) + label
        n = n // 26 - 1
        if n < 0:
            break
    return f"mode_{label}"


def assign_persistent_cluster_id(
    clusters: List[Cluster],
    ledger: PersistentLedger,
    attempt_index: int,
    *,
    max_hamming: int = 1,
) -> List[Tuple[str, Cluster]]:
    """For each cluster, find a matching prior mode in the ledger or create one.

    Returns [(mode_id, cluster), ...] in the same order as input clusters.
    Ledger is mutated in place.
    """
    assigned: List[Tuple[str, Cluster]] = []
    used_ids: set = set()
    for c in clusters:
        match_id: Optional[str] = None
        best_dist = max_hamming + 1
        for mode in ledger["modes"]:
            if mode["id"] in used_ids:
                continue
            d = _hamming(c["medoid"], mode["medoid"])
            if d <= max_hamming and d < best_dist:
                best_dist = d
                match_id = mode["id"]
        if match_id is None:
            new_id = _mode_label(ledger["next_id"])
            ledger["next_id"] += 1
            ledger["modes"].append(
                {"id": new_id, "medoid": c["medoid"], "attempts": [attempt_index]}
            )
            match_id = new_id
        else:
            for mode in ledger["modes"]:
                if mode["id"] == match_id and attempt_index not in mode["attempts"]:
                    mode["attempts"].append(attempt_index)
                    break
        used_ids.add(match_id)
        assigned.append((match_id, c))
    return assigned


def render_persistence_summary(ledger: PersistentLedger) -> str:
    if not ledger["modes"]:
        return ""
    lines = ["Cross-attempt mode persistence:"]
    for mode in ledger["modes"]:
        runs = ",".join(str(a) for a in sorted(set(mode["attempts"])))
        lines.append(f"  - {mode['id']}: appeared in attempt(s) {runs}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _format_axis_value(axis: str, val: Any) -> str:
    if isinstance(val, bool):
        return "yes" if val else "no"
    return str(val)


def render_cluster_block(
    samples: Iterable[Dict[str, Any]],
    *,
    max_steps: int = 512,
    slow_threshold_seconds: float = 30.0,
    max_clusters_shown: int = 6,
    persistent_ledger: Optional[PersistentLedger] = None,
    attempt_index: Optional[int] = None,
) -> str:
    """Build the single cluster-organized failure block.

    Pass `persistent_ledger` + `attempt_index` to enable Change 2 (cross-attempt
    cluster IDs and the persistence summary footer).
    """
    wrong = [
        s for s in samples
        if not s.get("is_correct") and s.get("accuracy_applicable", True)
    ]
    if not wrong:
        return ""

    fps = [
        fingerprint(s, max_steps=max_steps, slow_threshold_seconds=slow_threshold_seconds)
        for s in wrong
    ]
    clusters = cluster(fps, max_hamming=1)

    if persistent_ledger is not None and attempt_index is not None:
        labeled = assign_persistent_cluster_id(
            clusters, persistent_ledger, attempt_index
        )
    else:
        labeled = [(f"cluster_{i+1}", c) for i, c in enumerate(clusters)]

    meta = cluster_meta_stats(clusters)
    lines: List[str] = []
    lines.append("\nFailure Clusters (17-axis fingerprint, Hamming ≤ 1):")
    lines.append(
        f"  Wrong samples: {meta['n_wrong_samples']} grouped into "
        f"{meta['n_clusters']} cluster(s)."
    )
    if meta["axes_constant_across_clusters"]:
        constant_summary = ", ".join(
            f"{axis}={_format_axis_value(axis, v)}"
            for axis, v in meta["axes_constant_across_clusters"].items()
        )
        lines.append(f"  Constant across all clusters: {constant_summary}")
    if meta["axes_varying_across_clusters"]:
        lines.append(
            "  Varying across clusters: "
            + ", ".join(meta["axes_varying_across_clusters"])
        )

    for mode_id, c in labeled[:max_clusters_shown]:
        profile = cluster_axis_profile(c)
        size = len(c["members"])
        lines.append(f"\n  [{mode_id}] {size} sample(s):")
        # Show only axes whose dominant value is "non-trivial" (not the
        # global constant); reduces noise in per-cluster profiles.
        constants = meta["axes_constant_across_clusters"]
        salient = [
            (axis, val, share)
            for axis, (val, share) in profile.items()
            if axis not in constants
        ]
        if not salient:
            # Fallback: print top axes anyway so the cluster has a description.
            salient = [(axis, val, share) for axis, (val, share) in profile.items()][:6]
        for axis, val, share in salient:
            lines.append(
                f"    {axis}: {_format_axis_value(axis, val)} ({share:.0%})"
            )

    if len(labeled) > max_clusters_shown:
        lines.append(
            f"\n  …and {len(labeled) - max_clusters_shown} smaller cluster(s)."
        )

    if persistent_ledger is not None and attempt_index is not None:
        persistence = render_persistence_summary(persistent_ledger)
        if persistence:
            lines.append("")
            lines.append(persistence)

    return "\n".join(lines)
