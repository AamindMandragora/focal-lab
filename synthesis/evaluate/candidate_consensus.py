"""No-gold candidate consensus selection.

Inputs:
  - candidate answer records, grouped by caller-supplied problem id;
  - a caller-supplied equivalence key for each candidate;
  - source/source-family metadata;
  - a no-gold quality score.

Outputs:
  - one selected candidate per problem group;
  - audit fields explaining the selected cluster and tie-breaks.

Algorithm:
  1. group candidates by problem id;
  2. cluster each group by equivalence key;
  3. prefer clusters with more candidate support, then more source-family
     support, then more exact-source support, then better candidate quality;
  4. inside the winning cluster, choose the highest-quality candidate.

The selector never accepts expected answers or correctness labels. Evaluation
code may compare its output against gold later, but gold is not part of this
selection API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Iterable


@dataclass(frozen=True)
class Candidate:
    """One candidate answer and its no-gold selection metadata."""

    group_id: Hashable
    expression: str
    equivalence_key: Hashable
    source: str
    source_family: str
    quality_score: float


@dataclass(frozen=True)
class ConsensusSelection:
    """Audit-friendly result for one problem group."""

    candidate: Candidate
    cluster_key: Hashable
    agreement_score: tuple[int, int, int, float]
    candidate_count: int
    source_count: int
    family_count: int
    best_quality_score: float


def select_consensus(candidates: Iterable[Candidate]) -> dict[Hashable, ConsensusSelection]:
    """Select one no-gold consensus candidate per group.

    Inputs:
      - candidates: iterable of `Candidate` records. Each record must already
        contain any benchmark-specific normalization in `equivalence_key`.

    Outputs:
      - `{group_id: ConsensusSelection}`. Groups with no candidates are absent.

    Algorithm:
      1. bucket records by group id;
      2. bucket each group's records by equivalence key;
      3. score clusters by support tuple;
      4. choose the best candidate from the best cluster by quality score.
    """

    grouped: dict[Hashable, list[Candidate]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate.group_id, []).append(candidate)

    return {
        group_id: _select_for_group(group_candidates)
        for group_id, group_candidates in grouped.items()
    }


def _select_for_group(candidates: list[Candidate]) -> ConsensusSelection:
    clusters: dict[Hashable, list[Candidate]] = {}
    for candidate in candidates:
        clusters.setdefault(candidate.equivalence_key, []).append(candidate)

    cluster_key, cluster_candidates = max(
        clusters.items(),
        key=lambda item: _cluster_score(item[1]),
    )
    chosen = max(cluster_candidates, key=_candidate_score)
    sources = {candidate.source for candidate in cluster_candidates}
    families = {candidate.source_family for candidate in cluster_candidates}
    best_quality = float(chosen.quality_score)

    return ConsensusSelection(
        candidate=chosen,
        cluster_key=cluster_key,
        agreement_score=_cluster_score(cluster_candidates),
        candidate_count=len(cluster_candidates),
        source_count=len(sources),
        family_count=len(families),
        best_quality_score=best_quality,
    )


def _cluster_score(candidates: list[Candidate]) -> tuple[int, int, int, float]:
    sources = {candidate.source for candidate in candidates}
    families = {candidate.source_family for candidate in candidates}
    best_quality = max(float(candidate.quality_score) for candidate in candidates)
    return (
        len(candidates),
        len(families),
        len(sources),
        best_quality,
    )


def _candidate_score(candidate: Candidate) -> tuple[float, str, str]:
    return (
        float(candidate.quality_score),
        str(candidate.source_family),
        str(candidate.source),
    )
