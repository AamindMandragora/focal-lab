"""Convert direct-eval reports into no-gold consensus candidates.

Inputs:
  - completed direct-eval report dictionaries with `sample_outputs`;
  - optional source-family labels supplied by the caller.

Outputs:
  - H31 `Candidate` records that can be passed to the no-gold consensus
    selector.

Algorithm:
  1. read source metadata from the report;
  2. walk `sample_outputs` in stable 1-based order;
  3. keep only extracted, syntax-valid answers with non-empty `actual` text;
  4. normalize the expression text into the equivalence key;
  5. assign a simple no-gold quality score from syntax/failure metadata.

This module must not use gold answer fields. Correctness is measured only after
selection by separate evaluation code.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

from synthesis.evaluate.candidate_consensus import Candidate

_CANDIDATE_LINE_RE = re.compile(
    r"^\s*(?:[*\-]\s*)?(?:\d+\.\s*)?Candidate\s+(?P<label>[A-Za-z]|\d+)\s*:\s*(?P<body>.+?)\s*$",
    re.IGNORECASE,
)
_TRAILING_COMMENT_RE = re.compile(
    r"\s+\((?:same|derived|incorrect|correct|equivalent|missing|wait|this)\b.*$",
    re.IGNORECASE,
)


def candidates_from_direct_eval_reports(
    reports: Iterable[Mapping[str, Any]],
    *,
    include_candidate_lines: bool = False,
) -> list[Candidate]:
    """Build candidates from several completed direct-eval reports."""

    candidates: list[Candidate] = []
    for report in reports:
        source_family = str(report.get("output_name") or report.get("source_id") or "unknown")
        candidates.extend(candidates_from_direct_eval_report(
            report,
            source_family=source_family,
            include_candidate_lines=include_candidate_lines,
        ))
    return candidates


def candidates_from_direct_eval_report(
    report: Mapping[str, Any],
    *,
    source_family: str | None = None,
    include_candidate_lines: bool = False,
) -> list[Candidate]:
    """Build no-gold candidates from one direct-eval report."""

    source = str(report.get("source_id") or report.get("output_name") or "unknown")
    family = str(source_family or report.get("output_name") or source)
    sample_outputs = report.get("sample_outputs") or []

    candidates: list[Candidate] = []
    for index, sample in enumerate(sample_outputs, start=1):
        if not isinstance(sample, Mapping):
            continue
        expression = _usable_expression(sample)
        if expression is None:
            continue
        candidates.append(Candidate(
            group_id=index,
            expression=expression,
            equivalence_key=_normalise_expression(expression),
            source=source,
            source_family=family,
            quality_score=_quality_score(sample),
        ))
        if include_candidate_lines:
            candidates.extend(_candidate_line_candidates(
                sample,
                group_id=index,
                source=source,
                source_family=family,
            ))
    return candidates


def _candidate_line_candidates(
    sample: Mapping[str, Any],
    *,
    group_id: int,
    source: str,
    source_family: str,
) -> list[Candidate]:
    candidates: list[Candidate] = []
    seen: set[tuple[str, str]] = set()
    for label, expression in _candidate_lines(sample):
        key = (label, expression)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(Candidate(
            group_id=group_id,
            expression=expression,
            equivalence_key=_normalise_expression(expression),
            source=f"{source}:candidate_line:{label}",
            source_family=f"{source_family}:candidate_lines",
            quality_score=0.75,
        ))
    return candidates


def _candidate_lines(sample: Mapping[str, Any]) -> list[tuple[str, str]]:
    lines: list[tuple[str, str]] = []
    for field_name in ("full_output", "scored_output"):
        value = sample.get(field_name)
        if not isinstance(value, str):
            continue
        for line in value.splitlines():
            match = _CANDIDATE_LINE_RE.match(line)
            if match is None:
                continue
            expression = _clean_candidate_line_expression(match.group("body"))
            if expression:
                lines.append((match.group("label").upper(), expression))
    return lines


def _clean_candidate_line_expression(expression: str) -> str:
    expression = _normalise_expression(expression)
    dollar_match = re.match(r"^(\$[^$]+\$)", expression)
    if dollar_match is not None:
        return _normalise_expression(dollar_match.group(1))
    expression = _TRAILING_COMMENT_RE.sub("", expression)
    return _normalise_expression(expression)


def _usable_expression(sample: Mapping[str, Any]) -> str | None:
    if not sample.get("has_extracted_answer"):
        return None
    if not sample.get("is_syntax_valid"):
        return None
    actual = sample.get("actual")
    if actual is None:
        return None
    expression = _normalise_expression(str(actual))
    if not expression:
        return None
    return expression


def _normalise_expression(expression: str) -> str:
    return " ".join(expression.strip().split())


def _quality_score(sample: Mapping[str, Any]) -> float:
    failure_location = str(sample.get("failure_location") or "")
    if failure_location == "correct":
        return 1.0
    if failure_location == "syntax_valid_semantic_mismatch":
        return 0.8
    if sample.get("answer_source") == "last_visible_span":
        return 0.7
    return 0.6
