"""CPU-only scoring for recovered GSM candidate lines.

Inputs:
  - GSM split manifest paths that point to CRANE source examples;
  - completed direct-eval reports;
  - candidate-line records from `candidate_report_adapter`.

Outputs:
  - per-candidate correctness rows;
  - union and selected-candidate correctness indices.

Algorithm:
  1. load canonical GSM examples in split order;
  2. parse candidate lines from report text;
  3. normalize display math into CRANE-style Python expressions;
  4. score only after selection using the existing GSM equivalence checker.

Gold fields must never be used to choose a candidate. They are used only by this
post-selection measurement layer.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from synthesis.evaluate.candidate_consensus import Candidate, select_consensus
from synthesis.evaluate.candidate_report_adapter import candidates_from_direct_eval_reports
from synthesis.evaluate.evaluator import (
    _crane_test_expression_equivalence,
    _crane_validate_expression_equivalence,
)
from synthesis.evaluate.benchmarks.gsm_symbolic.dataset import load_gsm_from_crane_folder
from synthesis.project_defaults import default_gsm_source_dir


@dataclass(frozen=True)
class ScoredCandidateLine:
    """One scored candidate-line record."""

    candidate: Candidate
    normalized_expression: str
    is_correct: bool


@dataclass(frozen=True)
class CandidateLineScoreResult:
    """Summary of candidate-line scoring."""

    rows: list[ScoredCandidateLine]
    candidate_line_count: int
    correct_union_indices: list[int]
    selected_correct_indices: list[int]


def load_gsm_split_examples(
    split_file: str | Path,
    *,
    split_name: str = "train",
    crane_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Load canonical GSM examples in the order used by a split manifest."""

    manifest = json.loads(Path(split_file).read_text())
    key = f"{split_name}_indices"
    indices = manifest.get(key)
    if not isinstance(indices, list) or not all(isinstance(index, int) for index in indices):
        raise ValueError(f"{key} must be a list of integer example indices")
    manifest_crane_dir = manifest.get("crane_dir")
    if crane_dir is None and not manifest_crane_dir:
        raise ValueError("split manifest must include crane_dir")

    source_dir = Path(crane_dir or manifest_crane_dir).expanduser()
    if crane_dir is None and not source_dir.exists():
        portable_default = default_gsm_source_dir()
        if portable_default.exists():
            source_dir = portable_default
        else:
            raise FileNotFoundError(
                "CRANE GSM folder not found at the recorded manifest path "
                f"({source_dir}) or the local default ({portable_default}). "
                "Set CRANE_GSM_SYMBOLIC_DIR or pass crane_dir explicitly."
            )
    return load_gsm_from_crane_folder(source_dir, indices=indices)


def score_candidate_lines(
    reports: Sequence[Mapping[str, Any]],
    examples: Sequence[Mapping[str, Any]],
) -> CandidateLineScoreResult:
    """Score parsed candidate-line expressions against aligned GSM examples."""

    all_candidates = candidates_from_direct_eval_reports(reports, include_candidate_lines=True)
    line_candidates = [
        candidate for candidate in all_candidates
        if ":candidate_line:" in candidate.source
    ]
    selected = select_consensus(line_candidates)

    rows: list[ScoredCandidateLine] = []
    correct_union: set[int] = set()
    for candidate in line_candidates:
        group_index = int(candidate.group_id)
        if group_index < 1 or group_index > len(examples):
            continue
        normalized = normalize_gsm_candidate_expression(candidate.expression)
        is_correct = score_gsm_expression(normalized, examples[group_index - 1], already_normalized=True)
        if is_correct:
            correct_union.add(group_index)
        rows.append(ScoredCandidateLine(
            candidate=candidate,
            normalized_expression=normalized,
            is_correct=is_correct,
        ))

    selected_correct: list[int] = []
    for group_id, selection in selected.items():
        group_index = int(group_id)
        if group_index < 1 or group_index > len(examples):
            continue
        if score_gsm_expression(selection.candidate.expression, examples[group_index - 1]):
            selected_correct.append(group_index)

    return CandidateLineScoreResult(
        rows=rows,
        candidate_line_count=len(line_candidates),
        correct_union_indices=sorted(correct_union),
        selected_correct_indices=sorted(selected_correct),
    )


def score_gsm_expression(
    expression: str,
    example: Mapping[str, Any],
    *,
    already_normalized: bool = False,
) -> bool:
    """Return whether an expression matches a canonical GSM example answer."""

    expected = str(example.get("answer_parsed") or example.get("expected") or "").strip()
    variable_types = example.get("variable_types") or {}
    if not expected or not isinstance(variable_types, Mapping):
        return False
    normalized = expression if already_normalized else normalize_gsm_candidate_expression(expression)
    if not normalized:
        return False
    try:
        return bool(_crane_validate_expression_equivalence(
            normalized,
            expected,
            dict(variable_types),
        ))
    except Exception:
        return bool(_crane_test_expression_equivalence(
            normalized,
            expected,
            list(variable_types),
            dict(variable_types),
        ))


def normalize_gsm_candidate_expression(expression: str) -> str:
    """Normalize common report candidate syntax into CRANE-style Python math."""

    text = str(expression or "").strip()
    text = text.strip("$").strip()
    text = text.replace("\\times", "*")
    text = text.replace("\\cdot", "*")
    text = text.replace("^", "**")
    text = re.sub(r"\\text\{([^{}]+)\}", r"\1", text)
    text = _replace_latex_fractions(text)
    text = re.sub(r"(?P<num>\d)\s*\{(?P<var>[A-Za-z_][A-Za-z0-9_]*)\}", r"\g<num>*\g<var>", text)
    text = re.sub(r"\{(?P<var>[A-Za-z_][A-Za-z0-9_]*)\}", r"\g<var>", text)
    text = re.sub(r"(?P<num>\d)(?P<var>[A-Za-z_][A-Za-z0-9_]*)", r"\g<num>*\g<var>", text)
    text = text.replace("%", "")
    return " ".join(text.split())


def _replace_latex_fractions(text: str) -> str:
    pattern = re.compile(r"\\frac\{([^{}]+)\}\{([^{}]+)\}")
    while True:
        updated = pattern.sub(r"((\1)/(\2))", text)
        if updated == text:
            return text
        text = updated
