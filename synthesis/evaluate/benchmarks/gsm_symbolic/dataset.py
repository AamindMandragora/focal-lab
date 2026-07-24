"""
GSM-Symbolic dataset loading utilities.

Handles loading the apple/GSM-Symbolic dataset from HuggingFace with various
configurations (main, p1, p2) and supports limiting/sampling for efficient evaluation.
Also backfills CRANE metadata such as variable_types when the local CRANE checkout
is available.
"""

from __future__ import annotations

import json
import os
import random
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional, Sequence

from synthesis.evaluate.benchmarks.split_utils import (
    proportional_allocations,
    split_manifest_metadata,
    stratified_sample_indices,
)
from synthesis.project_defaults import default_gsm_source_dir

DEFAULT_CRANE_GSM_DIR = Path(
    default_gsm_source_dir()
)

GSM_DIFFICULTIES: tuple[str, ...] = ("easy", "medium", "hard")
GSM_CONFIG_DIFFICULTY: dict[str, str] = {
    "main": "easy",
    "p1": "medium",
    "p2": "hard",
}


def _normalize_gsm_text(text: str) -> str:
    """Normalize question text for robust lookups across dataset variants."""
    return " ".join(str(text).split())


@lru_cache(maxsize=None)
def _load_crane_gsm_metadata(crane_dir_str: str) -> dict[str, dict[Any, dict[str, Any]]]:
    """Load CRANE GSM metadata indexed by original id and original question text."""
    crane_dir = Path(crane_dir_str)
    by_original_id: dict[int, dict[str, Any]] = {}
    by_question: dict[str, dict[str, Any]] = {}

    if not crane_dir.exists():
        return {"by_original_id": by_original_id, "by_question": by_question}

    for path in sorted(crane_dir.glob('*.json')):
        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue

        metadata = {
            'variable_types': payload.get('variable_types') or {},
            'question_parsed': payload.get('question_parsed') or payload.get('question', ''),
            'question_annotated': payload.get('question_annotated') or '',
            'answer_parsed': payload.get('answer_parsed') or '',
            'crane_source_file': str(path),
            'crane_id_orig': payload.get('id_orig'),
            'crane_id_shuffled': payload.get('id_shuffled'),
        }

        original_id = payload.get('id_orig')
        if isinstance(original_id, int):
            by_original_id[original_id] = metadata

        question_text = payload.get('question')
        if question_text:
            by_question[_normalize_gsm_text(question_text)] = metadata

    return {"by_original_id": by_original_id, "by_question": by_question}


def enrich_gsm_symbolic_example(
    example: dict[str, Any],
    crane_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Attach CRANE metadata to one HF GSM-Symbolic example when available."""
    crane_dir = Path(crane_dir) if crane_dir is not None else DEFAULT_CRANE_GSM_DIR
    indices = _load_crane_gsm_metadata(str(crane_dir))

    enriched = dict(example)
    metadata = None

    original_id = enriched.get('original_id')
    if original_id is not None:
        try:
            metadata = indices['by_original_id'].get(int(original_id))
        except (TypeError, ValueError):
            metadata = None

    if metadata is None:
        original_question = enriched.get('original_question') or enriched.get('question')
        if original_question:
            metadata = indices['by_question'].get(_normalize_gsm_text(original_question))

    if metadata is None:
        return enriched

    for key, value in metadata.items():
        enriched.setdefault(key, value)

    return enriched


def load_gsm_from_crane_folder(
    crane_dir: Path | str | None = None,
    limit: Optional[int] = None,
    indices: Optional[Sequence[int]] = None,
) -> list[dict[str, Any]]:
    """
    Load GSM-Symbolic examples directly from CRANE's local JSON files.

    Args:
        crane_dir: Path to folder of JSON files (defaults to DEFAULT_CRANE_GSM_DIR).
        limit: Optional limit on number of examples.
        indices: Optional sorted-folder indices to select before applying limit.

    Returns:
        List of example dicts with the same fields the evaluator expects:
        ``question`` is the symbolic template (``question_parsed``) when present,
        with instantiated prose available as ``question_instantiated``.
    """
    crane_dir = Path(crane_dir) if crane_dir is not None else DEFAULT_CRANE_GSM_DIR
    if not crane_dir.exists():
        raise FileNotFoundError(f"CRANE GSM folder not found: {crane_dir}")

    paths = sorted(crane_dir.glob('*.json'))
    if indices is not None:
        selected_paths = []
        for idx in indices:
            if idx < 0 or idx >= len(paths):
                raise IndexError(
                    f"GSM CRANE index {idx} is out of range for {len(paths)} files in {crane_dir}"
                )
            selected_paths.append(paths[idx])
        paths = selected_paths

    rows: list[dict[str, Any]] = []
    for original_index, path in enumerate(paths):
        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue
        # Prefer symbolic placeholders `{var}` (question_parsed) over instantiated prose so GSM-Symbolic
        # prompts, scoring rows, and CRANE subprocess alignment use one representation.
        symbolic_q = payload.get('question_parsed') or ''
        instantiated_q = payload.get('question', '')
        rows.append({
            'question': symbolic_q or instantiated_q,
            'question_instantiated': instantiated_q,
            'answer': payload.get('answer', ''),
            'variable_types': payload.get('variable_types') or {},
            'question_parsed': symbolic_q or '',
            'question_annotated': payload.get('question_annotated') or '',
            'answer_parsed': payload.get('answer_parsed') or '',
            'id_orig': payload.get('id_orig'),
            'id_shuffled': payload.get('id_shuffled'),
            'crane_source_file': str(path),
            'crane_source_index': (
                indices[original_index] if indices is not None else original_index
            ),
        })

    if limit is not None and limit > 0:
        rows = rows[:limit]

    print(f'Loaded {len(rows)} examples from CRANE folder {crane_dir}')
    return rows


def _count_arithmetic_ops(text: str) -> int:
    """Count arithmetic operators inside GSM answer computations."""
    spans = re.findall(r"<<([^>]+)>>", text or "")
    if not spans:
        spans = [text or ""]
    return sum(len(re.findall(r"(?<![A-Za-z])[+\-*/](?![A-Za-z])", span)) for span in spans)


def _gsm_complexity_score(payload: dict[str, Any]) -> float:
    """Heuristic difficulty proxy used when CRANE examples have no labels."""
    question = str(payload.get("question") or "")
    answer = str(payload.get("answer_parsed") or payload.get("answer") or "")
    variable_types = payload.get("variable_types") or {}
    n_vars = len(variable_types) if isinstance(variable_types, dict) else 0
    n_words = len(question.split())
    n_steps = answer.count("<<")
    n_ops = _count_arithmetic_ops(answer)
    expression_chars = sum(len(span) for span in re.findall(r"<<([^>]+)>>", answer))
    has_fraction = bool(re.search(r"\d+\s*/\s*\d+|half|quarter|third", question, re.I))
    has_percent = "%" in question or "percent" in question.lower()
    has_decimal = bool(re.search(r"\d+\.\d+", question + " " + answer))
    return (
        n_ops * 3.0
        + n_steps * 2.0
        + n_vars * 1.5
        + n_words / 8.0
        + expression_chars / 20.0
        + (2.0 if has_fraction else 0.0)
        + (1.5 if has_percent else 0.0)
        + (1.0 if has_decimal else 0.0)
    )


def _count_expression_ops(text: str) -> int:
    """Count arithmetic operators in a parsed expression without natural-language noise."""
    return len(re.findall(r"(?<![A-Za-z])[+\-*/](?![A-Za-z])", text or ""))


def _gsm_rubric_profile(payload: dict[str, Any]) -> dict[str, Any]:
    """Compute an auditable difficulty rubric for one CRANE GSM question."""
    question = str(payload.get("question") or "")
    answer = str(payload.get("answer") or "")
    answer_parsed = str(payload.get("answer_parsed") or "")
    variable_types = payload.get("variable_types") or {}
    variable_count = len(variable_types) if isinstance(variable_types, dict) else 0
    question_word_count = len(question.split())
    answer_step_count = answer.count("<<")
    expression_op_count = _count_expression_ops(answer_parsed)
    text = f"{question} {answer_parsed}".lower()
    flags = {
        "probability": any(
            word in text
            for word in ("probability", "likely", "chance", "odds", "die", "dice")
        ),
        "percent": "%" in text or "percent" in text,
        "multi_percent": len(re.findall(r"%|percent", text)) >= 2,
        "average": "average" in text or "mean" in text,
        "unit_conversion": bool(
            re.search(
                r"inch|foot|feet|mile|hour|minute|second|dozen|cent|dollar",
                text,
            )
        ),
        "nested_fraction": len(
            re.findall(r"/|half|third|quarter|fifth|sixth|seventh|eighth", text)
        ) >= 2,
        "ratio_or_rate": any(
            phrase in text
            for phrase in ("ratio", "rate", "for every", " per ", " each ")
        ),
        "comparison_or_remainder": any(
            phrase in text
            for phrase in (
                "more than",
                "less than",
                "difference",
                "left",
                "remaining",
                "twice",
                "double",
                "triple",
            )
        ),
    }
    complex_feature_count = sum(1 for value in flags.values() if value)
    score = (
        answer_step_count * 1.4
        + expression_op_count * 1.4
        + variable_count * 0.35
        + question_word_count / 35.0
        + complex_feature_count * 0.9
    )
    if flags["probability"]:
        score += 1.0
    if flags["multi_percent"]:
        score += 1.0
    if flags["average"]:
        score += 0.7

    if (
        answer_step_count <= 2
        and expression_op_count <= 3
        and variable_count <= 8
        and question_word_count <= 65
        and complex_feature_count <= 2
    ):
        absolute_label = "easy"
    elif (
        answer_step_count >= 5
        or expression_op_count >= 6
        or variable_count >= 10
        or question_word_count >= 80
        or score >= 14.0
        or (complex_feature_count >= 4 and score >= 11.0)
    ):
        absolute_label = "hard"
    else:
        absolute_label = "medium"

    rationale_parts = [
        f"{answer_step_count} solution step(s)",
        f"{expression_op_count} parsed arithmetic op(s)",
        f"{variable_count} variable(s)",
        f"{question_word_count} question word(s)",
    ]
    active_flags = [name for name, value in flags.items() if value]
    if active_flags:
        rationale_parts.append("features: " + ", ".join(active_flags))

    return {
        "score": round(score, 6),
        "absolute_label": absolute_label,
        "answer_step_count": answer_step_count,
        "expression_op_count": expression_op_count,
        "variable_count": variable_count,
        "question_word_count": question_word_count,
        "complex_feature_count": complex_feature_count,
        "features": flags,
        "rationale": "; ".join(rationale_parts),
    }


def _relative_difficulty_labels_from_profiles(
    profiles: Sequence[tuple[int, dict[str, Any]]],
) -> dict[int, str]:
    """Assign easy/medium/hard as relative thirds over rubric scores."""
    scored = sorted(
        ((float(profile["score"]), idx) for idx, profile in profiles),
        key=lambda item: (item[0], item[1]),
    )
    n = len(scored)
    base = n // 3
    remainder = n % 3
    bucket_sizes = [
        base + (1 if bucket < remainder else 0)
        for bucket in range(3)
    ]
    labels: dict[int, str] = {}
    offset = 0
    for difficulty, bucket_size in zip(GSM_DIFFICULTIES, bucket_sizes):
        for _, idx in scored[offset: offset + bucket_size]:
            labels[idx] = difficulty
        offset += bucket_size
    return labels


def _explicit_gsm_difficulty(payload: dict[str, Any]) -> str | None:
    """Return an explicit easy/medium/hard label when the source row has one."""
    for key in ("difficulty", "difficulty_label", "level"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip().lower() in GSM_DIFFICULTIES:
            return value.strip().lower()

    config = payload.get("config") or payload.get("gsm_config") or payload.get("split_config")
    if isinstance(config, str):
        return GSM_CONFIG_DIFFICULTY.get(config.strip().lower())

    return None


def _load_crane_payloads(crane_dir: Path) -> list[tuple[int, Path, dict[str, Any]]]:
    paths = sorted(crane_dir.glob("*.json"))
    rows: list[tuple[int, Path, dict[str, Any]]] = []
    for idx, path in enumerate(paths):
        try:
            rows.append((idx, path, json.loads(path.read_text())))
        except Exception:
            continue
    return rows


def infer_gsm_crane_difficulty_labels(
    crane_dir: Path | str | None = None,
    prefer_existing: bool = True,
) -> dict[str, Any]:
    """
    Infer easy/medium/hard labels for a CRANE GSM folder.

    CRANE's checked-in GSM JSONs are original GSM questions and usually do not
    carry the GSM-Symbolic config label. When explicit labels are missing, this
    falls back to deterministic complexity tertiles and records that source in
    the returned manifest.
    """
    crane_dir = Path(crane_dir) if crane_dir is not None else DEFAULT_CRANE_GSM_DIR
    rows = _load_crane_payloads(crane_dir)
    if not rows:
        raise FileNotFoundError(f"No GSM JSON files found under {crane_dir}")

    labels: dict[int, str] = {}
    if prefer_existing:
        for idx, _, payload in rows:
            explicit = _explicit_gsm_difficulty(payload)
            if explicit is not None:
                labels[idx] = explicit

    if len(labels) == len(rows):
        row_sources = {
            str(payload.get("difficulty_source") or "explicit")
            for _, _, payload in rows
            if _explicit_gsm_difficulty(payload) is not None
        }
        source = (
            "json:" + next(iter(row_sources))
            if len(row_sources) == 1 and row_sources != {"explicit"}
            else "explicit"
        )
        scored_rows = []
    else:
        source = "complexity_tertile"
        scored_rows = sorted(
            (
                (_gsm_complexity_score(payload), idx, path)
                for idx, path, payload in rows
            ),
            key=lambda item: (item[0], item[1]),
        )
        n = len(scored_rows)
        base = n // 3
        remainder = n % 3
        bucket_sizes = [
            base + (1 if bucket < remainder else 0)
            for bucket in range(3)
        ]
        offset = 0
        for difficulty, bucket_size in zip(GSM_DIFFICULTIES, bucket_sizes):
            for _, idx, _ in scored_rows[offset: offset + bucket_size]:
                labels[idx] = difficulty
            offset += bucket_size

    composition = Counter(labels.values())
    return {
        "difficulty_source": source,
        "difficulty_order": list(GSM_DIFFICULTIES),
        "difficulty_by_index": {str(idx): labels[idx] for idx in sorted(labels)},
        "difficulty_composition": dict(sorted(composition.items())),
        "complexity_scores": {
            str(idx): score for score, idx, _ in scored_rows
        } if scored_rows else {},
    }


def annotate_gsm_crane_difficulty_labels(
    crane_dir: Path | str | None = None,
    *,
    overwrite: bool = False,
    backup: bool = True,
) -> dict[str, Any]:
    """
    Add easy/medium/hard difficulty metadata directly to CRANE GSM JSON files.

    The written fields are:
        difficulty, difficulty_source, difficulty_order, difficulty_score

    Existing labels are preserved unless overwrite=True.
    """
    crane_dir = Path(crane_dir) if crane_dir is not None else DEFAULT_CRANE_GSM_DIR
    rows = _load_crane_payloads(crane_dir)
    if not rows:
        raise FileNotFoundError(f"No GSM JSON files found under {crane_dir}")

    difficulty_info = infer_gsm_crane_difficulty_labels(
        crane_dir,
        prefer_existing=not overwrite,
    )
    labels = {int(idx): label for idx, label in difficulty_info["difficulty_by_index"].items()}
    scores = {
        int(idx): score for idx, score in difficulty_info.get("complexity_scores", {}).items()
    }

    updated = 0
    skipped = 0
    for idx, path, payload in rows:
        if not overwrite and _explicit_gsm_difficulty(payload) is not None:
            skipped += 1
            continue
        if backup:
            backup_path = path.with_suffix(path.suffix + ".bak_difficulty")
            if not backup_path.exists():
                backup_path.write_text(json.dumps(payload, indent=2) + "\n")
        payload["difficulty"] = labels[idx]
        payload["difficulty_source"] = difficulty_info["difficulty_source"]
        payload["difficulty_is_proxy"] = difficulty_info["difficulty_source"] != "explicit"
        payload["difficulty_order"] = list(GSM_DIFFICULTIES)
        if idx in scores:
            payload["difficulty_score"] = scores[idx]
        payload["difficulty_label_version"] = "2026-05-05"
        path.write_text(json.dumps(payload, indent=2) + "\n")
        updated += 1

    return {
        "crane_dir": str(crane_dir),
        "updated_files": updated,
        "skipped_existing_files": skipped,
        **difficulty_info,
    }


def _summarize_hf_gsm_records(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    configs = sorted({str(record["config"]) for record in records})
    difficulties = sorted({str(record["difficulty"]) for record in records})
    ids_by_config: dict[str, list[Any]] = {}
    instances_by_config: dict[str, dict[str, Any]] = {}
    for config in configs:
        config_records = [record for record in records if record["config"] == config]
        ids_by_config[config] = sorted({record.get("id") for record in config_records})
        instances = sorted(
            record.get("instance")
            for record in config_records
            if record.get("instance") is not None
        )
        if instances:
            instances_by_config[config] = {
                "min": instances[0],
                "max": instances[-1],
                "count": len(instances),
            }
    return {
        "row_count": len(records),
        "configs": configs,
        "difficulties": difficulties,
        "ids_by_config": ids_by_config,
        "instances_by_config": instances_by_config,
    }


def _unique_hf_difficulty(records: Sequence[dict[str, Any]]) -> str | None:
    difficulties = {str(record["difficulty"]) for record in records}
    return next(iter(difficulties)) if len(difficulties) == 1 else None


def _load_hf_gsm_symbolic_match_indices(
    split: str = "test",
) -> dict[str, dict[Any, list[dict[str, Any]]]]:
    """Index HF GSM-Symbolic rows by generated question, original question, and id."""
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError(
            "Missing dependency `datasets`. Install with: pip install datasets"
        ) from e

    by_question: dict[str, list[dict[str, Any]]] = {}
    by_original_question: dict[str, list[dict[str, Any]]] = {}
    by_original_id: dict[int, list[dict[str, Any]]] = {}
    by_id: dict[int, list[dict[str, Any]]] = {}

    for config, difficulty in GSM_CONFIG_DIFFICULTY.items():
        ds = load_dataset("apple/GSM-Symbolic", name=config, split=split)
        for row in ds:
            record = {
                "config": config,
                "difficulty": difficulty,
                "id": row.get("id"),
                "instance": row.get("instance"),
                "original_id": row.get("original_id"),
            }
            question = _normalize_gsm_text(row.get("question") or "")
            if question:
                by_question.setdefault(question, []).append(record)
            original_question = _normalize_gsm_text(row.get("original_question") or "")
            if original_question:
                by_original_question.setdefault(original_question, []).append(record)
            try:
                by_original_id.setdefault(int(row.get("original_id")), []).append(record)
            except (TypeError, ValueError):
                pass
            try:
                by_id.setdefault(int(row.get("id")), []).append(record)
            except (TypeError, ValueError):
                pass

    return {
        "by_question": by_question,
        "by_original_question": by_original_question,
        "by_original_id": by_original_id,
        "by_id": by_id,
    }


def annotate_gsm_crane_hf_difficulty_matches(
    crane_dir: Path | str | None = None,
    *,
    split: str = "test",
    overwrite: bool = False,
    backup: bool = True,
) -> dict[str, Any]:
    """
    Try to label CRANE GSM JSONs by matching them to apple/GSM-Symbolic.

    Labeling only changes `difficulty` when a match maps to exactly one HF
    difficulty. Ambiguous matches are recorded in `hf_gsm_symbolic_match` but do
    not overwrite existing proxy labels.
    """
    crane_dir = Path(crane_dir) if crane_dir is not None else DEFAULT_CRANE_GSM_DIR
    rows = _load_crane_payloads(crane_dir)
    if not rows:
        raise FileNotFoundError(f"No GSM JSON files found under {crane_dir}")

    indices = _load_hf_gsm_symbolic_match_indices(split=split)
    stats: Counter[str] = Counter()
    updated = 0
    label_updates = 0

    for idx, path, payload in rows:
        question = _normalize_gsm_text(payload.get("question") or "")
        original_id = payload.get("id_orig")
        shuffled_id = payload.get("id_shuffled")

        candidates: list[tuple[str, list[dict[str, Any]]]] = []
        if question:
            candidates.append(("question", indices["by_question"].get(question, [])))
            candidates.append(("original_question", indices["by_original_question"].get(question, [])))
        if isinstance(original_id, int):
            candidates.append(("original_id", indices["by_original_id"].get(original_id, [])))
        if isinstance(shuffled_id, int):
            candidates.append(("id", indices["by_id"].get(shuffled_id, [])))

        match_kind = "none"
        match_records: list[dict[str, Any]] = []
        first_ambiguous_kind = "none"
        first_ambiguous_records: list[dict[str, Any]] = []
        assigned_difficulty: str | None = None
        for kind, records in candidates:
            if not records:
                continue
            assigned_difficulty = _unique_hf_difficulty(records)
            if assigned_difficulty is not None:
                match_kind = kind
                match_records = records
                break
            if not first_ambiguous_records:
                first_ambiguous_kind = kind
                first_ambiguous_records = records

        if assigned_difficulty is None and first_ambiguous_records:
            match_kind = first_ambiguous_kind
            match_records = first_ambiguous_records

        if not match_records:
            stats["no_hf_match"] += 1
            match_payload = {
                "dataset": "apple/GSM-Symbolic",
                "split": split,
                "match_kind": "none",
                "label_status": "no_match",
            }
        else:
            summary = _summarize_hf_gsm_records(match_records)
            ambiguous = assigned_difficulty is None
            stats[f"{match_kind}_{'ambiguous' if ambiguous else 'unique'}"] += 1
            match_payload = {
                "dataset": "apple/GSM-Symbolic",
                "split": split,
                "match_kind": match_kind,
                "label_status": "ambiguous" if ambiguous else "unique",
                "assigned_difficulty": assigned_difficulty,
                **summary,
            }

        if backup:
            backup_path = path.with_suffix(path.suffix + ".bak_hf_match")
            if not backup_path.exists():
                backup_path.write_text(json.dumps(payload, indent=2) + "\n")

        payload["hf_gsm_symbolic_match"] = match_payload
        if assigned_difficulty is not None and (
            overwrite or payload.get("difficulty_source") in {None, "complexity_tertile"}
        ):
            payload["difficulty"] = assigned_difficulty
            payload["difficulty_source"] = f"hf_gsm_symbolic_{match_kind}"
            payload["difficulty_is_proxy"] = False
            label_updates += 1
        path.write_text(json.dumps(payload, indent=2) + "\n")
        updated += 1

    return {
        "crane_dir": str(crane_dir),
        "hf_dataset": "apple/GSM-Symbolic",
        "hf_split": split,
        "updated_files": updated,
        "difficulty_label_updates": label_updates,
        "match_stats": dict(sorted(stats.items())),
    }


def annotate_gsm_crane_rubric_difficulty_labels(
    crane_dir: Path | str | None = None,
    *,
    backup: bool = True,
) -> dict[str, Any]:
    """
    Add auditable rubric-based absolute difficulty labels to CRANE GSM JSONs.

    `difficulty` is the absolute rubric judgment. This is more semantically
    faithful than forcing the local 100-example CRANE set into equal thirds.
    """
    crane_dir = Path(crane_dir) if crane_dir is not None else DEFAULT_CRANE_GSM_DIR
    rows = _load_crane_payloads(crane_dir)
    if not rows:
        raise FileNotFoundError(f"No GSM JSON files found under {crane_dir}")

    profiles = [(idx, _gsm_rubric_profile(payload)) for idx, _, payload in rows]
    profile_by_idx = dict(profiles)
    composition = Counter(profile["absolute_label"] for _, profile in profiles)

    updated = 0
    for idx, path, payload in rows:
        if backup:
            backup_path = path.with_suffix(path.suffix + ".bak_rubric_difficulty")
            if not backup_path.exists():
                backup_path.write_text(json.dumps(payload, indent=2) + "\n")
        profile = profile_by_idx[idx]
        payload["difficulty"] = profile["absolute_label"]
        payload["difficulty_source"] = "rubric_absolute_v1"
        payload["difficulty_is_proxy"] = False
        payload["difficulty_score"] = profile["score"]
        payload["difficulty_absolute_label"] = profile["absolute_label"]
        payload["difficulty_rubric"] = {
            key: value
            for key, value in profile.items()
            if key not in {"score", "absolute_label"}
        }
        payload["difficulty_label_version"] = "2026-05-05-rubric-v1"
        path.write_text(json.dumps(payload, indent=2) + "\n")
        updated += 1

    return {
        "crane_dir": str(crane_dir),
        "updated_files": updated,
        "difficulty_source": "rubric_absolute_v1",
        "difficulty_composition": dict(sorted(composition.items())),
        "rubric_version": "2026-05-05-rubric-v1",
    }


def _validate_difficulty_counts(counts: dict[str, int], field_name: str) -> dict[str, int]:
    normalized: dict[str, int] = {}
    for difficulty in GSM_DIFFICULTIES:
        value = int(counts.get(difficulty, 0))
        if value < 0:
            raise ValueError(f"{field_name}[{difficulty}] must be non-negative")
        normalized[difficulty] = value
    unknown = sorted(set(counts) - set(GSM_DIFFICULTIES))
    if unknown:
        raise ValueError(f"Unknown GSM difficulty labels in {field_name}: {unknown}")
    if sum(normalized.values()) <= 0:
        raise ValueError(f"{field_name} must select at least one example")
    return normalized


def make_gsm_proportional_train_eval_split(
    crane_dir: Path | str | None = None,
    train_size: int = 49,
    eval_size: int = 49,
    seed: int = 429,
) -> dict[str, Any]:
    """Create a disjoint train/eval split with benchmark-proportional difficulty mix.

    Defaults reproduce the committed gsm_symbolic_crane_proportional_49x49_seed429
    manifest: a 49/49 disjoint split of the 100-example CRANE GSM pool. seed=429
    keeps the difficulty mix balanced across train and eval (it places all the
    hard-failure indices in train so the held-out 49 is a clean ceiling). The CRANE
    baseline is scored on this same held-out 49, so the comparison stays fair.
    """
    if train_size < 0 or eval_size < 0:
        raise ValueError("train_size and eval_size must be non-negative")
    if train_size + eval_size <= 0:
        raise ValueError("train_size + eval_size must be positive")

    crane_dir = Path(crane_dir) if crane_dir is not None else DEFAULT_CRANE_GSM_DIR
    rows = _load_crane_payloads(crane_dir)
    if not rows:
        raise FileNotFoundError(f"No GSM JSON files found under {crane_dir}")

    difficulty_info = infer_gsm_crane_difficulty_labels(crane_dir)
    labels = {int(idx): label for idx, label in difficulty_info["difficulty_by_index"].items()}
    selected = stratified_sample_indices(
        labels,
        split_sizes={"train": train_size, "eval": eval_size},
        difficulties=GSM_DIFFICULTIES,
        seed=seed,
    )
    population = {
        difficulty: len([idx for idx, label in labels.items() if label == difficulty])
        for difficulty in GSM_DIFFICULTIES
    }
    return split_manifest_metadata(
        seed=seed,
        split_strategy="stratified_proportional",
        labels_by_index=labels,
        split_sizes={"train": train_size, "eval": eval_size},
        selected=selected,
        extra={
            "crane_dir": str(crane_dir),
            "train_size_requested": train_size,
            "eval_size_requested": eval_size,
            "train_allocations_requested": proportional_allocations(
                train_size, population, order=GSM_DIFFICULTIES
            ),
            "eval_allocations_requested": proportional_allocations(
                eval_size, population, order=GSM_DIFFICULTIES
            ),
            **difficulty_info,
        },
    )


def make_gsm_stratified_train_eval_split(
    crane_dir: Path | str | None = None,
    train_counts: Optional[dict[str, int]] = None,
    eval_counts: Optional[dict[str, int]] = None,
    seed: int = 123,
    *,
    train_size: Optional[int] = None,
    eval_size: Optional[int] = None,
) -> dict[str, Any]:
    """Create a deterministic train/eval split with requested difficulty counts.

    When ``train_size`` / ``eval_size`` are provided, difficulty counts are derived
    proportionally from the local CRANE GSM pool instead of the legacy 16/16/16 default.
    """
    if train_size is not None or eval_size is not None:
        return make_gsm_proportional_train_eval_split(
            crane_dir=crane_dir,
            train_size=0 if train_size is None else train_size,
            eval_size=100 if eval_size is None else eval_size,
            seed=seed,
        )

    crane_dir = Path(crane_dir) if crane_dir is not None else DEFAULT_CRANE_GSM_DIR
    rows = _load_crane_payloads(crane_dir)
    if not rows:
        raise FileNotFoundError(f"No GSM JSON files found under {crane_dir}")

    train_counts = _validate_difficulty_counts(
        train_counts or {difficulty: 16 for difficulty in GSM_DIFFICULTIES},
        "train_counts",
    )
    eval_counts = _validate_difficulty_counts(
        eval_counts or {difficulty: 16 for difficulty in GSM_DIFFICULTIES},
        "eval_counts",
    )
    difficulty_info = infer_gsm_crane_difficulty_labels(crane_dir)
    labels = {int(idx): label for idx, label in difficulty_info["difficulty_by_index"].items()}

    by_difficulty: dict[str, list[int]] = {difficulty: [] for difficulty in GSM_DIFFICULTIES}
    for idx, _, _ in rows:
        by_difficulty[labels[idx]].append(idx)

    rng = random.Random(seed)
    train_indices: list[int] = []
    eval_indices: list[int] = []
    for difficulty in GSM_DIFFICULTIES:
        needed = train_counts[difficulty] + eval_counts[difficulty]
        available = list(by_difficulty[difficulty])
        if len(available) < needed:
            raise ValueError(
                f"Not enough {difficulty} GSM examples for stratified split: "
                f"need {needed} ({train_counts[difficulty]} train + "
                f"{eval_counts[difficulty]} eval), have {len(available)}"
            )
        rng.shuffle(available)
        train_indices.extend(available[: train_counts[difficulty]])
        eval_indices.extend(available[train_counts[difficulty]: needed])

    train_indices = sorted(train_indices)
    eval_indices = sorted(eval_indices)

    def composition(indices: Sequence[int]) -> dict[str, int]:
        return dict(sorted(Counter(labels[idx] for idx in indices).items()))

    return {
        "seed": seed,
        "split_strategy": "stratified",
        "crane_dir": str(crane_dir),
        "total_examples": len(rows),
        "train_indices": train_indices,
        "eval_indices": eval_indices,
        "train_size": len(train_indices),
        "eval_size": len(eval_indices),
        "train_counts_requested": train_counts,
        "eval_counts_requested": eval_counts,
        "train_composition": composition(train_indices),
        "eval_composition": composition(eval_indices),
        **difficulty_info,
    }


def write_gsm_stratified_train_eval_split(
    output_path: Path | str,
    crane_dir: Path | str | None = None,
    train_counts: Optional[dict[str, int]] = None,
    eval_counts: Optional[dict[str, int]] = None,
    seed: int = 123,
    *,
    train_size: Optional[int] = None,
    eval_size: Optional[int] = None,
) -> dict[str, Any]:
    """Write a stratified train/eval manifest for the local CRANE GSM folder."""
    if train_size is not None or eval_size is not None:
        split = make_gsm_proportional_train_eval_split(
            crane_dir=crane_dir,
            train_size=0 if train_size is None else train_size,
            eval_size=100 if eval_size is None else eval_size,
            seed=seed,
        )
    else:
        split = make_gsm_stratified_train_eval_split(
            crane_dir=crane_dir,
            train_counts=train_counts,
            eval_counts=eval_counts,
            seed=seed,
        )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(split, indent=2) + "\n")
    return split


def write_gsm_proportional_train_eval_split(
    output_path: Path | str,
    crane_dir: Path | str | None = None,
    train_size: int = 49,
    eval_size: int = 49,
    seed: int = 429,
) -> dict[str, Any]:
    """Write a proportional stratified GSM manifest for the local CRANE GSM folder."""
    return write_gsm_stratified_train_eval_split(
        output_path,
        crane_dir=crane_dir,
        seed=seed,
        train_size=train_size,
        eval_size=eval_size,
    )
