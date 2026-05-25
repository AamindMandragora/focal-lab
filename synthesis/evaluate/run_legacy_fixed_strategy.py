"""Run fixed baseline strategies using legacy repository code paths."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from synthesis.evaluate.vendored_syncode import ensure_vendored_syncode_importable

ensure_vendored_syncode_importable()

from synthesis.evaluate.completion_text import completion_for_scoring, strip_prompt_prefix
from synthesis.evaluate.rejection_sampling import (
    DEFAULT_REJECTION_SEARCH_STEPS,
    build_rejection_sampling_session,
    rejection_sample_completion,
)


_MAX_PROMPT_CHARS = 50000  # ~12.5K tokens; leaves room for generation within 16384-token context


def _truncate_prompt(prompt: str, base_prompt: str) -> str:
    """Keep *prompt* within ``_MAX_PROMPT_CHARS`` by dropping the oldest appended molecules."""
    if len(prompt) <= _MAX_PROMPT_CHARS:
        return prompt
    suffix = prompt[len(base_prompt):]
    lines = suffix.split("\n")
    while len(base_prompt) + len("\n".join(lines)) > _MAX_PROMPT_CHARS and len(lines) > 1:
        lines.pop(0)
    return base_prompt + "\n".join(lines)


def _ensure_repo_cache_env() -> Path:
    """Point HF + SynCode pickles at a single repo-local ``cache/`` directory.

    Legacy CRANE/IterGen historically defaulted to ``legacy/CRANE/src/iter_cache/``
    (cwd-relative), duplicating multi‑GB model snapshots. Setting ``CSD_CACHE_ROOT``
    (or these defaults) keeps Hugging Face checkpoints and ``mask_stores/`` / ``parsers/``
    together under ``<repo>/cache/``.
    """
    repo_root = Path(__file__).resolve().parents[2]
    cache_root = Path(os.environ.get("CSD_CACHE_ROOT", str(repo_root / "cache"))).expanduser().resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    root_s = str(cache_root)

    os.environ.setdefault("CSD_CACHE_ROOT", root_s)
    os.environ.setdefault("HF_HOME", root_s)
    os.environ.setdefault("HF_CACHE", root_s)
    os.environ.setdefault("TRANSFORMERS_CACHE", root_s)

    syn_existing = os.environ.get("SYNCODE_CACHE") or os.environ.get("ITER_SYNCODE_CACHE")
    if syn_existing:
        syn = syn_existing if syn_existing.endswith(os.sep) else syn_existing + os.sep
        os.environ.setdefault("SYNCODE_CACHE", syn)
        os.environ.setdefault("ITER_SYNCODE_CACHE", syn)
    else:
        syn = root_s if root_s.endswith(os.sep) else root_s + os.sep
        os.environ.setdefault("SYNCODE_CACHE", syn)
        os.environ.setdefault("ITER_SYNCODE_CACHE", syn)

    return cache_root


def _normalize_dataset(dataset: str) -> str:
    return "gsm_symbolic" if dataset == "gsm" else dataset


def _baseline_row_question(dataset: str, example: dict[str, Any], fallback: str) -> str:
    """Question string stored in baseline JSON and used for GSM example lookup."""
    from synthesis.evaluate.baseline_store import normalize_baseline_question

    return normalize_baseline_question(dataset, example=example, fallback=fallback)


def _legacy_adapter_baseline_row(
    *,
    dataset: str,
    example: dict[str, Any],
    prompt: str,
    raw_generated: str,
    extracted: str | None,
    correct: bool,
    syntax_valid: bool,
    generation_seconds: float | None = None,
    num_tokens: int | None = None,
) -> dict[str, Any]:
    """Build a unified baseline row for ``baseline_store`` export."""
    from synthesis.evaluate.baseline_store import compose_baseline_answer
    from synthesis.evaluate.completion_text import completion_for_scoring

    generated_suffix = completion_for_scoring(prompt, raw_generated)
    return compose_baseline_answer(
        dataset=dataset,
        example=example,
        prompt=prompt,
        generated=generated_suffix,
        extracted=extracted or "",
        correct=correct,
        syntax_valid=syntax_valid,
        generation_seconds=generation_seconds,
        num_tokens=num_tokens,
    )


def _legacy_benchmark_prompt(
    logic: Any,
    evaluator: Any,
    example: dict[str, Any],
    profile: str,
    *,
    dataset: str,
    strategy: str,
) -> str:
    """Render the standardized tier prompt for a legacy fixed strategy row.

    profile:
      - ``expression_only`` → tier 1 (GCD, IterGen, rejection sampling).
      - ``chain_of_thought`` / ``evaluator_default`` → tier 2 (Unconstrained, CRANE paths in-repo).
    """
    from synthesis.evaluate.prompt_tiers import (
        format_prompt_for_tier,
        prompt_tier_for_strategy,
    )

    tier = 1 if profile == "expression_only" else 2
    if profile in ("evaluator_default", "chain_of_thought"):
        tier = 2
    elif profile == "expression_only":
        tier = 1
    else:
        tier = prompt_tier_for_strategy(strategy)

    constrained = profile == "expression_only" and strategy in ("gcd", "itergen")
    if profile == "expression_only" and strategy == "rejection_sampling":
        constrained = True
    return format_prompt_for_tier(
        evaluator,
        example,
        benchmark=dataset,
        tier=tier,
        constrained_suffix=constrained,
        strategy=strategy,
    )


def _baseline_run_metadata(
    args: argparse.Namespace,
    dataset: str,
    *,
    adapter: str | None = None,
) -> dict[str, Any]:
    from synthesis.evaluate.prompt_tiers import (
        benchmark_max_new_tokens,
        effective_max_new_tokens,
        prompt_tier_for_strategy,
    )

    tier = prompt_tier_for_strategy(args.strategy)
    meta: dict[str, Any] = {
        "benchmark": dataset,
        "strategy": args.strategy,
        "prompt_tier": tier,
        "eval_model": args.eval_model,
        "benchmark_max_new_tokens": benchmark_max_new_tokens(dataset),
        "effective_max_new_tokens": effective_max_new_tokens(dataset, args.eval_max_steps),
    }
    if adapter:
        meta["adapter"] = adapter
    if args.strategy == "rejection_sampling":
        meta["rejection_search_steps"] = int(
            getattr(args, "rejection_search_steps", DEFAULT_REJECTION_SEARCH_STEPS)
        )
        meta["rejection_temperature"] = 1.0
    return meta


def _legacy_gsm_symbolic_grammar_base(repo_root: Path, examples: list[dict[str, Any]]) -> str:
    """Tighten ``gsm.lark`` from a batch (allowed vars / numeric-only), matching GCD semantics.

    Returns grammar text still using ``syncode: \"<<\" start \">>\"`` (full delimited span).
    """
    from synthesis.evaluate.benchmarks.gsm_symbolic.grammar import (
        build_dynamic_grammar,
        build_numeric_only_grammar,
        extract_variables_from_mapping,
    )

    base_gsm_grammar = (repo_root / "synthesis" / "evaluate" / "grammars" / "gsm.lark").read_text()
    variable_names: set[str] = set()
    for ex in examples:
        vt = ex.get("variable_types") or {}
        if isinstance(vt, dict):
            variable_names.update(extract_variables_from_mapping(vt))
    gsm_allowed_variables = sorted(variable_names)
    if gsm_allowed_variables:
        return build_dynamic_grammar(base_gsm_grammar, gsm_allowed_variables)
    return build_numeric_only_grammar(base_gsm_grammar)


def _legacy_sql_grammar_base(repo_root: Path) -> str:
    return (repo_root / "synthesis" / "evaluate" / "grammars" / "sql.lark").read_text()


def _legacy_delimited_span_grammar(base_grammar: str) -> str:
    from synthesis.evaluate.benchmarks.common.delimiter_grammar import (
        build_delimited_span_grammar,
    )

    return build_delimited_span_grammar(base_grammar)


def _legacy_constrained_body_grammar(base_grammar: str, *, require_symbolic: bool) -> str:
    from synthesis.evaluate.benchmarks.common.delimiter_grammar import (
        build_constrained_body_grammar,
    )

    return build_constrained_body_grammar(base_grammar, require_symbolic=require_symbolic)


def _gsm_example_requires_symbolic_grammar(example: dict[str, Any]) -> bool:
    from synthesis.evaluate.benchmarks.gsm_symbolic.grammar import (
        extract_variables_from_mapping,
    )

    vt = example.get("variable_types") or {}
    if isinstance(vt, dict) and extract_variables_from_mapping(vt):
        return True
    return False


def _tier1_grammar_for_example(
    repo_root: Path,
    dataset: str,
    example: dict[str, Any],
) -> str:
    if dataset == "gsm_symbolic":
        base = _legacy_gsm_symbolic_grammar_base(repo_root, [example])
        return _legacy_constrained_body_grammar(
            base,
            require_symbolic=_gsm_example_requires_symbolic_grammar(example),
        )
    if dataset == "spider":
        return _legacy_constrained_body_grammar(
            _legacy_sql_grammar_base(repo_root),
            require_symbolic=False,
        )
    raise ValueError(f"Unsupported dataset for tier-1 constrained grammar: {dataset}")



def _gsm_symbolic_scored_body(completion: str) -> str:
    """First-line GSM expression body for tier-1 constrained decoders (no delimiters)."""
    lines = (completion or "").strip().splitlines()
    if not lines:
        return ""
    text = lines[0].strip()
    if text.startswith("<<") and text.endswith(">>"):
        text = text[2:-2].strip()
    elif text.startswith("<<"):
        text = text[2:].strip()
        if text.endswith(">>"):
            text = text[:-2].strip()
    return text


def _legacy_local_cuda_device(device_arg: str, *, touch_cuda: bool = True) -> str:
    """CUDA device string valid for the GPUs visible in this process.

    When ``touch_cuda`` is false, do not call ``torch.cuda`` (required before vLLM
  spawns workers so the parent has not initialized CUDA yet).
    """
    if device_arg and device_arg not in {"auto", "cuda"}:
        if device_arg.startswith("cuda"):
            return device_arg
        return f"cuda:{device_arg}"
    if not touch_cuda:
        return "cuda:0"
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
    except Exception:
        pass
    return "cuda"


def _legacy_cuda_device_for_backend(device_arg: str, backend: str) -> str:
    return _legacy_local_cuda_device(device_arg, touch_cuda=backend != "vllm")






def _configure_fixed_eval_runtime(eval_runtime: Any, args: argparse.Namespace, dataset: str) -> None:
    if dataset == "gsm_symbolic":
        repo_root = Path(__file__).resolve().parents[2]
        env_gsm = os.environ.get("CRANE_GSM_SYMBOLIC_DIR")
        eval_runtime.gsm_source_dir = (
            Path(env_gsm).expanduser()
            if env_gsm
            else repo_root / "legacy" / "CRANE" / "src" / "gsm_symbolic"
        )
        if args.gsm_split_file:
            eval_runtime.gsm_split_file = Path(args.gsm_split_file)
        eval_runtime.gsm_split_name = args.gsm_split_name
    if dataset == "spider":
        if args.spider_split_file:
            eval_runtime.spider_split_file = Path(args.spider_split_file)
        eval_runtime.spider_split_name = args.spider_split_name


def _crane_grammar_name(dataset: str) -> str:
    if dataset == "gsm_symbolic":
        return "gsm"
    if dataset == "spider":
        return "sql"
    raise ValueError(f"CRANE adapter supports gsm_symbolic/spider, got {dataset}")


def _mode_for_strategy(strategy: str) -> tuple[str, bool]:
    if strategy == "unconstrained":
        # Match CRANE GSM/SQL protocol: always request chain-of-thought from the subprocess model.
        return "original", True
    if strategy == "crane":
        return "adaptive", True
    raise ValueError(f"Unsupported CRANE-backed strategy: {strategy}")


def _aggregate_run_metrics(
    rows: list[dict[str, Any]],
    *,
    run_wall_time_seconds: float | None,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {"num_examples": len(rows)}
    times = [
        float(r["generation_seconds"])
        for r in rows
        if r.get("generation_seconds") is not None
    ]
    toks = [int(r["num_tokens"]) for r in rows if r.get("num_tokens") is not None]
    if times:
        metrics["total_generation_seconds"] = round(sum(times), 4)
        metrics["mean_generation_seconds_per_example"] = round(sum(times) / len(times), 6)
        metrics["examples_with_generation_timing"] = len(times)
    if toks:
        metrics["total_output_tokens"] = int(sum(toks))
        metrics["mean_output_tokens_per_example"] = round(sum(toks) / len(toks), 4)
        metrics["examples_with_token_counts"] = len(toks)
    if run_wall_time_seconds is not None:
        metrics["run_wall_time_seconds"] = round(float(run_wall_time_seconds), 4)
    return metrics


def _build_minimal_json(
    rows: list[dict[str, Any]],
    output_json: Path,
    *,
    dataset: str,
    run_wall_time_seconds: float | None = None,
    extra_metrics: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    from synthesis.evaluate.baseline_store import save_minimal_baseline_from_rows

    save_minimal_baseline_from_rows(
        rows,
        output_json,
        dataset=_normalize_dataset(dataset),
        run_wall_time_seconds=run_wall_time_seconds,
        extra_metrics=extra_metrics,
        metadata=metadata,
    )


def _checkpoint_baseline_json(
    rows: list[dict[str, Any]],
    output_json: Path,
    *,
    dataset: str,
    run_started: float,
    extra_metrics: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Write partial baseline JSON so a long baseline run survives interruption."""
    meta = dict(metadata or {})
    meta["checkpoint"] = True
    meta["checkpoint_examples"] = len(rows)
    metrics = dict(extra_metrics or {})
    metrics["checkpoint_examples"] = len(rows)
    _build_minimal_json(
        rows,
        output_json,
        dataset=dataset,
        run_wall_time_seconds=time.perf_counter() - run_started,
        extra_metrics=metrics,
        metadata=meta,
    )


def _enrich_crane_baseline_rows(
    rows: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
    dataset: str,
    logic: Any,
    eval_runtime: Any,
    examples: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Attach prompts, normalized questions, and extracted answers to CRANE jsonl rows."""
    from synthesis.evaluate.baseline_store import normalize_baseline_question
    from synthesis.evaluate.completion_text import completion_for_scoring

    profile = "chain_of_thought"
    examples_by_question: dict[str, dict[str, Any]] = {}
    for example in examples:
        if dataset == "gsm_symbolic":
            for key in (
                example.get("question_parsed"),
                example.get("original_question"),
                example.get("question"),
                example.get("prompt"),
            ):
                if key:
                    examples_by_question[str(key)] = example
        else:
            q = normalize_baseline_question(dataset, example=example)
            if q:
                examples_by_question[q] = example
            embedded = str(example.get("question") or "")
            if embedded:
                examples_by_question[embedded] = example

    enriched: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        question_raw = str(row.get("question") or "")
        example = examples_by_question.get(question_raw)
        if example is None:
            norm_q = normalize_baseline_question(dataset, row=row)
            example = examples_by_question.get(norm_q)
        if example is None and idx < len(examples):
            example = examples[idx]

        prompt = row.get("prompt_used") or row.get("prompt")
        if not prompt and example is not None:
            prompt = _legacy_benchmark_prompt(
                logic,
                eval_runtime,
                example,
                profile,
                dataset=dataset,
                strategy=args.strategy,
            )
        prompt_s = str(prompt or "")

        raw_generated = str(
            row.get("llm_response") or row.get("response") or row.get("pred") or ""
        )
        extracted = row.get("parsed_completion")
        aux_for_row: dict[str, Any] | None = None
        if example is not None and (
            extracted is None or dataset == "spider"
        ):
            completion = completion_for_scoring(prompt_s or None, raw_generated)
            scored = (
                eval_runtime._truncate_gsm_output(completion)
                if dataset == "gsm_symbolic"
                else completion
            )
            extracted, _src, aux_for_row = logic.extract_actual(
                eval_runtime, scored, example
            )

        correct = bool(row.get("correct")) if isinstance(row.get("correct"), bool) else False
        syntax_valid = (
            bool(row.get("syntax_valid"))
            if isinstance(row.get("syntax_valid"), bool)
            else False
        )
        if aux_for_row is not None:
            syntax_valid = bool(aux_for_row.get("syntax_valid"))

        enriched.append(
            _legacy_adapter_baseline_row(
                dataset=dataset,
                example=example or {"question": question_raw},
                prompt=prompt_s,
                raw_generated=raw_generated,
                extracted=str(extracted) if extracted is not None else None,
                correct=correct,
                syntax_valid=syntax_valid,
                generation_seconds=row.get("generation_seconds"),
                num_tokens=row.get("num_tokens"),
            )
        )
    return enriched


def _load_latest_crane_results(crane_src_dir: Path, dataset: str) -> list[dict[str, Any]]:
    dataset_dir = crane_src_dir / "logging" / dataset
    if not dataset_dir.exists():
        return []

    candidates = sorted(dataset_dir.rglob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        return []

    latest = candidates[0]
    rows: list[dict[str, Any]] = []
    for line in latest.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _annotate_legacy_rows_with_syntax(
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    dataset: str,
) -> list[dict[str, Any]]:
    """Add benchmark syntax booleans to legacy rows that only report correctness."""
    if not rows:
        return rows
    if any(isinstance(row.get("syntax_valid"), bool) for row in rows):
        return rows

    from synthesis.evaluate.benchmarks.registry import get_logic
    from synthesis.evaluate.evaluator import Evaluator

    logic = get_logic(dataset)
    eval_runtime = Evaluator(
        dataset_name=dataset,
        model_name=args.eval_model,
        backend=args.eval_backend,
        device=args.device,
        sample_size=args.eval_sample_size,
        max_steps=args.eval_max_steps,
        step_token_budget=args.eval_step_token_budget,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
        gsm_split_file=args.gsm_split_file if dataset == "gsm_symbolic" else None,
        gsm_split_name=args.gsm_split_name,
        spider_split_file=args.spider_split_file if dataset == "spider" else None,
        spider_split_name=args.spider_split_name,
    )
    _configure_fixed_eval_runtime(eval_runtime, args, dataset)
    examples = logic.load_dataset_sample(eval_runtime)
    examples_by_question: dict[str, dict[str, Any]] = {}
    for example in examples:
        if dataset == "gsm_symbolic":
            keys = [
                example.get("question_parsed"),
                example.get("original_question"),
                example.get("question"),
                example.get("prompt"),
                example.get("question_instantiated"),
            ]
            for key in keys:
                if key:
                    examples_by_question[str(key)] = example
        else:
            q = str(example.get("question") or example.get("prompt") or "")
            if q:
                examples_by_question[q] = example

    for idx, row in enumerate(rows):
        question = str(row.get("question") or row.get("prompt") or "")
        example = examples_by_question.get(question)
        if example is None and idx < len(examples):
            example = examples[idx]
        if dataset == "gsm_symbolic" and example is not None:
            example = dict(example)
            if not isinstance(example.get("variable_types"), dict):
                gold_answer = str(row.get("gold_answer") or "")
                from synthesis.evaluate.benchmarks.gsm_symbolic.expression_normalize import (
                    reserved_equivalence_names,
                )

                numeric_vars = sorted(
                    set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", gold_answer))
                    - reserved_equivalence_names()
                )
                if numeric_vars:
                    example["variable_types"] = {name: "int" for name in numeric_vars}
        output_text = str(
            row.get("parsed_completion")
            or row.get("llm_response")
            or row.get("response")
            or row.get("pred")
            or ""
        )
        prompt_used = str(row.get("prompt_used") or "")
        completion = completion_for_scoring(prompt_used or None, output_text)
        scored_output = (
            eval_runtime._truncate_gsm_output(completion)
            if dataset == "gsm_symbolic"
            else completion
        )
        all_valid, segments = eval_runtime._check_syntax_validity(
            scored_output,
            example=example,
        )
        if dataset == "spider":
            actual, _answer_source, _aux = logic.extract_actual(
                eval_runtime,
                scored_output,
                example or {},
            )
            syntax_valid = bool(actual and re.search(r"\bselect\b", actual, flags=re.IGNORECASE))
        else:
            syntax_valid = bool(
                logic.example_syntax_pass(all_valid, segments, False, None)
            )
        row["syntax_valid"] = syntax_valid
    return rows




















def run_rejection_sampling_legacy_adapter(args: argparse.Namespace) -> int:
    """Standard rejection sampling: temperature-1 unconstrained decode, reject until valid."""
    run_started = time.perf_counter()
    dataset = _normalize_dataset(args.dataset)
    repo_root = Path(__file__).resolve().parents[2]

    from synthesis.evaluate.benchmarks.registry import get_logic
    from synthesis.evaluate.evaluator import Evaluator

    logic = get_logic(dataset)
    eval_runtime = Evaluator(
        dataset_name=dataset,
        model_name=args.eval_model,
        backend=args.eval_backend,
        device=args.device,
        sample_size=args.eval_sample_size,
        max_steps=args.eval_max_steps,
        step_token_budget=args.eval_step_token_budget,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
        gsm_split_file=args.gsm_split_file if dataset == "gsm_symbolic" else None,
        gsm_split_name=args.gsm_split_name,
        spider_split_file=args.spider_split_file if dataset == "spider" else None,
        spider_split_name=args.spider_split_name,
    )
    _configure_fixed_eval_runtime(eval_runtime, args, dataset)
    examples = logic.load_dataset_sample(eval_runtime)

    device = _legacy_cuda_device_for_backend(args.device, args.eval_backend)
    from synthesis.evaluate.prompt_tiers import effective_max_new_tokens

    decode_cap = effective_max_new_tokens(dataset, args.eval_max_steps)

    def _rejection_max_new_tokens() -> int:
        if dataset == "gsm_symbolic":
            return min(28, decode_cap)
        return decode_cap

    def _rejection_output(completion: str, example: dict[str, Any]) -> str:
        if dataset == "gsm_symbolic":
            return _gsm_symbolic_scored_body(completion)
        return (completion or "").strip()

    from synthesis.evaluate.syncode_run_session import release_cuda_cache

    n_attempts = max(1, int(getattr(args, "rejection_search_steps", DEFAULT_REJECTION_SEARCH_STEPS)))
    print(
        f"Rejection sampling: {n_attempts} attempts/example at temperature 1.0, "
        f"max_new_tokens={_rejection_max_new_tokens()} (eval-max-steps cap only)"
    )

    session = build_rejection_sampling_session(
        args.eval_model,
        device=device,
        max_new_tokens=_rejection_max_new_tokens(),
    )

    rows: list[dict[str, Any]] = []
    run_metadata = _baseline_run_metadata(args, dataset, adapter="rejection_sampling_syncode")
    output_json = Path(args.output_json)

    try:
        if session.mode == "original":
            session.ensure_ready()
        for example in examples:
            if session.mode != "original":
                grammar_text = _tier1_grammar_for_example(repo_root, dataset, example)
                session.apply_grammar(grammar_text)

            prompt = _legacy_benchmark_prompt(
                logic,
                eval_runtime,
                example,
                "expression_only",
                dataset=dataset,
                strategy="rejection_sampling",
            )

            def _syntax_valid_for_body(body: str) -> bool:
                completion = completion_for_scoring(prompt, body)
                scored = (
                    eval_runtime._truncate_gsm_output(completion)
                    if dataset == "gsm_symbolic"
                    else completion
                )
                syntax_ok, _segments = eval_runtime._check_syntax_validity(
                    scored, example=example
                )
                if dataset == "spider":
                    actual, _, _aux = logic.extract_actual(eval_runtime, scored, example)
                    syntax_ok = bool(actual and re.search(r"\bselect\b", actual, flags=re.IGNORECASE))
                return bool(syntax_ok)

            gen_started = time.perf_counter()
            output_text, attempts_used = rejection_sample_completion(
                session,
                prompt,
                max_attempts=n_attempts,
                normalize_output=lambda raw, ex=example: _rejection_output(raw, ex),
                is_syntax_valid=_syntax_valid_for_body,
            )
            gen_seconds = time.perf_counter() - gen_started

            completion = completion_for_scoring(prompt, output_text)
            scored_output = (
                eval_runtime._truncate_gsm_output(completion)
                if dataset == "gsm_symbolic"
                else completion
            )
            expected = logic.expected_answer(eval_runtime, example)
            actual, _answer_source, aux = logic.extract_actual(eval_runtime, scored_output, example)
            is_correct = bool(logic.is_correct(eval_runtime, actual, expected, example, aux, scored_output))

            syntax_valid, _segments = eval_runtime._check_syntax_validity(scored_output, example=example)
            if dataset == "spider":
                syntax_valid = bool(actual and re.search(r"\bselect\b", actual, flags=re.IGNORECASE))
            row = _legacy_adapter_baseline_row(
                dataset=dataset,
                example=example,
                prompt=prompt,
                raw_generated=output_text,
                extracted=actual,
                correct=bool(is_correct),
                syntax_valid=bool(syntax_valid),
                generation_seconds=gen_seconds,
            )
            row["rejection_attempts"] = attempts_used
            rows.append(row)
            _checkpoint_baseline_json(
                rows,
                output_json,
                dataset=dataset,
                run_started=run_started,
                extra_metrics={"adapter": "rejection_sampling_syncode"},
                metadata=run_metadata,
            )
            print(f"Checkpoint ({len(rows)}/{len(examples)}): {output_json}", flush=True)
    finally:
        session.close()
        release_cuda_cache()

    if not rows:
        raise RuntimeError("Rejection sampling produced no rows; refusing to write an empty baseline JSON")

    final_metadata = dict(run_metadata)
    final_metadata.pop("checkpoint", None)
    final_metadata["complete"] = True
    _build_minimal_json(
        rows,
        output_json,
        dataset=dataset,
        run_wall_time_seconds=time.perf_counter() - run_started,
        extra_metrics={"adapter": "rejection_sampling_syncode"},
        metadata=final_metadata,
    )
    print(f"Saved baseline JSON: {output_json}")
    return 0


def run_gcd_legacy_adapter(args: argparse.Namespace) -> int:
    """GCD baseline via vendored SynCode (grammar_strict mode = pure hard-mask constrained decoding)."""
    run_started = time.perf_counter()
    import sys

    dataset = _normalize_dataset(args.dataset)
    repo_root = Path(__file__).resolve().parents[2]

    syncode_root = repo_root / "synthesis" / "evaluate" / "syncode"
    syncode_pkg = syncode_root / "syncode"
    for p in [str(syncode_root), str(syncode_pkg)]:
        if p not in sys.path:
            sys.path.insert(0, p)

    from synthesis.evaluate.benchmarks.registry import get_logic

    logic = get_logic(dataset)

    from synthesis.evaluate.evaluator import Evaluator
    eval_runtime = Evaluator(
        dataset_name=dataset,
        model_name=args.eval_model,
        backend=args.eval_backend,
        device=args.device,
        sample_size=args.eval_sample_size,
        max_steps=args.eval_max_steps,
        step_token_budget=args.eval_step_token_budget,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
        gsm_split_file=args.gsm_split_file if dataset == "gsm_symbolic" else None,
        gsm_split_name=args.gsm_split_name,
        spider_split_file=args.spider_split_file if dataset == "spider" else None,
        spider_split_name=args.spider_split_name,
    )
    _configure_fixed_eval_runtime(eval_runtime, args, dataset)
    examples = logic.load_dataset_sample(eval_runtime)

    device = _legacy_cuda_device_for_backend(args.device, args.eval_backend)
    base_gsm_grammar = ""
    base_spider_grammar = ""
    def _grammar_for_example(example: dict[str, Any]) -> str:
        return _tier1_grammar_for_example(repo_root, dataset, example)

    from synthesis.evaluate.prompt_tiers import effective_max_new_tokens

    decode_cap = effective_max_new_tokens(dataset, args.eval_max_steps)

    def _gcd_max_new_tokens() -> int:
        if dataset == "gsm_symbolic":
            return min(28, decode_cap)
        return decode_cap

    def _gcd_prompt(prompt: str) -> str:
        return prompt

    def _gcd_output(completion: str, example: dict[str, Any]) -> str:
        if dataset == "gsm_symbolic":
            return _gsm_symbolic_scored_body(completion)
        return (completion or "").strip()

    from synthesis.evaluate.syncode_run_session import SyncodeRunSession, release_cuda_cache

    gcd_session = SyncodeRunSession(
        args.eval_model,
        device=device,
        mode="grammar_strict",
        quantize=False,
        parse_output_only=True,
        log_level=0,
        max_new_tokens=_gcd_max_new_tokens(),
        do_sample=False,
        num_return_sequences=1,
        opp=False,
    )
    rows: list[dict[str, Any]] = []

    try:
        for example in examples:
            grammar_text = _grammar_for_example(example)
            gcd_session.apply_grammar(grammar_text)

            prompt = _legacy_benchmark_prompt(
                logic, eval_runtime, example, "expression_only", dataset=dataset, strategy="gcd"
            )
            gen_started = time.perf_counter()
            gcd_prompt = _gcd_prompt(prompt)
            completions = gcd_session.infer(gcd_prompt, stop_words=None)
            gen_seconds = time.perf_counter() - gen_started
            raw_output = _gcd_output(completions[0] if completions else "", example)
            completion = completion_for_scoring(gcd_prompt, raw_output)
            scored_output = (
                eval_runtime._truncate_gsm_output(completion)
                if dataset == "gsm_symbolic"
                else completion
            )
            expected = logic.expected_answer(eval_runtime, example)
            actual, _answer_source, aux = logic.extract_actual(eval_runtime, scored_output, example)
            is_correct = bool(logic.is_correct(eval_runtime, actual, expected, example, aux, scored_output))

            syntax_valid, _segments = eval_runtime._check_syntax_validity(scored_output, example=example)
            if dataset == "spider":
                syntax_valid = bool(actual and re.search(r"\bselect\b", actual, flags=re.IGNORECASE))
            rows.append(
                _legacy_adapter_baseline_row(
                    dataset=dataset,
                    example=example,
                    prompt=gcd_prompt,
                    raw_generated=raw_output,
                    extracted=actual,
                    correct=bool(is_correct),
                    syntax_valid=bool(syntax_valid),
                    generation_seconds=gen_seconds,
                )
            )
    finally:
        gcd_session.close()
        release_cuda_cache()

    _build_minimal_json(
        rows,
        args.output_json,
        dataset=dataset,
        run_wall_time_seconds=time.perf_counter() - run_started,
        metadata=_baseline_run_metadata(args, dataset, adapter="gcd_syncode"),
    )
    print(f"Saved baseline JSON: {args.output_json}")
    return 0


def _itergen_add_import_paths(itergen_root: Path) -> None:
    candidates = [
        itergen_root,
        itergen_root / "itergen" / "syncode",
        itergen_root / "itergen" / "syncode" / "syncode",
    ]
    for candidate in candidates:
        candidate_str = str(candidate.resolve())
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)


def _itergen_generate(iter_gen: Any, prompt: Any) -> str:
    iter_gen.start(prompt)
    generated = iter_gen.forward()
    if isinstance(generated, list):
        if not generated:
            return ""
        return str(generated[0])
    return str(generated)


def _itergen_ignore_whitespace(grammar: Any) -> bool:
    from parsers import create_base_parser

    base_parser = create_base_parser(grammar)
    import regex

    ignore_whitespace = False
    for ig_name in ("WS", "WS_INLINE"):
        for terminal in base_parser.terminals:
            if terminal.name == ig_name:
                if regex.match(terminal.pattern.to_regexp(), " ") is not None:
                    ignore_whitespace = True
    return ignore_whitespace


def _itergen_grammar_bundle(
    grammar_text: str,
    *,
    tokenizer: Any,
    grammar_cache: dict[str, tuple[Any, Any, list[Any], bool]],
    dfa_mode: str,
) -> tuple[Any, Any, list[Any], bool]:
    cached = grammar_cache.get(grammar_text)
    if cached is not None:
        return cached
    from itergen import Grammar
    from itergen.syncode.syncode.dfa_mask_store import DFAMaskStore
    from parsers import create_parser

    grammar = Grammar(grammar_text)
    ignore_whitespace = _itergen_ignore_whitespace(grammar)
    dfa_mask_store = DFAMaskStore.load_dfa_mask_store(
        grammar=grammar,
        tokenizer=tokenizer,
        use_cache=True,
        mode=dfa_mode,
    )
    inc_parsers = [create_parser(grammar, ignore_whitespace=ignore_whitespace)]
    bundle = (grammar, dfa_mask_store, inc_parsers, ignore_whitespace)
    grammar_cache[grammar_text] = bundle
    return bundle


def _itergen_rebind_grammar(
    iter_gen: Any,
    grammar_text: str,
    *,
    grammar_cache: dict[str, tuple[Any, Any, list[Any], bool]],
    dfa_mode: str,
) -> None:
    """Swap tier-1 grammar/mask state without reloading the shared HF model."""
    if getattr(iter_gen, "_vas_bound_grammar_text", None) == grammar_text:
        return
    grammar, dfa_mask_store, inc_parsers, ignore_whitespace = _itergen_grammar_bundle(
        grammar_text,
        tokenizer=iter_gen.tokenizer,
        grammar_cache=grammar_cache,
        dfa_mode=dfa_mode,
    )
    iter_gen.grammar = grammar
    iter_gen.dfa_mask_store = dfa_mask_store
    iter_gen.inc_parsers = inc_parsers
    iter_gen._ignore_whitespace = ignore_whitespace
    iter_gen._vas_bound_grammar_text = grammar_text


def run_itergen_legacy_adapter(args: argparse.Namespace) -> int:
    prev_recursion = sys.getrecursionlimit()
    sys.setrecursionlimit(max(prev_recursion, 100_000))
    try:
        return _run_itergen_legacy_adapter_inner(args)
    finally:
        sys.setrecursionlimit(prev_recursion)


def _run_itergen_legacy_adapter_inner(args: argparse.Namespace) -> int:
    run_started = time.perf_counter()
    dataset = _normalize_dataset(args.dataset)
    repo_root = Path(__file__).resolve().parents[2]
    itergen_root = repo_root / "legacy" / "itergen"
    if not itergen_root.exists():
        raise RuntimeError(f"Legacy itergen directory not found: {itergen_root}")

    _itergen_add_import_paths(itergen_root)
    from itergen.main import IterGen

    from synthesis.evaluate.evaluator import Evaluator
    from synthesis.evaluate.benchmarks.registry import get_logic

    logic = get_logic(dataset)
    eval_runtime = Evaluator(
        dataset_name=dataset,
        model_name=args.eval_model,
        backend=args.eval_backend,
        device=args.device,
        sample_size=args.eval_sample_size,
        max_steps=args.eval_max_steps,
        step_token_budget=args.eval_step_token_budget,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
        gsm_split_file=args.gsm_split_file if dataset == "gsm_symbolic" else None,
        gsm_split_name=args.gsm_split_name,
        spider_split_file=args.spider_split_file if dataset == "spider" else None,
        spider_split_name=args.spider_split_name,
    )
    _configure_fixed_eval_runtime(eval_runtime, args, dataset)
    examples = logic.load_dataset_sample(eval_runtime)

    device = _legacy_cuda_device_for_backend(args.device, args.eval_backend)

    from synthesis.evaluate.prompt_tiers import effective_max_new_tokens

    decode_cap = effective_max_new_tokens(dataset, args.eval_max_steps)

    def _itergen_max_new_tokens() -> int:
        if dataset == "gsm_symbolic":
            return min(28, decode_cap)
        return decode_cap

    _new_tok = _itergen_max_new_tokens()

    def _grammar_for_example(example: dict[str, Any]) -> str:
        return _tier1_grammar_for_example(repo_root, dataset, example)

    from synthesis.evaluate.syncode_run_session import release_cuda_cache

    itergen_grammar_cache: dict[str, tuple[Any, Any, list[Any], bool]] = {}
    iter_gen_session: Any = None
    rows: list[dict[str, Any]] = []
    _session_ceiling = min(16384, max(2048, _new_tok + 4096))
    _itergen_dfa_mode = "grammar_strict"

    try:
        for example in examples:
            grammar_text = _grammar_for_example(example)
            if iter_gen_session is None:
                iter_gen_session = IterGen(
                    grammar=grammar_text,
                    model_id=args.eval_model,
                    device=device,
                    parse_output_only=True,
                    quantize=False,
                    max_tokens=_session_ceiling,
                    do_sample=False,
                    max_new_tokens=_new_tok,
                    num_return_sequences=1,
                )
                iter_gen_session._vas_bound_grammar_text = grammar_text
                itergen_grammar_cache[grammar_text] = (
                    iter_gen_session.grammar,
                    iter_gen_session.dfa_mask_store,
                    iter_gen_session.inc_parsers,
                    iter_gen_session._ignore_whitespace,
                )
            else:
                _itergen_rebind_grammar(
                    iter_gen_session,
                    grammar_text,
                    grammar_cache=itergen_grammar_cache,
                    dfa_mode=_itergen_dfa_mode,
                )
            iter_gen = iter_gen_session

            prompt = _legacy_benchmark_prompt(
                logic, eval_runtime, example, "expression_only", dataset=dataset, strategy="itergen"
            )

            gen_started = time.perf_counter()
            raw_completion = _itergen_generate(iter_gen, prompt)
            if dataset == "gsm_symbolic":
                raw_completion = _gsm_symbolic_scored_body(raw_completion)
            else:
                raw_completion = (raw_completion or "").strip()
            gen_seconds = time.perf_counter() - gen_started
            completion = completion_for_scoring(prompt, raw_completion)
            scored_output = (
                eval_runtime._truncate_gsm_output(completion)
                if dataset == "gsm_symbolic"
                else completion
            )
            expected = logic.expected_answer(eval_runtime, example)
            actual, _answer_source, aux = logic.extract_actual(eval_runtime, scored_output, example)
            is_correct = bool(logic.is_correct(eval_runtime, actual, expected, example, aux, scored_output))

            syntax_valid, _segments = eval_runtime._check_syntax_validity(scored_output, example=example)
            if dataset == "spider":
                syntax_valid = bool(actual and re.search(r"\bselect\b", actual, flags=re.IGNORECASE))
            rows.append(
                _legacy_adapter_baseline_row(
                    dataset=dataset,
                    example=example,
                    prompt=prompt,
                    raw_generated=raw_completion,
                    extracted=actual,
                    correct=bool(is_correct),
                    syntax_valid=bool(syntax_valid),
                    generation_seconds=gen_seconds,
                )
            )
    finally:
        iter_gen_session = None
        itergen_grammar_cache.clear()
        release_cuda_cache()

    _build_minimal_json(
        rows,
        args.output_json,
        dataset=dataset,
        run_wall_time_seconds=time.perf_counter() - run_started,
        metadata=_baseline_run_metadata(args, dataset, adapter="itergen_legacy"),
    )
    print(f"Saved baseline JSON: {args.output_json}")
    return 0






def run_unconstrained_spider_adapter(args: argparse.Namespace) -> int:
    """Unconstrained Spider baseline without grammar masking in legacy CRANE ``main.py``.

    Uses the same chain-of-thought Spider prompt as other legacy adapters and scores via
    execution match against the local Spider databases (``SPIDER_DB_DIR`` or repository layout).
    """
    from synthesis.evaluate.benchmarks.registry import get_logic
    from synthesis.evaluate.evaluator import Evaluator

    run_started = time.perf_counter()
    logic = get_logic("spider")
    eval_runtime = Evaluator(
        dataset_name="spider",
        model_name=args.eval_model,
        backend=args.eval_backend,
        device=args.device,
        sample_size=args.eval_sample_size,
        max_steps=args.eval_max_steps,
        step_token_budget=args.eval_step_token_budget,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
        spider_split_file=args.spider_split_file,
        spider_split_name=args.spider_split_name,
    )
    _configure_fixed_eval_runtime(eval_runtime, args, "spider")
    examples = logic.load_dataset_sample(eval_runtime)

    if args.eval_backend == "vllm":
        from vllm import LLM

        llm = LLM(
            model=args.eval_model,
            tensor_parallel_size=args.vllm_tensor_parallel_size,
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            max_model_len=16384,
            trust_remote_code=True,
        )
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        device = _legacy_local_cuda_device(args.device)
        tokenizer = AutoTokenizer.from_pretrained(args.eval_model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.eval_model,
            device_map=device,
            dtype=torch.float16,
            trust_remote_code=True,
        )

    from synthesis.evaluate.prompt_tiers import effective_max_new_tokens

    rows: list[dict[str, Any]] = []
    max_new = effective_max_new_tokens("spider", args.eval_max_steps)

    for example in examples:
        prompt = _legacy_benchmark_prompt(
            logic,
            eval_runtime,
            example,
            "chain_of_thought",
            dataset="spider",
            strategy="unconstrained",
        )
        num_toks: int | None = None
        if args.eval_backend == "vllm":
            from vllm import SamplingParams as _SP

            gen_started = time.perf_counter()
            sp = _SP(max_tokens=max_new, temperature=0.0)
            outputs = llm.generate([prompt], sp)
            gen_seconds = time.perf_counter() - gen_started
            completion = outputs[0].outputs[0]
            suffix = completion.text
            num_toks = len(completion.token_ids) if getattr(completion, "token_ids", None) else None
        else:
            import torch

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            gen_started = time.perf_counter()
            with torch.no_grad():
                out_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new,
                    do_sample=False,
                )
            gen_seconds = time.perf_counter() - gen_started
            new_ids = out_ids[0][inputs["input_ids"].shape[1]:]
            num_toks = int(new_ids.numel())
            suffix = tokenizer.decode(new_ids, skip_special_tokens=True)

        scored_output = suffix
        expected = logic.expected_answer(eval_runtime, example)
        actual, _answer_source, aux = logic.extract_actual(eval_runtime, scored_output, example)
        is_correct = bool(logic.is_correct(eval_runtime, actual, expected, example, aux, scored_output))
        syntax_valid = bool(actual and re.search(r"\bselect\b", actual, flags=re.IGNORECASE))

        rows.append(
            _legacy_adapter_baseline_row(
                dataset="spider",
                example=example,
                prompt=prompt,
                raw_generated=suffix,
                extracted=actual,
                correct=bool(is_correct),
                syntax_valid=bool(syntax_valid),
                generation_seconds=gen_seconds,
                num_tokens=num_toks,
            )
        )

    _build_minimal_json(
        rows,
        args.output_json,
        dataset="spider",
        run_wall_time_seconds=time.perf_counter() - run_started,
        metadata=_baseline_run_metadata(args, "spider", adapter="unconstrained_spider"),
    )
    print(f"Saved baseline JSON: {args.output_json}")
    return 0


def run_crane_legacy_adapter(args: argparse.Namespace) -> int:
    from synthesis.evaluate.prompt_tiers import TIER2_FEWSHOT_CAP

    dataset = _normalize_dataset(args.dataset)

    if dataset == "spider" and args.strategy == "unconstrained":
        return run_unconstrained_spider_adapter(args)

    mode, do_cot = _mode_for_strategy(args.strategy)
    grammar = _crane_grammar_name(dataset)

    repo_root = Path(__file__).resolve().parents[2]
    crane_src_dir = repo_root / "legacy" / "CRANE" / "src"
    tier2_grammar_path: Path | None = None
    cot_grammar_arg = grammar if mode != "original" else "text"
    out_grammar_arg = grammar if mode != "original" else "text"
    if dataset == "spider" and mode != "original":
        tier2_sql = _legacy_delimited_span_grammar(_legacy_sql_grammar_base(repo_root))
        tier2_grammar_path = crane_src_dir / ".vas_spider_tier2.lark"
        tier2_grammar_path.write_text(tier2_sql, encoding="utf-8")
        # Syncode loads the tier-2 Lark file; CRANE PARSE_MAP keys stay "sql".
        cot_grammar_arg = str(tier2_grammar_path)
        out_grammar_arg = "sql"
    if not crane_src_dir.exists():
        raise RuntimeError(f"Legacy CRANE src directory not found: {crane_src_dir}")

    cmd = [
        "python",
        "main.py",
        "--dataset",
        dataset,
        "--num_examples",
        str(args.eval_sample_size),
        "--num_shots",
        str(TIER2_FEWSHOT_CAP),
        "--overwrite_results",
        "True",
        "--write_file",
        "True",
        "--regex_parser",
        "True",
        "--modify_system_prompt",
        "True",
        "--cot_model",
        args.eval_model,
        "--cot_grammar_mode",
        mode,
        "--cot_grammar",
        cot_grammar_arg,
        "--out_grammar",
        out_grammar_arg,
    ]
    if do_cot:
        cmd.extend(["--do_cot", "True"])

    legacy_device = _legacy_local_cuda_device(args.device)
    cmd.extend(["--cot_device", legacy_device, "--llm_parser_device", legacy_device])

    if dataset == "gsm_symbolic":
        cmd.extend(["--start_symbol", "<<", "--end_symbol", ">>"])
        if args.gsm_split_file:
            cmd.extend(
                [
                    "--gsm-split-file",
                    str(args.gsm_split_file),
                    "--gsm-split-name",
                    args.gsm_split_name,
                ]
            )
    elif dataset == "spider":
        cmd.extend(["--start_symbol", "<<", "--end_symbol", ">>"])
    if dataset == "spider" and args.spider_split_file:
        cmd.extend(
            [
                "--spider-split-file",
                str(args.spider_split_file),
                "--spider-split-name",
                args.spider_split_name,
            ]
        )

    repo_syncode_root = repo_root / "synthesis" / "evaluate" / "syncode"
    repo_syncode_pkg = repo_syncode_root / "syncode"
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    prefix_paths = [str(repo_root), str(repo_syncode_root), str(repo_syncode_pkg)]
    env["PYTHONPATH"] = os.pathsep.join(
        prefix_paths + ([existing_pythonpath] if existing_pythonpath else [])
    )

    crane_run_started = time.perf_counter()
    try:
        subprocess.run(cmd, cwd=crane_src_dir, check=True, env=env)
    finally:
        if tier2_grammar_path is not None and tier2_grammar_path.is_file():
            tier2_grammar_path.unlink(missing_ok=True)

    from synthesis.evaluate.benchmarks.registry import get_logic
    from synthesis.evaluate.evaluator import Evaluator

    logic = get_logic(dataset)
    eval_runtime = Evaluator(
        dataset_name=dataset,
        model_name=args.eval_model,
        backend=args.eval_backend,
        device=args.device,
        sample_size=args.eval_sample_size,
        max_steps=args.eval_max_steps,
        step_token_budget=args.eval_step_token_budget,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
        gsm_split_file=args.gsm_split_file if dataset == "gsm_symbolic" else None,
        gsm_split_name=args.gsm_split_name,
        spider_split_file=args.spider_split_file if dataset == "spider" else None,
        spider_split_name=args.spider_split_name,
    )
    _configure_fixed_eval_runtime(eval_runtime, args, dataset)
    examples = logic.load_dataset_sample(eval_runtime)

    raw_rows = _annotate_legacy_rows_with_syntax(
        _load_latest_crane_results(crane_src_dir, dataset),
        args,
        dataset,
    )
    rows = _enrich_crane_baseline_rows(
        raw_rows,
        args=args,
        dataset=dataset,
        logic=logic,
        eval_runtime=eval_runtime,
        examples=examples,
    )
    adapter = "crane_legacy_main" if args.strategy == "crane" else "unconstrained_crane_main"
    _build_minimal_json(
        rows,
        args.output_json,
        dataset=dataset,
        run_wall_time_seconds=time.perf_counter() - crane_run_started,
        extra_metrics={"adapter": adapter},
        metadata=_baseline_run_metadata(args, dataset, adapter=adapter),
    )
    print(f"Saved baseline JSON: {args.output_json}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run legacy fixed strategy code and export minimal baseline JSON"
    )
    parser.add_argument(
        "--strategy",
        required=True,
        choices=["unconstrained", "gcd", "crane", "itergen", "rejection_sampling"],
    )
    parser.add_argument("--dataset", required=True, choices=["gsm", "gsm_symbolic", "spider"])
    parser.add_argument("--eval-model", required=True)
    parser.add_argument("--eval-sample-size", type=int, default=10)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--eval-backend", default="vllm")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval-max-steps", type=int, default=150)
    parser.add_argument("--eval-step-token-budget", type=int, default=1)
    parser.add_argument("--dafny-path", default="")
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument(
        "--vllm-tensor-parallel-size",
        type=int,
        default=None,
        help="vLLM tensor parallel size (default: VAS_MAX_CUDA_DEVICES; capped by that env)",
    )
    parser.add_argument("--gsm-split-file", type=str, default=None,
                        help="Optional GSM train/eval split manifest JSON")
    parser.add_argument("--gsm-split-name", type=str, choices=["train", "eval"], default="eval",
                        help="Which split from --gsm-split-file to use (default: eval)")
    parser.add_argument("--spider-split-file", type=str, default=None,
                        help="Optional Spider train/test split manifest JSON")
    parser.add_argument("--spider-split-name", type=str, choices=["train", "test", "eval"], default="eval",
                        help="Which split from --spider-split-file to use (default: eval)")
    parser.add_argument(
        "--rejection-search-steps",
        type=int,
        default=DEFAULT_REJECTION_SEARCH_STEPS,
        help=(
            "Max temperature-1 rejection-sampling decode attempts per example. "
            f"Default: {DEFAULT_REJECTION_SEARCH_STEPS}. Not tied to --eval-max-steps."
        ),
    )
    args = parser.parse_args()
    from synthesis.evaluate.benchmarks.common.model_utils import (
        configure_vllm_multiprocessing,
        resolve_vllm_tensor_parallel_size,
    )

    args.vllm_tensor_parallel_size = resolve_vllm_tensor_parallel_size(args.vllm_tensor_parallel_size)

    _ensure_repo_cache_env()
    if args.eval_backend == "vllm":
        configure_vllm_multiprocessing()

    dataset_normalized = _normalize_dataset(args.dataset)
    # CRANE main.py only supports gsm_symbolic and FOL — no Spider support
    # (no prompt_templates/spider.yaml, no generate_spider_with_itergen function).
    # For Spider, fall through to the legacy shared-evaluator adapters which DO
    # support Spider and were the source of the 05-18 baselines.
    crane_repo_datasets = {"gsm_symbolic"}
    crane_repo_strategies = {"unconstrained", "gcd", "crane", "itergen"}
    if (
        dataset_normalized in crane_repo_datasets
        and args.strategy in crane_repo_strategies
    ):
        from synthesis.evaluate.baselines.crane_repo_runner import (
            run_crane_repo_baseline,
        )
        raise SystemExit(run_crane_repo_baseline(args, dataset_normalized))

    if args.strategy == "gcd":
        raise SystemExit(run_gcd_legacy_adapter(args))
    if args.strategy == "itergen":
        raise SystemExit(run_itergen_legacy_adapter(args))    if args.strategy == "rejection_sampling":
        raise SystemExit(run_rejection_sampling_legacy_adapter(args))
    raise SystemExit(run_crane_legacy_adapter(args))


if __name__ == "__main__":
    main()
