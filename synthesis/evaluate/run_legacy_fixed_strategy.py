"""Run fixed baseline strategies using legacy repository code paths."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import subprocess
import sys
from pathlib import Path
from typing import Any


_MAX_PROMPT_CHARS = 50000  # ~12.5K tokens; leaves room for generation within 16384-token context
_MAX_SUFFIX_CHARS = 45000


def _truncate_prompt(prompt: str, base_prompt: str) -> str:
    """Keep *prompt* within ``_MAX_PROMPT_CHARS`` by dropping the oldest appended molecules."""
    if len(prompt) <= _MAX_PROMPT_CHARS:
        return prompt
    suffix = prompt[len(base_prompt):]
    lines = suffix.split("\n")
    while len(base_prompt) + len("\n".join(lines)) > _MAX_PROMPT_CHARS and len(lines) > 1:
        lines.pop(0)
    return base_prompt + "\n".join(lines)


def _cap_suffix(suffix: str) -> str:
    """Drop the oldest appended molecules if suffix exceeds ``_MAX_SUFFIX_CHARS``."""
    if len(suffix) <= _MAX_SUFFIX_CHARS:
        return suffix
    lines = suffix.split("\n")
    while len("\n".join(lines)) > _MAX_SUFFIX_CHARS and len(lines) > 1:
        lines.pop(0)
    return "\n".join(lines)


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
    if dataset == "gsm_symbolic":
        text = (
            example.get("question_parsed")
            or example.get("original_question")
            or example.get("question")
            or example.get("prompt")
        )
        return str(text or fallback)
    return str(example.get("question") or example.get("prompt") or fallback)


def _legacy_benchmark_prompt(logic: Any, evaluator: Any, example: dict[str, Any], profile: str) -> str:
    """User-message text for legacy fixed strategies (not used by metadecode).

    profile:
      - ``expression_only``: one delimited answer; used by IterGen and GCD.
      - ``chain_of_thought``: explicit reasoning then answer; used by CRANE adaptive SMILES.
      - ``evaluator_default``: ``logic.format_prompt`` (optional Spider reasoning; historical CARS).
    """
    if profile == "evaluator_default":
        return logic.format_prompt(evaluator, example)
    if profile == "expression_only":
        return logic.format_prompt_expression_only(evaluator, example)
    if profile == "chain_of_thought":
        cot = getattr(logic, "format_prompt_chain_of_thought", None)
        if callable(cot):
            return cot(evaluator, example)
        return logic.format_prompt(evaluator, example)
    raise ValueError(f"Unknown legacy prompt profile: {profile}")


def _configure_fixed_eval_runtime(eval_runtime: Any, args: argparse.Namespace, dataset: str) -> None:
    if dataset == "gsm_symbolic":
        repo_root = Path(__file__).resolve().parents[2]
        env_gsm = os.environ.get("CRANE_GSM_SYMBOLIC_DIR")
        eval_runtime.gsm_source_dir = (
            Path(env_gsm).expanduser()
            if env_gsm
            else repo_root / "legacy" / "CRANE" / "src" / "gsm_symbolic"
        )
        eval_runtime.gsm_split_file = args.gsm_split_file
        eval_runtime.gsm_split_name = args.gsm_split_name
    if dataset == "spider":
        eval_runtime.spider_split_file = args.spider_split_file
        eval_runtime.spider_split_name = args.spider_split_name


def _crane_grammar_name(dataset: str) -> str:
    if dataset == "gsm_symbolic":
        return "gsm"
    if dataset == "spider":
        return "sql"
    raise ValueError(f"CRANE adapter currently supports gsm_symbolic/spider, got {dataset}")


def _mode_for_strategy(strategy: str) -> tuple[str, bool]:
    if strategy == "unconstrained":
        # Match CRANE GSM/SQL protocol: always request chain-of-thought from the subprocess model.
        return "original", True
    if strategy == "crane":
        return "adaptive", True
    raise ValueError(f"Unsupported CRANE-backed strategy: {strategy}")


def _compose_baseline_answer_row(question: str, generated: str, row: dict[str, Any]) -> dict[str, Any]:
    entry: dict[str, Any] = {"question": question, "generated_answer": generated}
    if row.get("num_tokens") is not None:
        entry["num_tokens"] = int(row["num_tokens"])
    if row.get("generation_seconds") is not None:
        entry["generation_seconds"] = round(float(row["generation_seconds"]), 6)
    return entry


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
    run_wall_time_seconds: float | None = None,
) -> None:
    if not rows:
        payload = {
            "accuracy": 0.0,
            "syntax_rate": 0.0,
            "metrics": _aggregate_run_metrics([], run_wall_time_seconds=run_wall_time_seconds),
            "answers": [],
        }
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(payload, indent=2) + "\n")
        return

    correct_vals: list[float] = []
    syntax_vals: list[float] = []
    answers: list[dict[str, Any]] = []

    syntax_keys = [
        "is_syntax_valid",
        "syntax_valid",
        "grammar_valid",
        "out_parse_success",
    ]

    for row in rows:
        if isinstance(row.get("correct"), bool):
            correct_vals.append(1.0 if row["correct"] else 0.0)

        syntax_value = None
        for key in syntax_keys:
            if key in row and isinstance(row[key], bool):
                syntax_value = 1.0 if row[key] else 0.0
                break
        if syntax_value is None:
            syntax_value = 0.0
        syntax_vals.append(syntax_value)

        question = str(row.get("question") or row.get("prompt") or "")
        generated = row.get("llm_response")
        if generated is None:
            generated = row.get("response")
        if generated is None:
            generated = row.get("pred")
        if generated is None:
            generated = ""
        answers.append(_compose_baseline_answer_row(question, str(generated), row))

    payload = {
        "accuracy": sum(correct_vals) / max(1, len(correct_vals)),
        "syntax_rate": sum(syntax_vals) / max(1, len(syntax_vals)),
        "metrics": _aggregate_run_metrics(rows, run_wall_time_seconds=run_wall_time_seconds),
        "answers": answers,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2) + "\n")


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
                numeric_vars = sorted(
                    set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", gold_answer))
                    - {"int", "ToInt", "z3_floor_div"}
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
        scored_output = (
            eval_runtime._truncate_gsm_output(output_text)
            if dataset == "gsm_symbolic"
            else output_text
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


def _cars_model_id(eval_model: str) -> str:
    """Return the HuggingFace model ID to pass to CARS's run_task.py."""
    return eval_model


def _load_latest_cars_results(cars_root_dir: Path, class_name: str) -> list[dict[str, Any]]:
    runs_root = cars_root_dir / "runs_log"
    prefix = f"smiles_{class_name}"
    candidates = sorted(
        runs_root.rglob("*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        if prefix not in candidate.as_posix():
            continue
        try:
            payload = json.loads(candidate.read_text())
        except json.JSONDecodeError:
            continue
        steps = payload.get("steps")
        if isinstance(steps, list):
            return steps
    return []


def _load_cars_results_from_log_dir(log_dir: Path) -> list[dict[str, Any]]:
    candidates = sorted(log_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for candidate in candidates:
        try:
            payload = json.loads(candidate.read_text())
        except json.JSONDecodeError:
            continue
        steps = payload.get("steps")
        if isinstance(steps, list):
            return steps
    return []


def _extract_cars_log_dir(stdout: str, cars_root_dir: Path) -> Path | None:
    for line in stdout.splitlines():
        match = re.search(r"Saving results in folder\s+(.+)$", line.strip())
        if match:
            return cars_root_dir / match.group(1)
    return None


def _cars_add_import_paths(cars_root_dir: Path) -> None:
    candidate_str = str(cars_root_dir.resolve())
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)


def _cars_grammar_for_example(repo_root: Path, dataset: str, example: dict[str, Any]) -> str:
    if dataset == "gsm_symbolic":
        return (repo_root / "synthesis" / "evaluate" / "grammars" / "gsm.lark").read_text()
    if dataset == "spider":
        return (repo_root / "synthesis" / "evaluate" / "grammars" / "sql.lark").read_text()
    if dataset == "smiles":
        return str(example.get("grammar_text", ""))
    raise ValueError(f"Unsupported dataset for CARS adapter: {dataset}")


def _cars_tokens_to_text(tokens: list[Any]) -> str:
    eos_markers = {"<|eot_id|>", "<|im_end|>", "<|endoftext|>"}
    text_tokens: list[str] = []
    for token in tokens:
        token_str = str(token)
        if token_str in eos_markers:
            continue
        text_tokens.append(token_str)
    return "".join(text_tokens).strip()


def _cars_generate_text(cars_model: Any, prompt: str, max_new_tokens: int, max_attempts: int) -> str:
    formatted_prompt = cars_model._format_prompt(prompt)
    prompt_ids = cars_model.tokenizer.encode(
        formatted_prompt,
        return_tensors="pt",
        add_special_tokens=False,
    ).to(cars_model.model.device)
    cars_model.reset_sampling(learn_level=3, constrain_first=True)

    for _attempt in range(max(1, max_attempts)):
        try:
            current_ids, _current_scores, _current_raw_logprob = cars_model._generate(
                prompt_ids,
                max_new_tokens=max_new_tokens,
            )
        except ValueError:
            continue
        tokens = [cars_model.tokenizer.decode(token_id) for token_id in current_ids[0]]
        return _cars_tokens_to_text(tokens)
    return ""


def _cars_normalize_gsm_symbolic_output(raw: str) -> str:
    """Wrap bare GSM expressions so benchmark scoring can see them.

    GSM-Symbolic ``extract_actual`` only reads ``<< ... >>`` spans. Legacy CARS often
    emits a grammar-valid expression body with no delimiters, which yields
    ``actual is None`` and zero accuracy despite plausible-looking output.
    """
    text = (raw or "").strip()
    if not text:
        return raw
    if re.findall(r"<<\s*([^<>]+?)\s*>>", text):
        return raw
    expr = text.splitlines()[0].strip()
    return f"<<{expr}>>" if expr else raw


def _cars_set_cached_grammar(
    cars_model: Any,
    grammar_text: str,
    grammar_cache: dict[str, Any],
) -> None:
    cached = grammar_cache.get(grammar_text)
    if cached is None:
        cars_model._set_grammar_constraint(grammar_text)
        grammar_cache[grammar_text] = cars_model.grammar_constraint
    else:
        cars_model.grammar_constraint = cached


def run_cars_legacy_adapter(args: argparse.Namespace) -> int:
    run_started = time.perf_counter()
    dataset = _normalize_dataset(args.dataset)

    model_id = _cars_model_id(args.eval_model)

    repo_root = Path(__file__).resolve().parents[2]
    cars_root = repo_root / "legacy" / "cars"
    if not cars_root.exists():
        raise RuntimeError(f"Legacy cars directory not found: {cars_root}")

    _cars_add_import_paths(cars_root)

    import torch
    from cars.lib import ConstrainedModel
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
    )
    _configure_fixed_eval_runtime(eval_runtime, args, dataset)

    examples = logic.load_dataset_sample(eval_runtime)
    cars_model = ConstrainedModel(model_id, None, torch_dtype=torch.bfloat16)

    rows: list[dict[str, Any]] = []
    smiles_prompt_suffix: dict[str, str] = {}
    grammar_cache: dict[str, Any] = {}
    max_attempts = max(20, min(2000, int(args.eval_max_steps) * 2))
    max_new_tokens = max(32, int(args.eval_max_steps))

    for example in examples:
        grammar_text = _cars_grammar_for_example(repo_root, dataset, example)
        _cars_set_cached_grammar(cars_model, grammar_text, grammar_cache)

        if dataset == "smiles":
            cls = str(example.get("class_name", ""))
            example["prompt"] = example["prompt"].rstrip() + smiles_prompt_suffix.get(cls, "")

        prompt = logic.format_prompt(eval_runtime, example)
        gen_started = time.perf_counter()
        output_text = _cars_generate_text(
            cars_model,
            prompt,
            max_new_tokens=max_new_tokens,
            max_attempts=max_attempts,
        )
        gen_seconds = time.perf_counter() - gen_started
        if dataset == "gsm_symbolic":
            output_text = _cars_normalize_gsm_symbolic_output(output_text)
        scored_output = eval_runtime._truncate_gsm_output(output_text) if dataset == "gsm_symbolic" else output_text
        expected = logic.expected_answer(eval_runtime, example)
        actual, _answer_source, aux = logic.extract_actual(eval_runtime, scored_output, example)
        is_correct = bool(logic.is_correct(eval_runtime, actual, expected, example, aux, scored_output))

        syntax_valid, _segments = eval_runtime._check_syntax_validity(scored_output, example=example)
        if dataset == "spider":
            syntax_valid = bool(actual and re.search(r"\bselect\b", actual, flags=re.IGNORECASE))
        if dataset == "smiles":
            syntax_valid = bool(aux and aux.get("syntax_valid"))
            if syntax_valid and actual:
                cls = str(example.get("class_name", ""))
                smiles_prompt_suffix[cls] = _cap_suffix(smiles_prompt_suffix.get(cls, "") + f" {actual}\nMolecule:")

        question = _baseline_row_question(dataset, example, expected)
        rows.append(
            {
                "question": question,
                "llm_response": output_text,
                "correct": bool(is_correct),
                "syntax_valid": bool(syntax_valid),
                "generation_seconds": gen_seconds,
            }
        )

    if not rows:
        raise RuntimeError("CARS produced no rows; refusing to write an empty baseline JSON")

    _build_minimal_json(
        rows,
        args.output_json,
        run_wall_time_seconds=time.perf_counter() - run_started,
    )
    print(f"Saved baseline JSON: {args.output_json}")
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

    from syncode.infer import Syncode
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
    )
    _configure_fixed_eval_runtime(eval_runtime, args, dataset)
    examples = logic.load_dataset_sample(eval_runtime)

    device = "cuda" if args.device in {"auto", "cuda"} else args.device
    base_gsm_grammar = (repo_root / "synthesis" / "evaluate" / "grammars" / "gsm.lark").read_text()
    gsm_allowed_variables: list[str] = []
    if dataset == "gsm_symbolic":
        from synthesis.evaluate.benchmarks.gsm_symbolic.grammar import (
            build_dynamic_grammar,
            build_numeric_only_grammar,
            extract_variables_from_mapping,
        )

        variable_names: set[str] = set()
        for ex in examples:
            vt = ex.get("variable_types") or {}
            if isinstance(vt, dict):
                variable_names.update(extract_variables_from_mapping(vt))
        gsm_allowed_variables = sorted(variable_names)
        if gsm_allowed_variables:
            base_gsm_grammar = build_dynamic_grammar(base_gsm_grammar, gsm_allowed_variables)
        else:
            base_gsm_grammar = build_numeric_only_grammar(base_gsm_grammar)
        # The prompt already ends with "<<"; constrain the expression body plus closing marker.
        base_gsm_grammar = base_gsm_grammar.replace(
            'syncode: "<<" start ">>"',
            'syncode: start ">>"',
            1,
        )

    def _grammar_for_example(example: dict[str, Any]) -> str:
        if dataset == "gsm_symbolic":
            return base_gsm_grammar
        if dataset == "spider":
            return (repo_root / "synthesis" / "evaluate" / "grammars" / "sql.lark").read_text()
        if dataset == "smiles":
            return str(example.get("grammar_text", ""))
        raise ValueError(f"Unsupported dataset for GCD adapter: {dataset}")

    def _gcd_max_new_tokens() -> int:
        if dataset == "gsm_symbolic":
            return min(96, max(32, int(args.eval_max_steps)))
        if dataset == "smiles":
            return min(256, max(64, int(args.eval_max_steps)))
        return max(32, int(args.eval_max_steps))

    def _gcd_prompt(prompt: str) -> str:
        if dataset == "gsm_symbolic":
            return prompt.rstrip() + "<<"
        return prompt

    def _gcd_output(completion: str, example: dict[str, Any]) -> str:
        if dataset != "gsm_symbolic":
            return completion
        expr = completion.strip().splitlines()[0].strip()
        if expr.startswith("<<"):
            wrapped = expr if ">>" in expr else f"{expr}>>"
            expr = re.findall(r"<<\s*([^<>]*?)\s*>>", wrapped)
            expr = expr[-1] if expr else ""
        if expr.endswith(">>"):
            expr = expr[:-2].strip()
        if not expr:
            return ""

        for end in range(len(expr), 0, -1):
            candidate = expr[:end].strip()
            if not candidate:
                continue
            wrapped = f"<<{candidate}>>"
            all_valid, segments = eval_runtime._check_syntax_validity(
                wrapped,
                example=example,
            )
            if logic.example_syntax_pass(all_valid, segments, False, None):
                return wrapped
        return f"<<{expr}>>"

    syncode_cache: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    smiles_prompt_suffix: dict[str, str] = {}

    for example in examples:
        grammar_text = _grammar_for_example(example)
        cache_key = f"{dataset}:{hash(grammar_text)}"
        if cache_key not in syncode_cache:
            syncode_cache[cache_key] = Syncode(
                model=args.eval_model,
                mode="grammar_strict",
                quantize=False,
                device=device,
                grammar=grammar_text,
                parse_output_only=True,
                log_level=0,
                max_new_tokens=_gcd_max_new_tokens(),
                do_sample=False,
                num_return_sequences=1,
                opp=False,
            )
        sc = syncode_cache[cache_key]

        if dataset == "smiles":
            cls = str(example.get("class_name", ""))
            example["prompt"] = example["prompt"].rstrip() + smiles_prompt_suffix.get(cls, "")

        prompt = _legacy_benchmark_prompt(logic, eval_runtime, example, "expression_only")
        gen_started = time.perf_counter()
        completions = sc.infer(_gcd_prompt(prompt), stop_words=[">>"] if dataset == "gsm_symbolic" else None)
        gen_seconds = time.perf_counter() - gen_started
        output_text = _gcd_output(completions[0] if completions else "", example)
        scored_output = eval_runtime._truncate_gsm_output(output_text) if dataset == "gsm_symbolic" else output_text
        expected = logic.expected_answer(eval_runtime, example)
        actual, _answer_source, aux = logic.extract_actual(eval_runtime, scored_output, example)
        is_correct = bool(logic.is_correct(eval_runtime, actual, expected, example, aux, scored_output))

        syntax_valid, _segments = eval_runtime._check_syntax_validity(scored_output, example=example)
        if dataset == "spider":
            syntax_valid = bool(actual and re.search(r"\bselect\b", actual, flags=re.IGNORECASE))
        if dataset == "smiles":
            syntax_valid = bool(aux and aux.get("syntax_valid"))
            if syntax_valid and actual:
                cls = str(example.get("class_name", ""))
                smiles_prompt_suffix[cls] = _cap_suffix(smiles_prompt_suffix.get(cls, "") + f" {actual}\nMolecule:")

        question = _baseline_row_question(dataset, example, expected)
        rows.append(
            {
                "question": question,
                "llm_response": output_text,
                "correct": bool(is_correct),
                "syntax_valid": bool(syntax_valid),
                "generation_seconds": gen_seconds,
            }
        )

    _build_minimal_json(
        rows,
        args.output_json,
        run_wall_time_seconds=time.perf_counter() - run_started,
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


def run_itergen_legacy_adapter(args: argparse.Namespace) -> int:
    prev_recursion = sys.getrecursionlimit()
    sys.setrecursionlimit(max(prev_recursion, 50_000))
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
    )
    _configure_fixed_eval_runtime(eval_runtime, args, dataset)
    examples = logic.load_dataset_sample(eval_runtime)

    device = "cuda" if args.device in {"auto", "cuda"} else args.device

    def _grammar_for_example(example: dict[str, Any]) -> str:
        if dataset == "gsm_symbolic":
            return (repo_root / "synthesis" / "evaluate" / "grammars" / "gsm.lark").read_text()
        if dataset == "spider":
            return (repo_root / "synthesis" / "evaluate" / "grammars" / "sql.lark").read_text()
        if dataset == "smiles":
            return str(example.get("grammar_text", ""))
        raise ValueError(f"Unsupported dataset for itergen adapter: {dataset}")

    itergen_cache: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    smiles_prompt_suffix: dict[str, str] = {}

    for example in examples:
        grammar_text = _grammar_for_example(example)
        cache_key = f"{dataset}:{hash(grammar_text)}"
        if cache_key not in itergen_cache:
            itergen_cache[cache_key] = IterGen(
                grammar=grammar_text,
                model_id=args.eval_model,
                device=device,
                parse_output_only=True,
                quantize=False,
                max_tokens=max(256, int(args.eval_max_steps) + 64),
                do_sample=False,
                max_new_tokens=max(32, int(args.eval_max_steps)),
                num_return_sequences=1,
                stop_strings=["\n\n"],
            )
        iter_gen = itergen_cache[cache_key]

        if dataset == "smiles":
            cls = str(example.get("class_name", ""))
            example["prompt"] = example["prompt"].rstrip() + smiles_prompt_suffix.get(cls, "")

        prompt = _legacy_benchmark_prompt(logic, eval_runtime, example, "expression_only")
        gen_started = time.perf_counter()
        output_text = _itergen_generate(iter_gen, prompt)
        gen_seconds = time.perf_counter() - gen_started
        scored_output = eval_runtime._truncate_gsm_output(output_text) if dataset == "gsm_symbolic" else output_text
        expected = logic.expected_answer(eval_runtime, example)
        actual, _answer_source, aux = logic.extract_actual(eval_runtime, scored_output, example)
        is_correct = bool(logic.is_correct(eval_runtime, actual, expected, example, aux, scored_output))

        syntax_valid, _segments = eval_runtime._check_syntax_validity(scored_output, example=example)
        if dataset == "spider":
            syntax_valid = bool(actual and re.search(r"\bselect\b", actual, flags=re.IGNORECASE))
        if dataset == "smiles":
            syntax_valid = bool(aux and aux.get("syntax_valid"))
            if syntax_valid and actual:
                cls = str(example.get("class_name", ""))
                smiles_prompt_suffix[cls] = _cap_suffix(smiles_prompt_suffix.get(cls, "") + f" {actual}\nMolecule:")

        question = _baseline_row_question(dataset, example, expected)
        rows.append(
            {
                "question": question,
                "llm_response": output_text,
                "correct": bool(is_correct),
                "syntax_valid": bool(syntax_valid),
                "generation_seconds": gen_seconds,
            }
        )

    _build_minimal_json(
        rows,
        args.output_json,
        run_wall_time_seconds=time.perf_counter() - run_started,
    )
    print(f"Saved baseline JSON: {args.output_json}")
    return 0


def run_unconstrained_smiles_adapter(args: argparse.Namespace) -> int:
    """Unconstrained SMILES generation: generate without grammar masking, then score.

    Each sample appends all previously generated valid molecules to the prompt
    so the model is nudged to produce novel strings (matching the CARS paper
    protocol).  Temperature 0.8 gives diversity; greedy would repeat the same
    molecule every time.
    """
    from synthesis.evaluate.benchmarks.smiles.dataset import SMILES_CLASSES, get_smiles_task
    from synthesis.evaluate.benchmarks.smiles.metrics import (
        clean_smiles_output,
        evaluate_smiles_output,
    )

    run_started = time.perf_counter()
    device = "cuda" if args.device in {"auto", "cuda"} else args.device
    n_per_class = max(1, args.eval_sample_size)
    uc_smiles_cot_prefix = (
        "For each molecule requested below, give brief step-by-step reasoning about the "
        "constraints, then write the SMILES string.\n\n"
    )

    if args.eval_backend == "vllm":
        from vllm import LLM, SamplingParams

        llm = LLM(
            model=args.eval_model,
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            max_model_len=16384,
            trust_remote_code=True,
        )
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        tokenizer = AutoTokenizer.from_pretrained(args.eval_model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.eval_model, device_map=device, torch_dtype=torch.float16, trust_remote_code=True
        )

    rows: list[dict[str, Any]] = []
    seen_per_class: dict[str, set[str]] = {}

    for class_name in SMILES_CLASSES:
        task = get_smiles_task(class_name)
        base_prompt = task["prompt"]
        grammar_text = str(task["grammar_text"])
        prompt_exemplars = list(task.get("prompt_exemplars", []))
        seen_per_class[class_name] = set(prompt_exemplars)
        seed_prompt = uc_smiles_cot_prefix + base_prompt
        running_prompt = seed_prompt

        for _i in range(n_per_class):
            running_prompt = _truncate_prompt(running_prompt, seed_prompt)
            if args.eval_backend == "vllm":
                from vllm import SamplingParams as _SP

                gen_started = time.perf_counter()
                sp = _SP(max_tokens=args.eval_max_steps, temperature=0.8, stop=["\n\n"])
                outputs = llm.generate([running_prompt], sp)
                gen_seconds = time.perf_counter() - gen_started
                completion = outputs[0].outputs[0]
                gen_text = completion.text
                num_toks = len(completion.token_ids) if getattr(completion, "token_ids", None) else None
            else:
                inputs = tokenizer(running_prompt, return_tensors="pt").to(model.device)
                gen_started = time.perf_counter()
                with torch.no_grad():
                    out_ids = model.generate(
                        **inputs, max_new_tokens=args.eval_max_steps,
                        do_sample=True, temperature=0.8,
                    )
                gen_seconds = time.perf_counter() - gen_started
                new_ids = out_ids[0][inputs["input_ids"].shape[1]:]
                num_toks = int(new_ids.numel())
                gen_text = tokenizer.decode(new_ids, skip_special_tokens=True)

            smiles_eval = evaluate_smiles_output(
                class_name=class_name,
                output=gen_text,
                grammar_text=grammar_text,
                prompt_exemplars=prompt_exemplars,
                require_rdkit=True,
            )
            cleaned = clean_smiles_output(gen_text)
            is_novel = bool(
                smiles_eval.get("syntax_valid")
                and cleaned
                and cleaned not in seen_per_class[class_name]
            )
            if cleaned and smiles_eval.get("syntax_valid"):
                seen_per_class[class_name].add(cleaned)
                running_prompt = running_prompt.rstrip() + f" {cleaned}\nMolecule: "

            row_out: dict[str, Any] = {
                "question": class_name,
                "llm_response": gen_text,
                "correct": bool(
                    is_novel and smiles_eval.get("class_membership")
                ),
                "syntax_valid": bool(smiles_eval.get("syntax_valid", False)),
                "generation_seconds": gen_seconds,
            }
            if num_toks is not None:
                row_out["num_tokens"] = num_toks
            rows.append(row_out)

    _build_minimal_json(
        rows,
        args.output_json,
        run_wall_time_seconds=time.perf_counter() - run_started,
    )
    print(f"Saved baseline JSON: {args.output_json}")
    return 0


def run_unconstrained_spider_adapter(args: argparse.Namespace) -> int:
    """Unconstrained Spider baseline without legacy CRANE ``main.py``.

    CRANE's ``PARSE_MAP`` only registers ``gsm_symbolic`` and ``fol``, so
    ``--dataset spider`` crashes with ``KeyError``. This path uses the same
    chain-of-thought Spider prompt as other legacy adapters and scores via
    ``benchmarks/sql_spider`` execution logic.
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
    )
    _configure_fixed_eval_runtime(eval_runtime, args, "spider")
    examples = logic.load_dataset_sample(eval_runtime)

    device = "cuda" if args.device in {"auto", "cuda"} else args.device

    if args.eval_backend == "vllm":
        from vllm import LLM

        llm = LLM(
            model=args.eval_model,
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            max_model_len=16384,
            trust_remote_code=True,
        )
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        tokenizer = AutoTokenizer.from_pretrained(args.eval_model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.eval_model,
            device_map=device,
            dtype=torch.float16,
            trust_remote_code=True,
        )

    rows: list[dict[str, Any]] = []
    max_new = max(32, int(args.eval_max_steps))

    for example in examples:
        prompt = _legacy_benchmark_prompt(logic, eval_runtime, example, "chain_of_thought")
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

        output_text = prompt + suffix
        scored_output = output_text
        expected = logic.expected_answer(eval_runtime, example)
        actual, _answer_source, aux = logic.extract_actual(eval_runtime, scored_output, example)
        is_correct = bool(logic.is_correct(eval_runtime, actual, expected, example, aux, scored_output))
        syntax_valid = bool(actual and re.search(r"\bselect\b", actual, flags=re.IGNORECASE))

        question = _baseline_row_question("spider", example, expected)
        row_out: dict[str, Any] = {
            "question": question,
            "llm_response": output_text,
            "correct": bool(is_correct),
            "syntax_valid": bool(syntax_valid),
            "generation_seconds": gen_seconds,
        }
        if num_toks is not None:
            row_out["num_tokens"] = num_toks
        rows.append(row_out)

    _build_minimal_json(
        rows,
        args.output_json,
        run_wall_time_seconds=time.perf_counter() - run_started,
    )
    print(f"Saved baseline JSON: {args.output_json}")
    return 0


def _crane_via_adaptive_syncode(args: argparse.Namespace, dataset: str) -> int:
    """Run CRANE-style adaptive constrained decoding via vendored AdaptiveSynCode.

    Used for benchmarks where the legacy CRANE codebase lacks grammar support
    (e.g. SMILES). AdaptiveSynCode implements the same << >> switching logic.
    """
    run_started = time.perf_counter()
    import sys as _sys

    repo_root = Path(__file__).resolve().parents[2]
    syncode_root = repo_root / "synthesis" / "evaluate" / "syncode"
    syncode_pkg = syncode_root / "syncode"
    for p in [str(syncode_root), str(syncode_pkg)]:
        if p not in _sys.path:
            _sys.path.insert(0, p)

    from syncode.infer import AdaptiveSynCode
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
    )
    _configure_fixed_eval_runtime(eval_runtime, args, dataset)
    examples = logic.load_dataset_sample(eval_runtime)

    device = "cuda" if args.device in {"auto", "cuda"} else args.device

    def _grammar_for_example(example: dict[str, Any]) -> str:
        if dataset == "smiles":
            return str(example.get("grammar_text", ""))
        if dataset == "spider":
            return (repo_root / "synthesis" / "evaluate" / "grammars" / "sql.lark").read_text()
        raise ValueError(f"Unsupported dataset for AdaptiveSynCode adapter: {dataset}")

    syncode_cache: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    smiles_prompt_suffix: dict[str, str] = {}

    for example in examples:
        grammar_text = _grammar_for_example(example)
        cache_key = f"{dataset}:{hash(grammar_text)}"
        if cache_key not in syncode_cache:
            syncode_cache[cache_key] = AdaptiveSynCode(
                model=args.eval_model,
                mode="grammar_strict",
                quantize=False,
                device=device,
                grammar=grammar_text,
                parse_output_only=True,
                log_level=0,
                start_symbol="<<",
                end_symbol=">>",
                max_new_tokens=max(32, int(args.eval_max_steps)),
                do_sample=False,
                num_return_sequences=1,
            )
        sc = syncode_cache[cache_key]

        if dataset == "smiles":
            cls = str(example.get("class_name", ""))
            example["prompt"] = example["prompt"].rstrip() + smiles_prompt_suffix.get(cls, "")

        prompt = _legacy_benchmark_prompt(logic, eval_runtime, example, "chain_of_thought")
        gen_started = time.perf_counter()
        completions = sc.infer(prompt)
        gen_seconds = time.perf_counter() - gen_started
        output_text = completions[0] if completions else ""
        scored_output = output_text
        expected = logic.expected_answer(eval_runtime, example)
        actual, _answer_source, aux = logic.extract_actual(eval_runtime, scored_output, example)
        is_correct = bool(logic.is_correct(eval_runtime, actual, expected, example, aux, scored_output))

        syntax_valid, _segments = eval_runtime._check_syntax_validity(scored_output, example=example)
        if dataset == "smiles":
            syntax_valid = bool(aux and aux.get("syntax_valid"))
            if syntax_valid and actual:
                cls = str(example.get("class_name", ""))
                smiles_prompt_suffix[cls] = _cap_suffix(smiles_prompt_suffix.get(cls, "") + f" {actual}\nMolecule:")

        question = _baseline_row_question(dataset, example, expected)
        rows.append(
            {
                "question": question,
                "llm_response": output_text,
                "correct": bool(is_correct),
                "syntax_valid": bool(syntax_valid),
                "generation_seconds": gen_seconds,
            }
        )

    _build_minimal_json(
        rows,
        args.output_json,
        run_wall_time_seconds=time.perf_counter() - run_started,
    )
    print(f"Saved baseline JSON: {args.output_json}")
    return 0


def run_crane_legacy_adapter(args: argparse.Namespace) -> int:
    dataset = _normalize_dataset(args.dataset)

    if dataset == "smiles" and args.strategy == "unconstrained":
        return run_unconstrained_smiles_adapter(args)

    if dataset == "spider" and args.strategy == "unconstrained":
        return run_unconstrained_spider_adapter(args)

    if dataset == "smiles" and args.strategy == "crane":
        return _crane_via_adaptive_syncode(args, dataset)

    mode, do_cot = _mode_for_strategy(args.strategy)
    grammar = _crane_grammar_name(dataset)

    repo_root = Path(__file__).resolve().parents[2]
    crane_src_dir = repo_root / "legacy" / "CRANE" / "src"
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
        "1",
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
        grammar if mode != "original" else "text",
        "--out_grammar",
        grammar if mode != "original" else "text",
    ]
    if do_cot:
        cmd.extend(["--do_cot", "True"])

    if dataset == "gsm_symbolic":
        cmd.extend(["--start_symbol", "<<", "--end_symbol", ">>"])

    repo_syncode_root = repo_root / "synthesis" / "evaluate" / "syncode"
    repo_syncode_pkg = repo_syncode_root / "syncode"
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    syncode_paths = [str(repo_syncode_root), str(repo_syncode_pkg)]
    env["PYTHONPATH"] = os.pathsep.join(
        syncode_paths + ([existing_pythonpath] if existing_pythonpath else [])
    )

    crane_run_started = time.perf_counter()
    subprocess.run(cmd, cwd=crane_src_dir, check=True, env=env)

    rows = _annotate_legacy_rows_with_syntax(
        _load_latest_crane_results(crane_src_dir, dataset),
        args,
        dataset,
    )
    _build_minimal_json(
        rows,
        args.output_json,
        run_wall_time_seconds=time.perf_counter() - crane_run_started,
    )
    print(f"Saved baseline JSON: {args.output_json}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run legacy fixed strategy code and export minimal baseline JSON"
    )
    parser.add_argument("--strategy", required=True, choices=["unconstrained", "gcd", "crane", "itergen", "cars"])
    parser.add_argument("--dataset", required=True, choices=["gsm", "gsm_symbolic", "spider", "smiles"])
    parser.add_argument("--eval-model", required=True)
    parser.add_argument("--eval-sample-size", type=int, default=10)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--eval-backend", default="vllm")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval-max-steps", type=int, default=150)
    parser.add_argument("--eval-step-token-budget", type=int, default=1)
    parser.add_argument("--dafny-path", default="")
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--gsm-split-file", type=str, default=None,
                        help="Optional GSM train/eval split manifest JSON")
    parser.add_argument("--gsm-split-name", type=str, choices=["train", "eval"], default="eval",
                        help="Which split from --gsm-split-file to use (default: eval)")
    parser.add_argument("--spider-split-file", type=str, default=None,
                        help="Optional Spider train/test split manifest JSON")
    parser.add_argument("--spider-split-name", type=str, choices=["train", "test", "eval"], default="eval",
                        help="Which split from --spider-split-file to use (default: eval)")
    args = parser.parse_args()

    _ensure_repo_cache_env()

    if args.strategy == "gcd":
        raise SystemExit(run_gcd_legacy_adapter(args))
    if args.strategy == "itergen":
        raise SystemExit(run_itergen_legacy_adapter(args))
    if args.strategy == "cars":
        raise SystemExit(run_cars_legacy_adapter(args))
    raise SystemExit(run_crane_legacy_adapter(args))


if __name__ == "__main__":
    main()
