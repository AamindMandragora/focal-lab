"""Run fixed baseline strategies using legacy repository code paths."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


def _normalize_dataset(dataset: str) -> str:
    return "gsm_symbolic" if dataset == "gsm" else dataset


def _crane_grammar_name(dataset: str) -> str:
    if dataset == "gsm_symbolic":
        return "gsm"
    if dataset == "spider":
        return "sql"
    raise ValueError(f"CRANE adapter currently supports gsm_symbolic/spider, got {dataset}")


def _mode_for_strategy(strategy: str) -> tuple[str, bool]:
    if strategy == "unconstrained":
        return "original", False
    if strategy == "crane":
        return "adaptive", True
    raise ValueError(f"Unsupported CRANE-backed strategy: {strategy}")


def _write_empty_baseline(output_json: Path) -> None:
    payload = {"accuracy": 0.0, "syntax_rate": 0.0, "answers": []}
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2) + "\n")


def _build_minimal_json(rows: list[dict[str, Any]], output_json: Path) -> None:
    if not rows:
        payload = {"accuracy": 0.0, "syntax_rate": 0.0, "answers": []}
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(payload, indent=2) + "\n")
        return

    correct_vals: list[float] = []
    syntax_vals: list[float] = []
    answers: list[dict[str, str]] = []

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
            syntax_value = 1.0
        syntax_vals.append(syntax_value)

        question = str(row.get("question") or row.get("prompt") or "")
        generated = row.get("llm_response")
        if generated is None:
            generated = row.get("response")
        if generated is None:
            generated = row.get("pred")
        if generated is None:
            generated = ""
        answers.append({"question": question, "generated_answer": str(generated)})

    payload = {
        "accuracy": sum(correct_vals) / max(1, len(correct_vals)),
        "syntax_rate": sum(syntax_vals) / max(1, len(syntax_vals)),
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


def _cars_model_num(eval_model: str) -> str | None:
    model = eval_model.lower()
    if "qwen" in model and "14b" in model:
        return "3"
    if "qwen" in model and "7b" in model:
        return "2"
    if "llama" in model and "8b" in model:
        return "1"
    return None


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


def _cars_tokens_to_text(tokens: list[Any]) -> str:
    eos_markers = {"<|eot_id|>", "<|im_end|>", "<|endoftext|>"}
    text_tokens: list[str] = []
    for token in tokens:
        token_str = str(token)
        if token_str in eos_markers:
            continue
        text_tokens.append(token_str)
    return "".join(text_tokens).strip()


def run_cars_legacy_adapter(args: argparse.Namespace) -> int:
    dataset = _normalize_dataset(args.dataset)
    if dataset != "smiles":
        print(
            f"[warn] Legacy CARS adapter currently supports smiles only (got {dataset}). "
            f"Writing empty baseline JSON to {args.output_json}."
        )
        _write_empty_baseline(args.output_json)
        return 0

    model_num = _cars_model_num(args.eval_model)
    if model_num is None:
        print(
            f"[warn] Legacy CARS supports model presets for 7B/14B Qwen or Llama-3.1-8B. "
            f"Received eval model '{args.eval_model}'. Writing empty baseline JSON to {args.output_json}."
        )
        _write_empty_baseline(args.output_json)
        return 0

    repo_root = Path(__file__).resolve().parents[2]
    cars_root = repo_root / "legacy" / "cars"
    if not cars_root.exists():
        raise RuntimeError(f"Legacy cars directory not found: {cars_root}")

    from synthesis.evaluate.benchmarks.smiles.dataset import SMILES_CLASSES, get_smiles_task
    from synthesis.evaluate.benchmarks.smiles.metrics import evaluate_smiles_output

    rows: list[dict[str, Any]] = []
    for class_name in SMILES_CLASSES:
        grammar_path = cars_root / "datasets" / "smiles" / f"{class_name}.lark"
        prompt_path = cars_root / "datasets" / "smiles" / f"{class_name}.txt"
        if not grammar_path.exists() or not prompt_path.exists():
            print(
                f"[warn] Missing legacy CARS assets for class={class_name}; skipping."
            )
            continue

        cmd = [
            "python",
            "run_task.py",
            str(grammar_path.relative_to(cars_root)),
            str(prompt_path.relative_to(cars_root)),
            "cars",
            model_num,
        ]
        subprocess.run(cmd, cwd=cars_root, check=True)

        task = get_smiles_task(class_name)
        class_steps = _load_latest_cars_results(cars_root, class_name)
        if not class_steps:
            continue

        for step in class_steps[: args.eval_sample_size]:
            tokens = step.get("tokens")
            if not isinstance(tokens, list):
                continue
            generated = _cars_tokens_to_text(tokens)
            smiles_eval = evaluate_smiles_output(
                class_name=class_name,
                output=generated,
                grammar_text=str(task["grammar_text"]),
                prompt_exemplars=task.get("prompt_exemplars", []),
                require_rdkit=True,
            )
            rows.append(
                {
                    "question": class_name,
                    "llm_response": generated,
                    "correct": bool(smiles_eval.get("valid_class_membership", False)),
                    "syntax_valid": bool(smiles_eval.get("syntax_valid", False)),
                }
            )

    _build_minimal_json(rows, args.output_json)
    print(f"Saved baseline JSON: {args.output_json}")
    return 0


def run_gcd_legacy_adapter(args: argparse.Namespace) -> int:
    """GCD baseline via vendored SynCode (grammar_strict mode = pure hard-mask constrained decoding)."""
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
    examples = logic.load_dataset_sample(eval_runtime)

    device = "cuda" if args.device in {"auto", "cuda"} else args.device

    def _grammar_for_example(example: dict[str, Any]) -> str:
        if dataset == "gsm_symbolic":
            return (repo_root / "synthesis" / "evaluate" / "grammars" / "gsm.lark").read_text()
        if dataset == "spider":
            return (repo_root / "synthesis" / "evaluate" / "grammars" / "sql.lark").read_text()
        if dataset == "smiles":
            return str(example.get("grammar_text", ""))
        raise ValueError(f"Unsupported dataset for GCD adapter: {dataset}")

    syncode_cache: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []

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
                max_new_tokens=max(32, int(args.eval_max_steps)),
                do_sample=False,
                num_return_sequences=1,
            )
        sc = syncode_cache[cache_key]
        prompt = logic.format_prompt(eval_runtime, example)
        completions = sc.infer(prompt)
        output_text = completions[0] if completions else ""
        scored_output = eval_runtime._truncate_gsm_output(output_text) if dataset == "gsm_symbolic" else output_text
        expected = logic.expected_answer(eval_runtime, example)
        actual, _answer_source, aux = logic.extract_actual(eval_runtime, scored_output, example)
        is_correct = bool(logic.is_correct(eval_runtime, actual, expected, example, aux, scored_output))

        syntax_valid, _segments = eval_runtime._check_syntax_validity(scored_output, example=example)
        if dataset == "spider":
            syntax_valid = bool(actual and re.search(r"\bselect\b", actual, flags=re.IGNORECASE))
        if dataset == "smiles":
            syntax_valid = bool(aux and aux.get("syntax_valid"))

        question = str(example.get("question") or example.get("prompt") or expected)
        rows.append(
            {
                "question": question,
                "llm_response": output_text,
                "correct": bool(is_correct),
                "syntax_valid": bool(syntax_valid),
            }
        )

    _build_minimal_json(rows, args.output_json)
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
        prompt = logic.format_prompt(eval_runtime, example)
        output_text = _itergen_generate(iter_gen, prompt)
        scored_output = eval_runtime._truncate_gsm_output(output_text) if dataset == "gsm_symbolic" else output_text
        expected = logic.expected_answer(eval_runtime, example)
        actual, _answer_source, aux = logic.extract_actual(eval_runtime, scored_output, example)
        is_correct = bool(logic.is_correct(eval_runtime, actual, expected, example, aux, scored_output))

        syntax_valid, _segments = eval_runtime._check_syntax_validity(scored_output, example=example)
        if dataset == "spider":
            syntax_valid = bool(actual and re.search(r"\bselect\b", actual, flags=re.IGNORECASE))
        if dataset == "smiles":
            syntax_valid = bool(aux and aux.get("syntax_valid"))

        question = str(example.get("question") or example.get("prompt") or expected)
        rows.append(
            {
                "question": question,
                "llm_response": output_text,
                "correct": bool(is_correct),
                "syntax_valid": bool(syntax_valid),
            }
        )

    _build_minimal_json(rows, args.output_json)
    print(f"Saved baseline JSON: {args.output_json}")
    return 0


def run_unconstrained_smiles_adapter(args: argparse.Namespace) -> int:
    """Unconstrained SMILES generation: generate without grammar masking, then score."""
    from synthesis.evaluate.benchmarks.smiles.dataset import load_smiles, SMILES_CLASSES
    from synthesis.evaluate.benchmarks.smiles.metrics import evaluate_smiles_output

    examples = load_smiles(classes=list(SMILES_CLASSES), samples_per_class=max(1, args.eval_sample_size))

    device = "cuda" if args.device in {"auto", "cuda"} else args.device

    if args.eval_backend == "vllm":
        from vllm import LLM, SamplingParams

        llm = LLM(
            model=args.eval_model,
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            max_model_len=4096,
            trust_remote_code=True,
        )
        sampling_params = SamplingParams(
            max_tokens=args.eval_max_steps,
            temperature=0.0,
            stop=["\n\n"],
        )

        prompts = [ex.get("prompt", "") for ex in examples]
        outputs = llm.generate(prompts, sampling_params)
        generated_texts = [out.outputs[0].text for out in outputs]
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        tokenizer = AutoTokenizer.from_pretrained(args.eval_model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.eval_model, device_map=device, torch_dtype=torch.float16, trust_remote_code=True
        )
        generated_texts = []
        for ex in examples:
            prompt = ex.get("prompt", "")
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out_ids = model.generate(**inputs, max_new_tokens=args.eval_max_steps, do_sample=False)
            generated_texts.append(tokenizer.decode(out_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))

    rows: list[dict[str, Any]] = []
    for example, gen_text in zip(examples, generated_texts):
        class_name = example.get("class_name", "smiles")
        grammar_text = example.get("grammar_text", "")
        prompt_exemplars = example.get("prompt_exemplars", [])
        smiles_eval = evaluate_smiles_output(
            class_name=class_name,
            output=gen_text,
            grammar_text=grammar_text,
            prompt_exemplars=prompt_exemplars,
            require_rdkit=True,
        )
        rows.append(
            {
                "question": class_name,
                "llm_response": gen_text,
                "correct": bool(smiles_eval.get("unique_valid_candidate", False)),
                "syntax_valid": bool(smiles_eval.get("syntax_valid", False)),
            }
        )

    _build_minimal_json(rows, args.output_json)
    print(f"Saved baseline JSON: {args.output_json}")
    return 0


def _crane_via_adaptive_syncode(args: argparse.Namespace, dataset: str) -> int:
    """Run CRANE-style adaptive constrained decoding via vendored AdaptiveSynCode.

    Used for benchmarks where the legacy CRANE codebase lacks grammar support
    (e.g. SMILES). AdaptiveSynCode implements the same << >> switching logic.
    """
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
        prompt = logic.format_prompt(eval_runtime, example)
        completions = sc.infer(prompt)
        output_text = completions[0] if completions else ""
        scored_output = output_text
        expected = logic.expected_answer(eval_runtime, example)
        actual, _answer_source, aux = logic.extract_actual(eval_runtime, scored_output, example)
        is_correct = bool(logic.is_correct(eval_runtime, actual, expected, example, aux, scored_output))

        syntax_valid, _segments = eval_runtime._check_syntax_validity(scored_output, example=example)
        if dataset == "smiles":
            syntax_valid = bool(aux and aux.get("syntax_valid"))

        question = str(example.get("question") or example.get("prompt") or expected)
        rows.append(
            {
                "question": question,
                "llm_response": output_text,
                "correct": bool(is_correct),
                "syntax_valid": bool(syntax_valid),
            }
        )

    _build_minimal_json(rows, args.output_json)
    print(f"Saved baseline JSON: {args.output_json}")
    return 0


def run_crane_legacy_adapter(args: argparse.Namespace) -> int:
    dataset = _normalize_dataset(args.dataset)

    if dataset == "smiles" and args.strategy == "unconstrained":
        return run_unconstrained_smiles_adapter(args)

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

    subprocess.run(cmd, cwd=crane_src_dir, check=True)

    rows = _load_latest_crane_results(crane_src_dir, dataset)
    _build_minimal_json(rows, args.output_json)
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
    args = parser.parse_args()

    if args.strategy == "gcd":
        raise SystemExit(run_gcd_legacy_adapter(args))
    if args.strategy == "itergen":
        raise SystemExit(run_itergen_legacy_adapter(args))
    if args.strategy == "cars":
        raise SystemExit(run_cars_legacy_adapter(args))
    raise SystemExit(run_crane_legacy_adapter(args))


if __name__ == "__main__":
    main()
