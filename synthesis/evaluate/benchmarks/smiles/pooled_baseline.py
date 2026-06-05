"""Legacy fixed-strategy SMILES baselines using the pooled native protocol."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Callable

from synthesis.evaluate.benchmarks.smiles.pooled_eval import (
    DEFAULT_SMILES_POOLED_SUCCESS_TARGET,
    SMILES_POOLED_MAX_NEW_TOKENS,
    SmilesPooledConfig,
    SmilesStopCriterion,
    finalize_pooled_smiles_metadata,
    pooled_smiles_extra_metrics,
    run_smiles_pooled_class,
    smiles_pooled_config_from_args,
)
from synthesis.evaluate.prompt_tiers import prompt_tier_for_strategy


def _smiles_baseline_row(
    *,
    example: dict[str, Any],
    prompt: str,
    output_text: str,
    eval_row: dict[str, Any],
    syntax_valid: bool,
    is_correct: bool,
    generation_seconds: float,
    token_count: int | None,
    grammar_ok: bool,
    class_name: str,
    extra: dict[str, Any],
) -> dict[str, Any]:
    from synthesis.evaluate.run_legacy_fixed_strategy import _legacy_adapter_baseline_row

    row = _legacy_adapter_baseline_row(
        dataset="smiles",
        example=example,
        prompt=prompt,
        raw_generated=output_text,
        extracted=eval_row.get("smiles") or "",
        correct=is_correct,
        syntax_valid=syntax_valid,
        generation_seconds=generation_seconds,
        num_tokens=token_count,
    )
    row["grammar_success"] = grammar_ok
    row["class_name"] = class_name
    row.update(extra)
    return row


def _cars_roots(repo_root: Path) -> Path:
    import os

    cars_root_override = os.environ.get("CARS_REPO_DIR")
    if cars_root_override:
        cars_root = Path(cars_root_override).expanduser().resolve()
    else:
        upstream_cars = Path(os.path.expanduser("~/cars")).resolve()
        cars_root = upstream_cars if upstream_cars.exists() else repo_root / "legacy" / "cars"
    if not cars_root.exists():
        raise RuntimeError(f"cars directory not found: {cars_root}")
    return cars_root


def run_smiles_pooled_legacy_adapter(args: Any) -> int:
    from synthesis.evaluate.benchmarks.registry import get_logic
    from synthesis.evaluate.benchmarks.smiles.dataset import normalize_smiles_classes
    from synthesis.evaluate.evaluator import Evaluator
    from synthesis.evaluate.run_legacy_fixed_strategy import (
        DEFAULT_CARS_SEARCH_STEPS,
        _baseline_run_metadata,
        _build_minimal_json,
        _cars_add_import_paths,
        _cars_encode_prompt,
        _cars_grammar_for_example,
        _cars_model_id,
        _cars_set_cached_grammar,
        _cars_tokens_to_text,
        _checkpoint_baseline_json,
        _configure_fixed_eval_runtime,
        _itergen_add_import_paths,
        _itergen_generate,
        _itergen_rebind_grammar,
        _legacy_adapter_baseline_row,
        _legacy_cuda_device_for_backend,
        _normalize_dataset,
        _tier1_grammar_for_example,
    )
    from synthesis.evaluate.rs import build_rs_session
    from synthesis.evaluate.syncode_run_session import SyncodeRunSession, release_cuda_cache

    run_started = time.perf_counter()
    dataset = _normalize_dataset(args.dataset)
    if dataset != "smiles":
        raise ValueError(f"pooled SMILES adapter requires dataset=smiles, got {dataset!r}")

    repo_root = Path(__file__).resolve().parents[4]
    strategy = str(args.strategy)
    adapter_by_strategy = {
        "cars": "cars_legacy_cars",
        "rs": "rs_syncode",
        "gcd": "gcd_syncode",
        "itergen": "itergen_legacy",
        "unconstrained": "unconstrained_smiles_syncode",
        "crane": "crane_smiles_syncode",
    }
    if strategy not in adapter_by_strategy:
        raise ValueError(f"Unsupported pooled SMILES strategy: {strategy}")

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
    )
    _configure_fixed_eval_runtime(eval_runtime, args, dataset)
    classes = normalize_smiles_classes(
        getattr(args, "smiles_classes", None) or eval_runtime.smiles_classes,
        require_non_empty=True,
    )

    stop_criterion = (
        SmilesStopCriterion.GRAMMAR_SUCCESS
        if strategy == "cars"
        else SmilesStopCriterion.NOVEL_VALID
    )
    config = smiles_pooled_config_from_args(
        args,
        stop_criterion=stop_criterion,
        prompt_tier=prompt_tier_for_strategy(strategy),
    )
    run_metadata = _baseline_run_metadata(args, dataset, adapter=adapter_by_strategy[strategy])
    if config.prompt_feedback.value == "static":
        run_metadata["prompt_style"] = "native_acrylates_txt_static"
    else:
        run_metadata["prompt_style"] = "native_acrylates_txt_dynamic_good_bad"
    run_metadata["prompt_tier"] = config.prompt_tier
    output_json = Path(args.output_json)
    rows: list[dict[str, Any]] = []
    device = _legacy_cuda_device_for_backend(args.device, args.eval_backend)

    def checkpoint(class_name: str) -> Callable[[], None]:
        def _checkpoint() -> None:
            _checkpoint_baseline_json(
                rows,
                output_json,
                dataset=dataset,
                run_started=run_started,
                extra_metrics={
                    **pooled_smiles_extra_metrics(rows, adapter=adapter_by_strategy[strategy]),
                    "class_name": class_name,
                },
                metadata=run_metadata,
            )

        return _checkpoint

    print(
        f"SMILES pooled {strategy}: up to {config.max_attempts} attempts/class, "
        f"stop after {config.success_target} "
        f"{'grammar successes' if stop_criterion == SmilesStopCriterion.GRAMMAR_SUCCESS else 'novel valid'}, "
        f"max_new_tokens={config.max_new_tokens}, "
        f"prompt={'static' if config.prompt_feedback.value == 'static' else 'dynamic good/bad'}",
        flush=True,
    )

    try:
        if strategy == "cars":
            import torch
            from cars.lib import ConstrainedModel

            cars_root = _cars_roots(repo_root)
            _cars_add_import_paths(cars_root)
            cars_model = ConstrainedModel(_cars_model_id(args.eval_model), None, torch_dtype=torch.bfloat16)
            grammar_cache: dict[str, Any] = {}

            for class_name in classes:
                task_example = {"class_name": class_name}
                grammar_text = _cars_grammar_for_example(repo_root, dataset, task_example)
                _cars_set_cached_grammar(cars_model, grammar_text, grammar_cache)
                cars_model.reset_sampling(learn_level=3, constrain_first=True)

                def generate_attempt(
                    prompt: str,
                    example: dict[str, Any],
                    attempt_idx: int,
                    *,
                    _cars_model: Any = cars_model,
                ) -> tuple[str, int | None, bool, dict[str, Any]]:
                    prompt_ids = _cars_encode_prompt(_cars_model, prompt)
                    try:
                        current_ids, _scores, _raw = _cars_model._generate(
                            prompt_ids,
                            max_new_tokens=config.max_new_tokens,
                        )
                        token_ids = [int(token_id) for token_id in current_ids[0]]
                        tokens = [_cars_model.tokenizer.decode(token_id) for token_id in token_ids]
                        return _cars_tokens_to_text(tokens), len(token_ids), True, {}
                    except ValueError as exc:
                        token_ids = exc.args[0] if exc.args else []
                        if isinstance(token_ids, (list, tuple)):
                            token_ids = [int(token_id) for token_id in token_ids]
                            tokens = [_cars_model.tokenizer.decode(token_id) for token_id in token_ids]
                            return _cars_tokens_to_text(tokens), len(token_ids), False, {}
                        return "", 0, False, {}

                run_smiles_pooled_class(
                    class_name=class_name,
                    config=config,
                    generate_attempt=generate_attempt,
                    rows=rows,
                    make_row=_smiles_baseline_row,
                    checkpoint=checkpoint(class_name),
                    log_prefix=f"CARS SMILES",
                )

        elif strategy in {"rs", "gcd", "unconstrained", "crane"}:
            if strategy == "rs":
                session = build_rs_session(
                    args.eval_model,
                    device=device,
                    max_new_tokens=config.max_new_tokens,
                )
                if session.mode == "original":
                    session.ensure_ready()
            elif strategy == "unconstrained":
                session = SyncodeRunSession(
                    args.eval_model,
                    device=device,
                    mode="original",
                    parse_output_only=True,
                    log_level=0,
                    max_new_tokens=config.max_new_tokens,
                    do_sample=False,
                    num_return_sequences=1,
                    opp=False,
                )
                session.ensure_ready()
            else:
                mode = "grammar_strict" if strategy == "gcd" else "grammar_mask"
                gen_kwargs: dict[str, Any] = {
                    "max_new_tokens": config.max_new_tokens,
                    "do_sample": strategy == "crane",
                    "num_return_sequences": 1,
                    "opp": False,
                }
                if strategy == "crane":
                    gen_kwargs["temperature"] = 1.0
                session = SyncodeRunSession(
                    args.eval_model,
                    device=device,
                    mode=mode,
                    parse_output_only=True,
                    log_level=0,
                    **gen_kwargs,
                )

            for class_name in classes:
                example_stub = {"class_name": class_name}

                def generate_attempt(
                    prompt: str,
                    example: dict[str, Any],
                    attempt_idx: int,
                    *,
                    _session: SyncodeRunSession = session,
                    _class_name: str = class_name,
                ) -> tuple[str, int | None, bool, dict[str, Any]]:
                    if strategy not in {"rs", "unconstrained"}:
                        grammar_text = _tier1_grammar_for_example(repo_root, dataset, example)
                        _session.apply_grammar(grammar_text)
                    batch = _session.infer(prompt, stop_words=None)
                    raw = (batch[0] if batch else "") or ""
                    return raw.strip(), None, True, {}

                run_smiles_pooled_class(
                    class_name=class_name,
                    config=config,
                    generate_attempt=generate_attempt,
                    rows=rows,
                    make_row=_smiles_baseline_row,
                    checkpoint=checkpoint(class_name),
                    log_prefix=strategy.upper(),
                )
            if strategy in {"rs", "gcd", "unconstrained", "crane"}:
                session.close()
                release_cuda_cache()

        elif strategy == "itergen":
            itergen_root = repo_root / "legacy" / "itergen"
            if not itergen_root.exists():
                raise RuntimeError(f"Legacy itergen directory not found: {itergen_root}")
            _itergen_add_import_paths(itergen_root)
            from itergen.main import IterGen

            iter_gen_session: Any = None
            grammar_cache: dict[str, Any] = {}
            _session_ceiling = min(16384, max(2048, config.max_new_tokens + 4096))

            for class_name in classes:
                from synthesis.evaluate.benchmarks.smiles.dataset import get_smiles_task

                task = get_smiles_task(class_name)
                grammar_text = _tier1_grammar_for_example(repo_root, dataset, task)
                if iter_gen_session is None:
                    iter_gen_session = IterGen(
                        grammar=grammar_text,
                        model_id=args.eval_model,
                        device=device,
                        parse_output_only=True,
                        quantize=False,
                        max_tokens=_session_ceiling,
                        do_sample=False,
                        max_new_tokens=config.max_new_tokens,
                        num_return_sequences=1,
                    )
                    iter_gen_session._vas_bound_grammar_text = grammar_text
                    grammar_cache[grammar_text] = (
                        iter_gen_session.grammar,
                        iter_gen_session.dfa_mask_store,
                        iter_gen_session.inc_parsers,
                        iter_gen_session._ignore_whitespace,
                    )
                else:
                    _itergen_rebind_grammar(
                        iter_gen_session,
                        grammar_text,
                        grammar_cache=grammar_cache,
                        dfa_mode="grammar_strict",
                    )

                def generate_attempt(
                    prompt: str,
                    example: dict[str, Any],
                    attempt_idx: int,
                    *,
                    _iter_gen: Any = iter_gen_session,
                ) -> tuple[str, int | None, bool, dict[str, Any]]:
                    raw = _itergen_generate(_iter_gen, prompt)
                    return (raw or "").strip(), None, True, {}

                run_smiles_pooled_class(
                    class_name=class_name,
                    config=config,
                    generate_attempt=generate_attempt,
                    rows=rows,
                    make_row=_smiles_baseline_row,
                    checkpoint=checkpoint(class_name),
                    log_prefix="ITERGEN SMILES",
                )
            release_cuda_cache()
        else:
            raise ValueError(f"Unhandled SMILES strategy: {strategy}")
    finally:
        pass

    if not rows:
        raise RuntimeError("Pooled SMILES baseline produced no rows")

    _build_minimal_json(
        rows,
        output_json,
        dataset=dataset,
        run_wall_time_seconds=time.perf_counter() - run_started,
        extra_metrics=pooled_smiles_extra_metrics(rows, adapter=adapter_by_strategy[strategy]),
        metadata=finalize_pooled_smiles_metadata(
            run_metadata,
            prompt_feedback=config.prompt_feedback,
        ),
    )
    print(f"Saved baseline JSON: {output_json}")
    return 0
