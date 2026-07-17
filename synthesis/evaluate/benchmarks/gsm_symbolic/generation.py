"""
Generation methods for evaluation.

Provides CSD (Constrained Decoding Strategy) generation by delegating
entirely to the Dafny-verified CSD strategy.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional, Tuple, Union


from synthesis.evaluate.benchmarks.common.dafny_tokens import dafny_seq_to_str


def _enforce_max_steps(result_tokens: List[str], max_steps: int) -> None:
    if len(result_tokens) > max_steps:
        raise RuntimeError(
            f"CSD exceeded max_steps: generated {len(result_tokens)} tokens > {max_steps}"
        )


def run_crane_csd(
    env: dict,
    prompt_text: Union[str, List[dict]],
    max_steps: int,
    grammar_file: Path,
    debug_delimiters: bool = False,
    dynamic_parser=None,
    start_inside_constrained: bool = False,
    step_token_budget: int = 1,
    valid_tokens: Optional[List[str]] = None,
    valid_token_groups: Optional[List[List[str]]] = None,
    max_seconds: Optional[float] = None,
    completion_mode: bool = False,
    early_stop_on_answer: bool = False,
) -> Tuple[str, int, float, List[Tuple[str, bool]], List[dict]]:
    """
    Run generation using the Dafny-verified CSD strategy.

    Delegates entirely to the compiled Dafny strategy — no dataset-specific
    orchestration is performed here.

    Args:
        env: Environment dict with Dafny modules and model
        prompt_text: The prompt text (set as lm.instruction_text)
        max_steps: Maximum generation steps
        grammar_file: Path to grammar file for post-hoc segment validation
        debug_delimiters: Whether to print debug output
        dynamic_parser: Optional per-question parser
        start_inside_constrained: Begin with an internal constrained chunk
            already active. This is useful for tasks like Spider where the
            answer is parser-governed from the first token but chunk boundaries
            should not be serialized as visible delimiters.
        completion_mode: When True, set lm.instruction_text to the raw
            prompt_text string with no chat template applied. Required for base
            (non-instruction-tuned) completion models, which must see the prompt
            as a raw continuation rather than ChatML-wrapped.
        early_stop_on_answer: When True, stop generation as soon as the output
            contains a finished final-answer span ('final answer' followed by a
            complete <<...>> span), mirroring CRANE's answer stopping. The
            output-so-far is returned and scored through the normal path.

    Returns:
        Tuple of (output_text, token_count, time_seconds, constrained_segments)
    """
    _dafny = env["_dafny"]
    GeneratedCSD = env["GeneratedCSD"]
    lm = env["lm"]
    parser = dynamic_parser if dynamic_parser is not None else env["parser"]

    if completion_mode:
        # Base (non-instruction-tuned) completion models: feed the prompt as a
        # raw continuation with NO chat template. The model continues directly
        # from prompt_text (full_prompt = instruction_text + generated prefix),
        # matching how IterGen drives the base Coder model. Chat scaffolding is
        # intentionally left unset so no ChatML markers leak into the prompt.
        if not isinstance(prompt_text, str):
            raise ValueError("completion_mode requires prompt_text to be a string")
        lm.instruction_text = prompt_text
        if hasattr(lm, "ResetTaskGuidance"):
            lm.ResetTaskGuidance()
    else:
        if isinstance(prompt_text, list):
            chat_messages = prompt_text
        else:
            chat_messages = [{"role": "user", "content": prompt_text}]
        try:
            lm.instruction_text = lm.tokenizer.apply_chat_template(
                chat_messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            lm.instruction_text = lm.tokenizer.apply_chat_template(
                chat_messages, tokenize=False, add_generation_prompt=True
            )
        # Register chat_messages on the LM so AppendTaskGuidance (if the CSD
        # calls it) can re-template with guidance injected INSIDE the last user
        # message — instead of appending it after the assistant generation
        # marker (which previously landed guidance in the model's output space
        # and crashed accuracy by 18-22pp on this cell).
        if hasattr(lm, "set_chat_messages"):
            lm.set_chat_messages(chat_messages)
        if hasattr(lm, "ResetTaskGuidance"):
            lm.ResetTaskGuidance()
    start_time = time.time()
    runtime_deadline = None
    if max_seconds is not None:
        runtime_deadline = time.monotonic() + max_seconds
    if hasattr(lm, "SetRuntimeDeadline"):
        lm.SetRuntimeDeadline(runtime_deadline)
    if early_stop_on_answer and hasattr(lm, "SetAnswerEarlyStop"):
        lm.SetAnswerEarlyStop(True)

    eos_token_str = lm.tokenizer.eos_token or "<|endoftext|>"
    eos_token_dafny = _dafny.Seq(eos_token_str)
    generated_prefix = _dafny.SeqWithoutIsStrInference([])
    current_constrained = _dafny.SeqWithoutIsStrInference([])

    trace_state = env.get("csd_trace")
    if isinstance(trace_state, dict):
        trace_state["events"] = []

    if valid_token_groups is not None:
        token_groups = valid_token_groups
        flat_tokens = [t for group in token_groups for t in group]
    else:
        flat_tokens = valid_tokens or []
        token_groups = [flat_tokens] if flat_tokens else []

    valid_tokens_dafny = _dafny.SeqWithoutIsStrInference(
        [_dafny.Seq(t) for t in flat_tokens]
    )
    valid_token_groups_dafny = _dafny.SeqWithoutIsStrInference(
        [
            _dafny.SeqWithoutIsStrInference([_dafny.Seq(t) for t in group])
            for group in token_groups
        ]
    )

    from synthesis.evaluate.benchmarks.common.model_utils import AnswerCompleteStop

    import inspect
    _sig = inspect.signature(GeneratedCSD.default__.MyCSDStrategy)
    _param_names = list(_sig.parameters.keys())
    _n_params = len(_param_names)
    answer_early_stopped = False
    result = None
    try:
        if "validTokenGroups" in _param_names:
            result = GeneratedCSD.default__.MyCSDStrategy(
                lm, parser, _dafny.SeqWithoutIsStrInference([]), generated_prefix,
                start_inside_constrained, current_constrained,
                max_steps, step_token_budget, valid_token_groups_dafny, eos_token_dafny,
            )
        elif "validTokens" in _param_names or _n_params >= 10:
            result = GeneratedCSD.default__.MyCSDStrategy(
                lm, parser, _dafny.SeqWithoutIsStrInference([]), generated_prefix,
                start_inside_constrained, current_constrained,
                max_steps, step_token_budget, valid_tokens_dafny, eos_token_dafny,
            )
        elif _n_params >= 9:
            result = GeneratedCSD.default__.MyCSDStrategy(
                lm, parser, _dafny.SeqWithoutIsStrInference([]), generated_prefix,
                start_inside_constrained, current_constrained,
                max_steps, step_token_budget, eos_token_dafny,
            )
        else:
            result = GeneratedCSD.default__.MyCSDStrategy(
                lm, parser, _dafny.SeqWithoutIsStrInference([]), generated_prefix,
                start_inside_constrained, current_constrained,
                max_steps, eos_token_dafny,
            )
    except AnswerCompleteStop:
        # Generation-complete signal, not a failure: the final answer span is
        # finished. The per-step hook stashed the tokens generated so far;
        # return them through the normal scoring path below.
        answer_early_stopped = True
    finally:
        if hasattr(lm, "ClearRuntimeDeadline"):
            lm.ClearRuntimeDeadline()

    final_inside_constrained = False
    final_current_constrained = _dafny.SeqWithoutIsStrInference([])

    if answer_early_stopped:
        result_tokens = list(getattr(lm, "_early_stop_tokens", None) or [])
        total_cost = len(result_tokens)
    else:
        if isinstance(result, tuple) and len(result) == 4:
            csd_output, final_inside_constrained, final_current_constrained, total_cost = result
        elif isinstance(result, tuple):
            csd_output, total_cost = result
        else:
            csd_output = result
            total_cost = 0
        result_tokens = [dafny_seq_to_str(t) for t in csd_output]

    if early_stop_on_answer and hasattr(lm, "SetAnswerEarlyStop"):
        # Clear AFTER harvesting the stash so the flag cannot leak into the
        # next example's generation.
        lm.SetAnswerEarlyStop(False)
    _enforce_max_steps(result_tokens, max_steps)
    output_text = "".join(result_tokens)
    execution_time = time.time() - start_time

    constrained_segments: List[Tuple[str, bool]] = []
    helper_trace = list(trace_state.get("events", [])) if isinstance(trace_state, dict) else []
    task_guidance = getattr(lm, "task_guidance", None)
    if task_guidance:
        helper_trace.append({"helper": "AppendTaskGuidance", "detail": task_guidance})

    final_chunk_tokens = [
        dafny_seq_to_str(final_current_constrained[i])
        for i in range(len(final_current_constrained))
    ]
    final_chunk = "".join(final_chunk_tokens)
    hidden_chunk_used = start_inside_constrained and (
        bool(final_chunk)
        or any(
            event.get("helper") in {"AppendConstrainedToken", "ConstrainedStep", "CloseConstrainedSpan"}
            for event in helper_trace
        )
    )
    if hidden_chunk_used:
        # Hidden constrained chunks are real parser-governed chunks even though
        # no << / >> boundary tokens are rendered into user-visible output.
        constrained_segments.append((final_chunk or output_text, True))

    if debug_delimiters:
        print(f"  [DEBUG] Generation finished in {execution_time:.2f}s. Cost: {total_cost}. Tokens: {len(result_tokens)}")

    # Per-example timing dump: what fraction of wall time went to Python callbacks
    # (LM + parser) vs Dafny-internal work. If callbacks sum << execution_time,
    # the bottleneck is inside the generated Dafny strategy (not our Python code).
    try:
        from synthesis.evaluate.benchmarks.common.model_utils import _TIMINGS, _print_timings_breakdown
        from synthesis.evaluate.benchmarks.common.parser_utils import _PARSER_TIMINGS, print_parser_timings

        lm_total = sum(t for t, _ in _TIMINGS.values())
        parser_total = sum(t for t, _ in _PARSER_TIMINGS.values())
        callbacks_total = lm_total + parser_total
        dafny_internal = max(execution_time - callbacks_total, 0.0)
        print(
            f"[STEP_BREAKDOWN] wall={execution_time:.2f}s  lm_callbacks={lm_total:.2f}s  "
            f"parser_callbacks={parser_total:.2f}s  dafny_internal={dafny_internal:.2f}s  "
            f"({100*dafny_internal/max(execution_time,1e-6):.1f}% in dafny/uninstrumented)",
            flush=True,
        )
        _print_timings_breakdown(header="end_of_example (cumulative)")
        print_parser_timings(header="end_of_example (cumulative)")
    except Exception as _dbg_err:
        print(f"[STEP_BREAKDOWN] error printing timings: {_dbg_err}", flush=True)

    return output_text, len(result_tokens), execution_time, constrained_segments, helper_trace
