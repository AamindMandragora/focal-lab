"""Environment setup for SMILES evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from synthesis.evaluate.benchmarks.gsm_symbolic.environment import (
    _attach_helper_fastpath,
    _attach_helper_trace,
    _attach_lm_trace,
    load_compiled_modules,
    resolve_run_dir,
    verify_critical_tokens,
)


def setup_dafny_environment(
    run_dir: Path,
    model_name: str,
    backend: str = "huggingface",
    device: str = "cuda",
    grammar_file: Path | None = None,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    vllm_tensor_parallel_size: int | None = None,
    vllm_pipeline_parallel_size: int = 1,
    vllm_gpu_memory_utilization: float = 0.8,
    vllm_max_model_len: int = 4096,
    vllm_enforce_eager: bool = True,
) -> Dict[str, Any]:
    """Load compiled CSD, model runtime, and a SMILES grammar parser."""
    if grammar_file is None:
        raise ValueError("SMILES evaluation requires a class-specific grammar_file")

    _dafny, VerifiedDecoderAgent, GeneratedCSD = load_compiled_modules(run_dir)
    _attach_helper_fastpath(VerifiedDecoderAgent)
    trace_state: Dict[str, Any] = {"events": []}
    _attach_helper_trace(VerifiedDecoderAgent, trace_state)

    from synthesis.evaluate.benchmarks.common.model_utils import create_runtime_lm, load_runtime_tokenizer
    from synthesis.evaluate.benchmarks.common.parser_utils import create_lark_dafny_parser

    tok = load_runtime_tokenizer(model_name, backend=backend)
    lm = create_runtime_lm(
        model_name=model_name,
        backend=backend,
        device=device,
        VerifiedDecoderAgent=VerifiedDecoderAgent,
        _dafny=_dafny,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        vllm_tensor_parallel_size=vllm_tensor_parallel_size,
        vllm_pipeline_parallel_size=vllm_pipeline_parallel_size,
        vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
        vllm_max_model_len=vllm_max_model_len,
        vllm_enforce_eager=vllm_enforce_eager,
    )
    _attach_lm_trace(lm, trace_state)

    grammar_text = grammar_file.read_text()
    LarkDafnyParser = create_lark_dafny_parser(
        grammar_text,
        VerifiedDecoderAgent,
        _dafny,
        start="start",
        tokenizer=tok,
    )
    parser = LarkDafnyParser(lm._Tokens)

    return {
        "_dafny": _dafny,
        "VerifiedDecoderAgent": VerifiedDecoderAgent,
        "GeneratedCSD": GeneratedCSD,
        "lm": lm,
        "parser": parser,
        "tokenizer": tok,
        "csd_trace": trace_state,
    }


__all__ = [
    "load_compiled_modules",
    "resolve_run_dir",
    "setup_dafny_environment",
    "verify_critical_tokens",
]
