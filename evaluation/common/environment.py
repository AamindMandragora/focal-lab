"""
Shared environment setup for CSD evaluation.

Provides resolve_run_dir, legacy compiled-module loading, token checks,
setup_dafny_environment, and setup_python_native_environment.
"""

from __future__ import annotations

import gc
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_VAS_DIR = _PROJECT_ROOT / "generation" / "csd"

try:
    import torch
except ImportError:
    torch = None

from evaluation.common.run_artifacts import find_compiled_module_dir, resolve_run_dir


def _allow_cpu_fallback() -> bool:
    return os.environ.get("CSD_EVAL_CPU_FALLBACK", "").strip().lower() in {"1", "true", "yes", "on"}


def load_compiled_modules(run_dir: Path):
    """
    Load legacy Dafny-built CSD modules from a synthesis run directory.

    Args:
        run_dir: Path to the synthesis run directory

    Returns:
        Tuple of (_dafny, VerifiedDecoderAgent, GeneratedCSD) modules

    Raises:
        FileNotFoundError: If legacy compiled modules are not found
    """
    run_dir = resolve_run_dir(run_dir)
    module_dir = find_compiled_module_dir(run_dir)

    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))

    import _dafny
    import VerifiedDecoderAgent
    import GeneratedCSD

    return _dafny, VerifiedDecoderAgent, GeneratedCSD


def verify_critical_tokens(tokenizer, verbose: bool = True) -> Dict[str, Any]:
    """No-op — critical token verification removed (was dataset-specific)."""
    return {"found": [], "missing": []}


def setup_dafny_environment(
    run_dir: Path,
    model_name: str,
    device: str,
    vocab_size: int | None,
    grammar_file: Path,
    start_rule: str = "start",
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    add_gsm_delimiter_tokens: bool = False,
    extra_token_strings: Optional[list[str]] = None,
) -> Dict[str, Any]:
    """
    Load model and setup Dafny environment once.
    Returns reusable objects for generation.

    Args:
        run_dir: Path to the synthesis run directory
        model_name: HuggingFace model identifier
        device: Device to run on ("cuda" or "cpu")
        vocab_size: Size of constrained vocabulary. `None` or <= 0 uses the full tokenizer.
        grammar_file: Path to grammar file
        start_rule: Grammar start rule for the selected benchmark
        load_in_4bit: Whether to load model in 4-bit quantization
        load_in_8bit: Whether to load model in 8-bit quantization

    Returns:
        Environment dict with _dafny, VerifiedDecoderAgent, GeneratedCSD, lm, parser, tokenizer
    """
    _dafny, VerifiedDecoderAgent, GeneratedCSD = load_compiled_modules(run_dir)

    from evaluation.common.model_utils import create_huggingface_lm
    from evaluation.common.parser_utils import create_lark_dafny_parser

    used_cpu_fallback = False
    try:
        lm = create_huggingface_lm(
            model_name,
            device,
            vocab_size,
            VerifiedDecoderAgent,
            _dafny,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            add_gsm_delimiter_tokens=add_gsm_delimiter_tokens,
            extra_token_strings=extra_token_strings,
        )
    except RuntimeError as e:
        if "out of memory" not in str(e).lower():
            raise
        if not device.startswith("cuda") or torch is None or not _allow_cpu_fallback():
            raise
        torch.cuda.empty_cache()
        print("  Evaluation: CUDA OOM, falling back to CPU for model load.")
        lm = create_huggingface_lm(
            model_name,
            "cpu",
            vocab_size,
            VerifiedDecoderAgent,
            _dafny,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            add_gsm_delimiter_tokens=add_gsm_delimiter_tokens,
            extra_token_strings=extra_token_strings,
        )
        used_cpu_fallback = True

    grammar_text = grammar_file.read_text()
    LarkDafnyParser = create_lark_dafny_parser(
        grammar_text, VerifiedDecoderAgent, _dafny, start=start_rule
    )
    parser = LarkDafnyParser(lm._Tokens)

    result = {
        "_dafny": _dafny,
        "VerifiedDecoderAgent": VerifiedDecoderAgent,
        "GeneratedCSD": GeneratedCSD,
        "lm": lm,
        "parser": parser,
        "tokenizer": lm.tokenizer,
    }
    if used_cpu_fallback:
        result["_eval_cpu_fallback"] = True
    return result


def setup_python_native_environment(
    python_source_path: Path,
    model_name: str,
    device: str,
    vocab_size: int | None,
    grammar_file: Path,
    start_rule: str = "start",
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    add_gsm_delimiter_tokens: bool = False,
    extra_token_strings: Optional[list[str]] = None,
) -> Dict[str, Any]:
    """
    Load model and set up a Python-native (non-Dafny) evaluation environment.

    Instead of loading Dafny-compiled modules, this imports VerifiedAgentSynthesis
    and the strategy Python file directly, using plain Python types throughout.
    No _dafny runtime is needed.

    Returns a dict with "strategy_module", "lm", "parser", "tokenizer", "_mode": "native".
    """
    # Make VerifiedAgentSynthesis importable
    vas_dir = str(_VAS_DIR)
    if vas_dir not in sys.path:
        sys.path.insert(0, vas_dir)
    import VerifiedAgentSynthesis as VAS

    # Load the generated strategy module
    strategy_dir = str(python_source_path.parent)
    if strategy_dir not in sys.path:
        sys.path.insert(0, strategy_dir)
    spec = importlib.util.spec_from_file_location("generated_csd_native", python_source_path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"Could not load strategy from {python_source_path}")
    strategy_module = importlib.util.module_from_spec(spec)
    sys.modules["generated_csd_native"] = strategy_module
    spec.loader.exec_module(strategy_module)

    from evaluation.common.model_utils import create_huggingface_lm_native
    from evaluation.common.parser_utils import create_lark_native_parser

    used_cpu_fallback = False
    try:
        lm = create_huggingface_lm_native(
            model_name,
            device,
            vocab_size,
            VAS,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            add_gsm_delimiter_tokens=add_gsm_delimiter_tokens,
            extra_token_strings=extra_token_strings,
        )
    except RuntimeError as e:
        if "out of memory" not in str(e).lower():
            raise
        if not device.startswith("cuda") or torch is None or not _allow_cpu_fallback():
            raise
        torch.cuda.empty_cache()
        print("  Evaluation (native): CUDA OOM, falling back to CPU for model load.")
        lm = create_huggingface_lm_native(
            model_name,
            "cpu",
            vocab_size,
            VAS,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            add_gsm_delimiter_tokens=add_gsm_delimiter_tokens,
            extra_token_strings=extra_token_strings,
        )
        used_cpu_fallback = True

    grammar_text = grammar_file.read_text()
    LarkNativeParser = create_lark_native_parser(grammar_text, VAS, start=start_rule)
    parser = LarkNativeParser(lm.Tokens)

    result = {
        "strategy_module": strategy_module,
        "VerifiedAgentSynthesis": VAS,
        "lm": lm,
        "parser": parser,
        "tokenizer": lm.tokenizer,
        "_mode": "native",
    }
    if used_cpu_fallback:
        result["_eval_cpu_fallback"] = True
    return result


def release_evaluation_environment(env: Dict[str, Any] | None) -> None:
    """Drop model references and release CUDA cache between synthesis attempts."""
    if not env:
        return
    env.clear()
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
