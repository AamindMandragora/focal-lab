"""
Environment setup utilities for FOLIO evaluation.

Handles loading compiled CSD modules and setting up the Dafny environment
for constrained generation.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, Dict


def resolve_run_dir(run_dir: Path) -> Path:
    """
    Resolve a run directory path, handling 'latest' shortcut.
    
    If run_dir ends with 'latest' and doesn't exist as a directory,
    reads the actual path from 'latest_run.txt' in the parent directory.
    
    Args:
        run_dir: Path to the synthesis run directory (may be 'latest' shortcut)
        
    Returns:
        Resolved actual path to the run directory
    """
    if run_dir.name == "latest" and not run_dir.exists():
        latest_txt = run_dir.parent / "latest_run.txt"
        if latest_txt.exists():
            actual_path = Path(latest_txt.read_text().strip())
            if actual_path.exists():
                return actual_path
    return run_dir


def load_compiled_modules(run_dir: Path):
    """
    Load compiled CSD modules from a synthesis run directory.
    
    Args:
        run_dir: Path to the synthesis run directory
        
    Returns:
        Tuple of (_dafny, VerifiedDecoderAgent, GeneratedCSD) modules
        
    Raises:
        FileNotFoundError: If compiled modules are not found
    """
    # Resolve 'latest' shortcut if needed
    run_dir = resolve_run_dir(run_dir)
    
    module_dir = run_dir / "generated_csd"
    if not module_dir.exists():
        # Fallback to other possible directories
        for subdir in ["gsm_crane_csd", "folio_csd", "fol_csd"]:
            module_dir = run_dir / subdir
            if module_dir.exists():
                break
        else:
            # Check if GeneratedCSD.py is directly in run_dir
            if (run_dir / "GeneratedCSD.py").exists():
                module_dir = run_dir
            else:
                # Try to find any directory that contains GeneratedCSD.py
                found = list(run_dir.glob("*/GeneratedCSD.py"))
                if found:
                    module_dir = found[0].parent
                else:
                    raise FileNotFoundError(f"Compiled module directory not found in {run_dir}")
    
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))

    for module_name in ["GeneratedCSD", "VerifiedDecoderAgent", "module_", "System_", "_dafny"]:
        sys.modules.pop(module_name, None)

    _dafny = importlib.import_module("_dafny")
    VerifiedDecoderAgent = importlib.import_module("VerifiedDecoderAgent")
    GeneratedCSD = importlib.import_module("GeneratedCSD")

    return _dafny, VerifiedDecoderAgent, GeneratedCSD


def setup_dafny_environment(
    run_dir: Path,
    model_name: str,
    backend: str,
    device: str,
    grammar_file: Path,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    vllm_tensor_parallel_size: int | None = None,
    vllm_pipeline_parallel_size: int = 1,
    vllm_gpu_memory_utilization: float = 0.8,
    vllm_max_model_len: int = 4096,
    vllm_enforce_eager: bool = True,
) -> Dict[str, Any]:
    """
    Load model and setup Dafny environment once.
    Returns reusable objects for generation.

    Args:
        run_dir: Path to the synthesis run directory
        model_name: Model identifier
        backend: Runtime backend ("huggingface" or "vllm")
        device: Device to run on ("cuda" or "cpu")
        grammar_file: Path to grammar file
        load_in_4bit: Whether to load in 4-bit quantization
        load_in_8bit: Whether to load in 8-bit quantization
        vllm_tensor_parallel_size: Explicit tensor parallel size for vLLM
        vllm_pipeline_parallel_size: Explicit pipeline parallel size for vLLM
        vllm_gpu_memory_utilization: GPU memory fraction reserved by vLLM
        vllm_max_model_len: Max context length passed to vLLM
        vllm_enforce_eager: Disable cudagraph/compile in vLLM for stability

    Returns:
        Environment dict with:
        - "_dafny": Dafny runtime module
        - "VerifiedDecoderAgent": Dafny decoder agent module
        - "GeneratedCSD": Generated CSD module
        - "lm": Language model wrapper
        - "parser": Grammar parser
        - "tokenizer": Backend tokenizer
    """
    _dafny, VerifiedDecoderAgent, GeneratedCSD = load_compiled_modules(run_dir)

    from evaluations.common.model_utils import create_runtime_lm, load_runtime_tokenizer
    from evaluations.common.parser_utils import create_lark_dafny_parser

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

    # Create grammar parser
    grammar_text = grammar_file.read_text()
    # Use 'start' rule for FOL grammar (full FOL statements)
    LarkDafnyParser = create_lark_dafny_parser(grammar_text, VerifiedDecoderAgent, _dafny, start="start", tokenizer=tok)
    parser = LarkDafnyParser(lm._Tokens)

    return {
        "_dafny": _dafny,
        "VerifiedDecoderAgent": VerifiedDecoderAgent,
        "GeneratedCSD": GeneratedCSD,
        "lm": lm,
        "parser": parser,
        "tokenizer": tok,
    }


def verify_critical_tokens(tokenizer, verbose: bool = True) -> Dict[str, Any]:
    """No-op — critical token verification removed (was dataset-specific)."""
    return {"found": [], "missing": []}
