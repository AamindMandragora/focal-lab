"""
Strategy generator for CSD synthesis.

Uses either an OpenAI API model or a HuggingFace Transformers model to generate
Python strategy code for insertion into `generation/csd/GeneratedAgentTemplate.py`.
"""

import ast
import os
import re
import textwrap
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .prompts import (
    build_initial_prompt,
    build_verification_error_prompt,
    build_runtime_error_prompt,
    build_compilation_error_prompt,
    build_format_repair_prompt,
    build_evaluation_failure_prompt,
    build_structure_repair_prompt,
)
from .rationale import extract_proof_sketch, extract_rationale


def _hf_offline_enabled() -> bool:
    """True when HuggingFace loaders should stay strictly offline."""
    return any(os.environ.get(name, "").strip() in {"1", "true", "True"} for name in (
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
    ))


def _is_hf_connection_error(exc: Exception) -> bool:
    """Best-effort detection for transient offline/network lookup failures."""
    text = str(exc).lower()
    return any(marker in text for marker in (
        "failed to resolve",
        "name or service not known",
        "temporary failure in name resolution",
        "connection error",
        "maxretryerror",
        "httpsconnectionpool",
        "offline mode",
    ))


def _env_flag(name: str, default: bool) -> bool:
    """Read a boolean-ish environment flag."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _generator_load_in_4bit_enabled() -> bool:
    """Whether generator model loading should use 4-bit CUDA quantization."""
    return _env_flag("CSD_GENERATOR_LOAD_IN_4BIT", True)


def _is_openai_model_name(model_name: str) -> bool:
    """Best-effort classifier for OpenAI API model identifiers."""
    normalized = model_name.strip().lower()
    return normalized.startswith(("gpt-", "o1", "o3", "o4"))


def _auto_select_device() -> str:
    """Pick a usable accelerator for auto mode, or CPU when GPUs are too full."""
    if torch.cuda.is_available():
        default_min_free_gb = "8" if _generator_load_in_4bit_enabled() else "30"
        min_free_gb = float(os.environ.get("CSD_MIN_CUDA_FREE_GB", default_min_free_gb))
        min_free_bytes = int(min_free_gb * 1024**3)
        candidates: list[tuple[int, int]] = []
        for gpu_id in range(torch.cuda.device_count()):
            try:
                free_bytes, _total_bytes = torch.cuda.mem_get_info(gpu_id)
                candidates.append((int(free_bytes), gpu_id))
            except Exception:
                continue
        if candidates:
            free_bytes, gpu_id = max(candidates)
            if free_bytes >= min_free_bytes:
                device = f"cuda:{gpu_id}"
                free_gb = free_bytes / 1024**3
                print(f"Auto-selected {device} ({free_gb:.1f} GiB free).")
                return device
            free_gb = free_bytes / 1024**3
            print(
                "CUDA is available, but no GPU has enough free memory for "
                f"local generation/evaluation ({free_gb:.1f} GiB best, need {min_free_gb:.1f} GiB). "
                "Using CPU for generation."
            )
        else:
            print("CUDA is available, but free-memory probing failed. Using CPU for generation.")
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class StrategyGenerationError(RuntimeError):
    """Raised when the model cannot produce a structurally valid strategy."""


class StrategyGenerator:
    """
    Generates Python CSD strategies using an LLM.
    
    OpenAI model names are served through the OpenAI Responses API. Other model
    names are loaded locally via HuggingFace Transformers.
    """
    
    # Default model - can be overridden
    DEFAULT_MODEL = "gpt-5.4"
    
    TEMPLATE_PY_PATH = Path(__file__).resolve().parent / "csd" / "GeneratedAgentTemplate.py"
    TEMPLATE_DAFNY_PATH = Path(__file__).resolve().parent / "csd" / "GeneratedAgentTemplate.dfy"

    # Under this budget, Qwen often truncates before emitting a full rationale + loop body.
    MIN_STRATEGY_TOKENS = 192
    SEARCH_ATTEMPTS = 12
    DIAGNOSTIC_TEXT_LIMIT = 12_000
    ALLOWED_HELPER_METHODS = {
        "AdaptiveConstrainedStep",
        "AllValidNextTokensInLM",
        "AppendConstrainedStep",
        "AppendConstrainedOrRightDelimiterStep",
        "AppendConstrainedToken",
        "AppendForcedToken",
        "AppendLeftDelimiter",
        "AppendRightDelimiter",
        "AppendSoftConstrainedStep",
        "AppendTopKConstrainedStep",
        "AppendUnconstrainedAllowLeftDelimiterStep",
        "AppendUnconstrainedNudgeLeftDelimiterStep",
        "AppendUnconstrainedStep",
        "BiasForCompletion",
        "BiasLeftDelimiters",
        "BiasRightDelimiters",
        "CanConstrain",
        "Checkpoint",
        "CloseConstrainedSpan",
        "ConstrainedOrRightDelimiterStep",
        "ConstrainedStep",
        "ContainsLeftDelimiter",
        "ContainsRightDelimiter",
        "CountOccurrences",
        "EndsWithLeftDelimiter",
        "EndsWithRightDelimiter",
        "ForcedTokenStep",
        "GroupBoostedConstrainedStep",
        "HasBudget",
        "IntersectWithGrammar",
        "IsComplete",
        "IsDead",
        "IsLeftDelimiterToken",
        "IsRightDelimiterToken",
        "LastTokenBefore",
        "LongestValidSuffix",
        "MaskAllDelimiters",
        "MaskLeftDelimiters",
        "MaskRightDelimiters",
        "MinStepsToComplete",
        "OpenConstrainedSpan",
        "ParserDistanceToComplete",
        "PenalizedConstrainedStep",
        "RestoreCheckpoint",
        "RestoreIfDead",
        "SoftConstrainToGrammar",
        "SoftConstrainedStep",
        "TopKConstrainedStep",
        "TokensSinceLastDelimiter",
        "UnconstrainedAllowLeftDelimiterStep",
        "UnconstrainedBiasLeftDelimiterStep",
        "UnconstrainedNudgeLeftDelimiterStep",
        "UnconstrainedStep",
        "ValidContinuationCount",
    }
    ALLOWED_PARSER_METHODS = {
        "IsValidPrefix",
        "IsCompletePrefix",
        "IsDeadPrefix",
        "ValidNextToken",
        "ValidNextTokens",
        "EmptyPrefixIsValid",
        "ValidContinuationCount",
        "ParserDistanceToComplete",
    }

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        max_new_tokens: int = 800,
        temperature: float = 0.7,
        top_p: float = 0.9,
        generation_timeout: Optional[int] = None,
        strategy_language: str = "python",
    ):
        """
        Initialize the strategy generator.
        
        Args:
            model_name: Generation model name (default: gpt-5.4)
            device: Device to run on ('cuda', 'mps', 'cpu', or None for auto)
            torch_dtype: Torch dtype for model (default: auto based on device)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p (nucleus) sampling parameter
            generation_timeout: Max seconds per LLM call (None = no limit). Use to avoid unbounded hangs.
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        self.uses_openai = _is_openai_model_name(self.model_name)
        self.strategy_language = strategy_language
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.generation_timeout = generation_timeout  # seconds; None = no timeout
        self.load_in_4bit = _generator_load_in_4bit_enabled()
        
        # Auto-detect device. Avoid plain "cuda" when all GPUs are saturated;
        # that can hang in Transformers' load/warmup path before useful output.
        if device is None or device.strip().lower() == "auto":
            device = _auto_select_device()
        self.device = device
        
        # Auto-detect dtype
        if torch_dtype is None:
            if device is not None and (device == "cuda" or device.startswith("cuda:")):
                torch_dtype = torch.bfloat16
            else:
                torch_dtype = torch.float32
        self.torch_dtype = torch_dtype
        
        # Lazy loading - model loaded on first use
        self._model = None
        self._tokenizer = None
        self.last_raw_outputs: list[str] = []
        self.last_generation_diagnostics: list[dict[str, object]] = []
        self.last_structure_repair_trace: list[dict[str, object]] = []
        self.last_structure_validation_summary: dict[str, object] = {}
        self.last_rationale_repair_count = 0

        if self.strategy_language == "dafny":
            self.template_path = self.TEMPLATE_DAFNY_PATH
            self.strategy_begin_marker = "  // QWEN_INSERT_STRATEGY_BEGIN"
            self.strategy_end_marker = "  // QWEN_INSERT_STRATEGY_END"
        else:
            self.template_path = self.TEMPLATE_PY_PATH
            self.strategy_begin_marker = "    # QWEN_INSERT_STRATEGY_BEGIN"
            self.strategy_end_marker = "    # QWEN_INSERT_STRATEGY_END"

        # Load template
        self._template = self._load_template()
    
    def _load_template(self) -> str:
        """Load the configured strategy template."""
        if not self.template_path.exists():
            raise FileNotFoundError(
                f"Template not found at {self.template_path}. "
                "Make sure the configured generation/csd template exists."
            )
        return self.template_path.read_text()
    
    def _ensure_model_loaded(self) -> None:
        """Lazy-load the model and tokenizer. On CUDA OOM, try other GPUs before CPU."""
        if self.uses_openai:
            return
        if self._model is None:
            print(f"Loading {self.model_name}...")
            tokenizer_kwargs = {
                "trust_remote_code": True,
            }
            if _hf_offline_enabled():
                tokenizer_kwargs["local_files_only"] = True
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    **tokenizer_kwargs,
                )
            except Exception as e:
                if tokenizer_kwargs.get("local_files_only") or not _is_hf_connection_error(e):
                    raise
                print("  HuggingFace network lookup failed; retrying tokenizer load from local cache only.")
                tokenizer_kwargs["local_files_only"] = True
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    **tokenizer_kwargs,
                )

            device_map = self.device if (self.device and self.device != "mps") else None
            model_kwargs = {
                "device_map": device_map,
                "trust_remote_code": True,
            }
            if self.load_in_4bit and self.device and self.device.startswith("cuda"):
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                print("  Loading generator model in 4-bit CUDA mode.")
            else:
                model_kwargs["torch_dtype"] = self.torch_dtype
            if _hf_offline_enabled():
                model_kwargs["local_files_only"] = True

            def _is_accelerator_oom(exc: Exception) -> bool:
                error_str = str(exc).lower()
                return bool(
                    self.device
                    and (self.device.startswith("cuda") or self.device == "mps")
                    and (
                        "out of memory" in error_str
                        or "cudaerrormemoryallocation" in error_str
                        or "mem_get_info" in error_str
                        or "cudamemgetinfo" in error_str
                        or "cuda error" in error_str
                    )
                )

            def _load_after_device_oom(exc: Exception) -> None:
                print(f"⚠️  {self.device.upper()} out of memory: {exc}")
                self._model = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
                tried: set[int] = set()
                if self.device.startswith("cuda:"):
                    try:
                        tried.add(int(self.device.split(":")[1]))
                    except (IndexError, ValueError):
                        pass

                def free_memory(gpu_id: int) -> int:
                    try:
                        return int(torch.cuda.mem_get_info(gpu_id)[0])
                    except Exception:
                        return -1

                candidate_gpus = sorted(range(n_gpus), key=free_memory, reverse=True)
                loaded = False
                for gpu_id in candidate_gpus:
                    if gpu_id in tried:
                        continue
                    cand = f"cuda:{gpu_id}"
                    try:
                        torch.cuda.empty_cache()
                        self.device = cand
                        self.torch_dtype = torch.bfloat16
                        retry_kwargs = {
                            "device_map": self.device,
                            "trust_remote_code": True,
                        }
                        if self.load_in_4bit:
                            retry_kwargs["quantization_config"] = BitsAndBytesConfig(
                                load_in_4bit=True,
                                bnb_4bit_compute_dtype=torch.float16,
                                bnb_4bit_quant_type="nf4",
                                bnb_4bit_use_double_quant=True,
                            )
                        else:
                            retry_kwargs["torch_dtype"] = self.torch_dtype
                        if _hf_offline_enabled():
                            retry_kwargs["local_files_only"] = True
                        self._model = AutoModelForCausalLM.from_pretrained(
                            self.model_name,
                            **retry_kwargs,
                        )
                        print(f"   Loaded on {self.device} instead.")
                        loaded = True
                        break
                    except Exception as retry_exc:
                        if not _is_accelerator_oom(retry_exc):
                            raise
                        print(f"   {cand} also ran out of memory.")
                        self._model = None
                        torch.cuda.empty_cache()
                        continue

                if not loaded:
                    print("   Falling back to CPU (this will be slower)...")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    self.device = "cpu"
                    self.torch_dtype = torch.float32
                    cpu_kwargs = {
                        "torch_dtype": self.torch_dtype,
                        "trust_remote_code": True,
                    }
                    if _hf_offline_enabled():
                        cpu_kwargs["local_files_only"] = True
                    self._model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        **cpu_kwargs,
                    ).to(self.device)
                    print(f"Model loaded on {self.device} (CPU fallback)")

            try:
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    **model_kwargs,
                )
            except Exception as e:
                if _is_hf_connection_error(e) and not model_kwargs.get("local_files_only"):
                    print("  HuggingFace network lookup failed; retrying model load from local cache only.")
                    model_kwargs["local_files_only"] = True
                    try:
                        self._model = AutoModelForCausalLM.from_pretrained(
                            self.model_name,
                            **model_kwargs,
                        )
                    except Exception as retry_exc:
                        if not _is_accelerator_oom(retry_exc):
                            raise
                        _load_after_device_oom(retry_exc)
                elif _is_accelerator_oom(e):
                    _load_after_device_oom(e)
                else:
                    raise
            try:
                if self.device == "mps":
                    self._model = self._model.to(self.device)
                print(f"Model loaded on {self.device}")
            except Exception as e:
                if not _is_accelerator_oom(e):
                    raise
                _load_after_device_oom(e)
    
    def _generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Generate text using the configured model.
        
        Args:
            system_prompt: System message
            user_prompt: User message
            
        Returns:
            Generated text
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        if self.uses_openai:
            if not os.environ.get("OPENAI_API_KEY"):
                raise RuntimeError(
                    "OpenAI generation requires OPENAI_API_KEY. "
                    "Set it in the environment, or pass a HuggingFace model name."
                )
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError(
                    "OpenAI generation requires the `openai` package. "
                    "Install dependencies with `pip install -r requirements.txt`."
                ) from exc

            client_kwargs = {}
            if self.generation_timeout is not None and self.generation_timeout > 0:
                client_kwargs["timeout"] = self.generation_timeout
            client = OpenAI(**client_kwargs)
            response_kwargs = {
                "model": self.model_name,
                "input": messages,
                "max_output_tokens": max_new_tokens if max_new_tokens is not None else self.max_new_tokens,
                "temperature": temperature if temperature is not None else self.temperature,
                "top_p": self.top_p,
            }
            try:
                response = client.responses.create(**response_kwargs)
            except Exception as exc:
                text_exc = str(exc).lower()
                if "unsupported" not in text_exc or ("temperature" not in text_exc and "top_p" not in text_exc):
                    raise
                response_kwargs.pop("temperature", None)
                response_kwargs.pop("top_p", None)
                response = client.responses.create(**response_kwargs)
            text = getattr(response, "output_text", None)
            if text is None:
                chunks: list[str] = []
                for item in getattr(response, "output", []) or []:
                    for content in getattr(item, "content", []) or []:
                        content_text = getattr(content, "text", None)
                        if content_text:
                            chunks.append(content_text)
                text = "".join(chunks)
            response_text = (text or "").strip()
            self.last_raw_outputs.append(response_text)
            self.last_raw_outputs = self.last_raw_outputs[-10:]
            return response_text

        self._ensure_model_loaded()
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self._tokenizer(text, return_tensors="pt").to(self.device)

        def _run_generate():
            with torch.no_grad():
                out = self._model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens if max_new_tokens is not None else self.max_new_tokens,
                    temperature=temperature if temperature is not None else self.temperature,
                    top_p=self.top_p,
                    do_sample=True,
                    pad_token_id=self._tokenizer.eos_token_id
                )
            gen = out[0][inputs["input_ids"].shape[1]:]
            return self._tokenizer.decode(gen, skip_special_tokens=True)

        if self.generation_timeout is not None and self.generation_timeout > 0:
            with ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(_run_generate)
                try:
                    response = fut.result(timeout=self.generation_timeout)
                except FuturesTimeoutError:
                    raise RuntimeError(
                        f"Generation timed out after {self.generation_timeout}s. "
                        "If the model is on CPU or slow GPU, try --generation-timeout 0 (disable) or lower --max-tokens."
                    ) from None
        else:
            response = _run_generate()

        response = response.strip()
        self.last_raw_outputs.append(response)
        self.last_raw_outputs = self.last_raw_outputs[-10:]
        return response

    def _diagnostic_excerpt(self, text: str) -> str:
        """Keep failure reports useful without letting one raw generation dominate them."""
        if len(text) <= self.DIAGNOSTIC_TEXT_LIMIT:
            return text
        half = self.DIAGNOSTIC_TEXT_LIMIT // 2
        omitted = len(text) - self.DIAGNOSTIC_TEXT_LIMIT
        return (
            text[:half]
            + f"\n...[truncated {omitted} chars]...\n"
            + text[-half:]
        )
    
    def _extract_strategy(self, raw_output: str) -> str:
        """
        Extract the Python strategy body from Qwen's output.
        
        Handles cases where Qwen includes extra text, code blocks, etc.
        
        Args:
            raw_output: Raw text from Qwen
            
        Returns:
            Cleaned strategy body
        """
        code_block_pattern = r"```(?:python|py|dafny)?\s*([\s\S]*?)```"
        match = re.search(code_block_pattern, raw_output)
        if match:
            raw_output = match.group(1)
        else:
            truncated_pattern = r"^```(?:python|py|dafny)?\s*([\s\S]*)$"
            match = re.search(truncated_pattern, raw_output.strip())
            if match:
                raw_output = match.group(1)

        strategy = raw_output.strip()

        # If the model returned a full Python function, strip the signature and dedent the body.
        func_match = re.search(r"def\s+MyCSDStrategy\s*\([^)]*\)\s*(?:->\s*[^:]+)?\s*:\s*\n([\s\S]*)$", strategy)
        if func_match:
            strategy = textwrap.dedent(func_match.group(1)).strip()

        # Best-effort normalization from older Dafny-oriented outputs.
        strategy = re.sub(r"(?m)^(\s*)//", r"\1#", strategy)
        strategy = strategy.replace(":=", "=")
        strategy = strategy.replace("&&", " and ")
        strategy = strategy.replace("||", " or ")
        strategy = re.sub(r"(?<![=!])!(?!=)", " not ", strategy)
        strategy = re.sub(r"(?m)^(\s*)(invariant|decreases)\b", r"\1# \2", strategy)
        strategy = re.sub(r"(?m);\s*$", "", strategy)
        strategy = self._autofix_python_strategy(strategy)

        return strategy.strip()

    def _extract_dafny_strategy(self, raw_output: str) -> str:
        """Extract a Dafny method body from the model output."""
        code_block_pattern = r"```(?:python|py|dafny)?\s*([\s\S]*?)```"
        match = re.search(code_block_pattern, raw_output)
        if match:
            raw_output = match.group(1)
        else:
            truncated_pattern = r"^```(?:python|py|dafny)?\s*([\s\S]*)$"
            match = re.search(truncated_pattern, raw_output.strip())
            if match:
                raw_output = match.group(1)

        strategy = raw_output.strip()
        method_match = re.search(
            r"method\s+MyCSDStrategy\s*\([^)]*\)\s*(?:returns\s*\([^)]*\))?[\s\S]*?\{([\s\S]*)\}\s*$",
            strategy,
        )
        if method_match:
            strategy = textwrap.dedent(method_match.group(1)).strip()

        strategy = re.sub(r"(?m)^(\s*)#", r"\1//", strategy)
        return strategy.strip()

    def _has_required_comment_blocks(self, strategy_body: str) -> bool:
        """Return True when both reasoning scaffolds are present and non-empty."""
        rationale = extract_rationale(strategy_body)
        proof_sketch = extract_proof_sketch(strategy_body)
        return (
            rationale.rationale is not None
            and rationale.has_markers
            and proof_sketch.text is not None
            and proof_sketch.has_markers
        )

    def _ensure_nontrivial_dafny_strategy(self, strategy_body: str) -> str:
        """Lightweight structural validation for Dafny-first fallback mode."""
        body = self._body_without_rationale(strategy_body)
        executable_lines = [
            line for line in body.splitlines()
            if line.strip() and not line.lstrip().startswith("//") and not line.lstrip().startswith("#")
        ]
        if not executable_lines:
            raise ValueError("The body has no executable Dafny statements after the rationale block.")
        if "while " not in body:
            raise ValueError("The body must contain a while loop that performs decoding steps.")
        if "helpers." not in body:
            raise ValueError("The body must call helper methods from `helpers`.")
        return strategy_body

    def _ensure_rationale_block(self, strategy_body: str, *, max_repairs: int = 2) -> str:
        """
        Ensure the strategy body contains required rationale and proof-sketch markers.

        If missing, attempt a small number of "format repair" generations that rewrite
        the body into the required structure without changing semantics.
        """
        if self._has_required_comment_blocks(strategy_body):
            self.last_rationale_repair_count = 0
            return self._normalize_rationale_block(strategy_body)

        current = strategy_body
        for repair_round in range(1, max_repairs + 1):
            system_prompt, user_prompt = build_format_repair_prompt(
                current,
                strategy_language=self.strategy_language,
            )
            repaired_raw = self._generate_text(system_prompt, user_prompt)
            repaired = (
                self._extract_dafny_strategy(repaired_raw)
                if self.strategy_language == "dafny"
                else self._extract_strategy(repaired_raw)
            )
            if self._has_required_comment_blocks(repaired):
                self.last_rationale_repair_count = repair_round
                return self._normalize_rationale_block(repaired)
            current = repaired

        self.last_rationale_repair_count = max_repairs
        raise ValueError(
            "Generated strategy is missing required rationale/proof-sketch block markers "
            "(# CSD_RATIONALE_BEGIN ... # CSD_RATIONALE_END and "
            "# CSD_PROOF_SKETCH_BEGIN ... # CSD_PROOF_SKETCH_END)."
        )

    def _body_without_rationale(self, strategy_body: str) -> str:
        extracted = extract_rationale(strategy_body)
        return extracted.body_without_rationale if extracted.has_markers else strategy_body

    def _normalize_rationale_block(self, strategy_body: str) -> str:
        strategy_body = self._normalize_comment_block(
            strategy_body,
            begin_markers={"# CSD_RATIONALE_BEGIN", "// CSD_RATIONALE_BEGIN"},
            end_markers={"# CSD_RATIONALE_END", "// CSD_RATIONALE_END"},
        )
        return self._normalize_comment_block(
            strategy_body,
            begin_markers={"# CSD_PROOF_SKETCH_BEGIN", "// CSD_PROOF_SKETCH_BEGIN"},
            end_markers={"# CSD_PROOF_SKETCH_END", "// CSD_PROOF_SKETCH_END"},
        )

    def _normalize_comment_block(
        self,
        strategy_body: str,
        *,
        begin_markers: set[str],
        end_markers: set[str],
    ) -> str:
        lines = strategy_body.splitlines()
        begin_idx = None
        end_idx = None
        for i, line in enumerate(lines):
            if line.strip() in begin_markers:
                begin_idx = i
                break
        if begin_idx is None:
            return strategy_body
        for j in range(begin_idx + 1, len(lines)):
            if lines[j].strip() in end_markers:
                end_idx = j
                break
        if end_idx is None:
            return strategy_body

        normalized = list(lines)
        for k in range(begin_idx + 1, end_idx):
            raw = normalized[k]
            stripped = raw.strip()
            if not stripped:
                continue
            if stripped.startswith("#") or stripped.startswith("//"):
                continue
            indent = raw[: len(raw) - len(raw.lstrip())]
            normalized[k] = f"{indent}# {stripped}"
        return "\n".join(normalized)

    def _autofix_python_strategy(self, strategy_body: str) -> str:
        lines = strategy_body.splitlines()
        fixed: list[str] = []
        complete_block_indent: int | None = None
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped_line = line.lstrip()
            indent = len(line) - len(stripped_line)

            # Track `if/elif helpers.IsComplete(generated):` blocks so we can
            # rewrite invalid constrained-step usage into an explicit close.
            if complete_block_indent is not None and stripped_line and indent <= complete_block_indent:
                complete_block_indent = None
            if (
                stripped_line.startswith("if ")
                or stripped_line.startswith("elif ")
            ) and "helpers.IsComplete(generated)" in stripped_line and stripped_line.endswith(":") and "helpers.CanConstrain(generated)" not in stripped_line:
                complete_block_indent = indent

            if complete_block_indent is not None and indent > complete_block_indent:
                complete_replacement = (
                    r"\1generated, stepsLeft = helpers.AppendConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft)"
                    if _env_flag("CSD_REQUIRE_NATURAL_DELIMITERS", False)
                    else r"\1generated, stepsLeft = helpers.AppendRightDelimiter(generated, stepsLeft)"
                )
                line = re.sub(
                    r"^(\s*)generated\s*,\s*stepsLeft\s*=\s*helpers\.AppendConstrainedStep\(\s*prompt\s*,\s*generated\s*,\s*stepsLeft\s*\)\s*$",
                    complete_replacement,
                    line,
                )

            fixed.append(line)
            stripped = line.lstrip()
            if stripped.startswith("if ") and stripped.endswith(":"):
                branch_indent = " " * (indent + 4)
                if i + 3 < len(lines):
                    first_branch = lines[i + 1]
                    else_line = None
                    for j in range(i + 1, len(lines)):
                        if lines[j].startswith(" " * indent + "else:"):
                            else_line = j
                            break
                    if else_line is not None and else_line + 1 < len(lines):
                        branch_assign = re.match(
                            rf"^{re.escape(branch_indent)}([A-Za-z_]\w*)\s*,\s*([A-Za-z_]\w*)\s*=\s*helpers\.(?:ConstrainedStep|ConstrainedOrRightDelimiterStep|UnconstrainedStep|UnconstrainedAllowLeftDelimiterStep|UnconstrainedBiasLeftDelimiterStep|UnconstrainedNudgeLeftDelimiterStep|ForcedTokenStep)\(",
                            first_branch,
                        )
                        else_assign = re.match(
                            rf"^{re.escape(branch_indent)}([A-Za-z_]\w*)\s*,\s*([A-Za-z_]\w*)\s*=\s*helpers\.(?:ConstrainedStep|ConstrainedOrRightDelimiterStep|UnconstrainedStep|UnconstrainedAllowLeftDelimiterStep|UnconstrainedBiasLeftDelimiterStep|UnconstrainedNudgeLeftDelimiterStep|ForcedTokenStep)\(",
                            lines[else_line + 1],
                        )
                        if branch_assign and else_assign and branch_assign.groups() == else_assign.groups():
                            name1, name2 = branch_assign.groups()
                            prev_slice = "\n".join(fixed[:-1])
                            if not re.search(rf"(?m)^\s*{re.escape(name1)}\s*=", prev_slice):
                                fixed.insert(len(fixed) - 1, " " * indent + f"{name1} = eosToken")
                            if not re.search(rf"(?m)^\s*{re.escape(name2)}\s*=", prev_slice):
                                default_rhs = "stepsLeft"
                                fixed.insert(len(fixed) - 1, " " * indent + f"{name2} = {default_rhs}")
            i += 1
        normalized = "\n".join(fixed)
        # Normalize malformed helper prompt arguments frequently emitted by models:
        # use `prompt` instead of an empty list literal for prompt-taking helpers.
        normalized = re.sub(
            r"helpers\.(AppendUnconstrainedStep|AppendUnconstrainedAllowLeftDelimiterStep|AppendUnconstrainedNudgeLeftDelimiterStep|AppendConstrainedStep|AppendConstrainedOrRightDelimiterStep|UnconstrainedStep|UnconstrainedAllowLeftDelimiterStep|UnconstrainedBiasLeftDelimiterStep|UnconstrainedNudgeLeftDelimiterStep|ConstrainedStep|ConstrainedOrRightDelimiterStep|AppendSoftConstrainedStep|SoftConstrainedStep)\(\s*\[\s*\]\s*,",
            r"helpers.\1(prompt,",
            normalized,
        )
        # Append* helpers return `(generated, stepsLeft)` and must not be assigned
        # to `stepsLeft` alone.
        normalized = re.sub(
            r"(?m)^(\s*)stepsLeft\s*=\s*helpers\.(AppendUnconstrainedStep|AppendUnconstrainedAllowLeftDelimiterStep|AppendUnconstrainedNudgeLeftDelimiterStep|AppendConstrainedStep|AppendConstrainedOrRightDelimiterStep|AppendSoftConstrainedStep|AppendTopKConstrainedStep|AppendLeftDelimiter|AppendRightDelimiter|AppendForcedToken)\(",
            r"\1generated, stepsLeft = helpers.\2(",
            normalized,
        )
        if _env_flag("CSD_REQUIRE_NATURAL_DELIMITERS", False):
            # Natural right-delimiter helpers are intentionally completion-aware:
            # calling them when the suffix is complete is how `>>` becomes
            # available. Do not collapse the mixed guard to CanConstrain-only.
            normalized = re.sub(
                r"(?m)^(\s*(?:if|elif)\s+)helpers\.CanConstrain\(generated\)(\s*:)",
                r"\1(helpers.IsComplete(generated) or helpers.CanConstrain(generated))\2",
                normalized,
            )
        else:
            # Plain constrained-step calls have a !IsComplete precondition, so
            # legacy explicit-delimiter strategies need CanConstrain-only guards.
            normalized = re.sub(
                r"helpers\.IsComplete\(generated\)\s+or\s+helpers\.CanConstrain\(generated\)",
                r"helpers.CanConstrain(generated)",
                normalized,
            )
            normalized = re.sub(
                r"helpers\.CanConstrain\(generated\)\s+or\s+helpers\.IsComplete\(generated\)",
                r"helpers.CanConstrain(generated)",
                normalized,
            )
        # Transpiler-safe fallback for generated string cue scans over
        # LongestValidSuffix (e.g. "".join(...).lower() and "... in suffix").
        normalized = re.sub(
            r'(?m)^(\s*)([A-Za-z_]\w*)\s*=\s*""\.join\(\s*helpers\.LongestValidSuffix\(generated\)\s*\)\.lower\(\)\s*$',
            r'\1\2 = ""',
            normalized,
        )
        normalized = re.sub(
            r'(?m)^(\s*)if\s+.*\bin\s+suffix\b.*:\s*$',
            r"\1if False:",
            normalized,
        )
        # Conservatively avoid non-decreasing loop paths caused by `continue`.
        normalized = re.sub(r"(?m)^(\s*)continue\s*$", r"\1break", normalized)
        # Add a direct decreases guard in step-budget loops: only continue when
        # `stepsLeft` strictly decreases over the full iteration.
        lines3 = normalized.splitlines()
        guarded: list[str] = []
        i3 = 0
        while i3 < len(lines3):
            line3 = lines3[i3]
            stripped3 = line3.lstrip()
            indent3 = len(line3) - len(stripped3)
            if stripped3.startswith("while ") and stripped3.endswith(":") and "stepsLeft > 0" in stripped3:
                guarded.append(line3)
                body_lines: list[str] = []
                i3 += 1
                while i3 < len(lines3):
                    nxt = lines3[i3]
                    nxt_stripped = nxt.lstrip()
                    nxt_indent = len(nxt) - len(nxt_stripped)
                    if nxt_stripped and nxt_indent <= indent3:
                        break
                    body_lines.append(nxt)
                    i3 += 1
                body_indent = " " * (indent3 + 4)
                has_snapshot = any(
                    b.lstrip().startswith("stepsLeftBeforeIteration = stepsLeft")
                    for b in body_lines
                )
                has_decreases_guard = any(
                    b.lstrip().startswith("if stepsLeft >= stepsLeftBeforeIteration:")
                    for b in body_lines
                )
                if not has_snapshot:
                    guarded.append(f"{body_indent}stepsLeftBeforeIteration = stepsLeft")
                guarded.extend(body_lines)
                if not has_decreases_guard:
                    guarded.append(f"{body_indent}if stepsLeft >= stepsLeftBeforeIteration:")
                    guarded.append(f"{body_indent}    break")
                continue
            guarded.append(line3)
            i3 += 1
        normalized = "\n".join(guarded)
        # Salvage truncated generations by dropping malformed tail lines until
        # the method body parses as Python.
        parse_candidate = normalized
        for _ in range(24):
            try:
                wrapped = "def _strategy():\n" + textwrap.indent(parse_candidate, "    ")
                ast.parse(wrapped)
                normalized = parse_candidate
                break
            except SyntaxError:
                candidate_lines = parse_candidate.splitlines()
                if not candidate_lines:
                    break
                candidate_lines = candidate_lines[:-1]
                while candidate_lines and not candidate_lines[-1].strip():
                    candidate_lines.pop()
                parse_candidate = "\n".join(candidate_lines)
        # Add a conservative fallback branch for dangling top-level `if/elif`
        # chains in `stepsLeft` loops so the loop can always terminate.
        try:
            wrapped = "def _strategy():\n" + textwrap.indent(normalized, "    ")
            tree = ast.parse(wrapped)
            loop_progress_helpers = {
                "AppendUnconstrainedStep",
                "AppendUnconstrainedAllowLeftDelimiterStep",
                "AppendUnconstrainedNudgeLeftDelimiterStep",
                "AppendConstrainedStep",
                "AppendConstrainedOrRightDelimiterStep",
                "AppendSoftConstrainedStep",
                "AppendTopKConstrainedStep",
                "AppendLeftDelimiter",
                "AppendRightDelimiter",
                "AppendForcedToken",
                "UnconstrainedStep",
                "UnconstrainedAllowLeftDelimiterStep",
                "UnconstrainedBiasLeftDelimiterStep",
                "UnconstrainedNudgeLeftDelimiterStep",
                "ConstrainedStep",
                "ConstrainedOrRightDelimiterStep",
                "SoftConstrainedStep",
                "TopKConstrainedStep",
                "ForcedTokenStep",
            }

            def _contains_loop_progress(statements: list[ast.stmt]) -> bool:
                for statement in statements:
                    for inner in ast.walk(statement):
                        if isinstance(inner, (ast.Break, ast.Return)):
                            return True
                        if (
                            isinstance(inner, ast.Call)
                            and isinstance(inner.func, ast.Attribute)
                            and isinstance(inner.func.value, ast.Name)
                            and inner.func.value.id == "helpers"
                            and inner.func.attr in loop_progress_helpers
                        ):
                            return True
                return False

            lines4 = normalized.splitlines()
            insertions: list[tuple[int, int]] = []
            for node in ast.walk(tree):
                if not isinstance(node, ast.While) or "stepsLeft > 0" not in ast.unparse(node.test):
                    continue
                for stmt_idx, statement in enumerate(node.body):
                    if not isinstance(statement, ast.If):
                        continue
                    guard_test_text = ast.unparse(statement.test)
                    if (
                        "stepsLeftBeforeIteration" in guard_test_text
                        and "stepsLeft >=" in guard_test_text
                    ):
                        continue
                    tail_if = statement
                    while len(tail_if.orelse) == 1 and isinstance(tail_if.orelse[0], ast.If):
                        tail_if = tail_if.orelse[0]
                    if tail_if.orelse:
                        continue
                    trailing = node.body[stmt_idx + 1 :]
                    if _contains_loop_progress(trailing):
                        continue
                    insert_after = getattr(tail_if, "end_lineno", None)
                    if_line = getattr(statement, "lineno", None)
                    if insert_after is None or if_line is None:
                        continue
                    if_index = if_line - 2
                    insert_index = insert_after - 1
                    if not (0 <= if_index < len(lines4)) or not (0 <= insert_index <= len(lines4)):
                        continue
                    indent = len(lines4[if_index]) - len(lines4[if_index].lstrip())
                    probe = insert_index
                    while probe < len(lines4) and not lines4[probe].strip():
                        probe += 1
                    if probe < len(lines4) and lines4[probe].startswith(" " * indent + "else:"):
                        continue
                    insertions.append((insert_index, indent))
            for insert_index, indent in sorted(set(insertions), reverse=True):
                lines4[insert_index:insert_index] = [
                    " " * indent + "else:",
                    " " * (indent + 4) + "break",
                ]
            normalized = "\n".join(lines4)
        except Exception:
            pass
        return normalized

    def _structural_issue(self, strategy_body: str) -> str | None:
        body = self._body_without_rationale(strategy_body)
        prefer_scratch_spans = _env_flag("CSD_GSM_PREFER_SCRATCH_SPANS", False)
        require_natural_delimiters = _env_flag("CSD_REQUIRE_NATURAL_DELIMITERS", False)
        skip_structural_validation = _env_flag("CSD_SKIP_STRUCTURAL_VALIDATION", False)
        spider_force_single_sql_span = _env_flag("CSD_SPIDER_FORCE_SINGLE_SQL_SPAN", False)
        spider_force_span_at_start = _env_flag("CSD_SPIDER_FORCE_SPAN_AT_START", False)
        has_step_snapshot_guard = (
            "stepsLeftBeforeIteration = stepsLeft" in body
            and "if stepsLeft >= stepsLeftBeforeIteration:" in body
        )
        executable_lines = [
            line for line in body.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        if not executable_lines:
            return "The body has no executable Python statements after the rationale block."

        try:
            wrapped = "def _strategy():\n" + textwrap.indent(body, "    ")
            tree = ast.parse(wrapped)
        except SyntaxError as exc:
            return f"The body is not valid Python: {exc.msg}."

        if skip_structural_validation:
            return None

        parent_map: dict[ast.AST, ast.AST] = {}
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                parent_map[child] = parent

        has_while = any(isinstance(node, ast.While) for node in ast.walk(tree))
        if not has_while:
            return "The body must contain a while loop that performs decoding steps."

        expected_invariant_block = [
            "# invariant helpers.lm == lm",
            "# invariant helpers.parser == parser",
            "# invariant lm.ValidTokensIdsLogits()",
            "# invariant 0 <= stepsLeft <= maxSteps",
            "# invariant |generated| + stepsLeft <= maxSteps",
            "# decreases stepsLeft",
        ]
        body_lines = body.splitlines()

        constrained_step_calls = 0
        forced_token_calls = 0
        unconstrained_calls = 0
        split_prefix_step_calls = 0
        emits_left_delimiter = False
        emits_right_delimiter = False
        left_delimiter_lines: list[int] = []
        right_delimiter_lines: list[int] = []
        unconstrained_lines: list[int] = []
        appends_generated = False
        bare_forced_token_calls = 0
        bare_append_helper_calls = 0
        append_helper_wrong_targets: set[str] = set()
        keyword_helper_calls: set[str] = set()
        topk_unprovable_calls: set[str] = set()
        delimiter_calls_outside_loop: set[str] = set()
        unguarded_right_delimiter_calls: set[str] = set()
        nondecreasing_else_lines: list[int] = []
        dangling_if_chain_lines: list[int] = []
        top_level_break_lines: list[int] = []
        manual_stepsleft_mutations: list[int] = []
        none_checkpoint_assign_lines: list[int] = []
        insufficient_reason_budget_lines: list[int] = []
        insufficient_answer_budget_lines: list[int] = []
        trivial_reason_budget_lines: list[int] = []
        trivial_answer_budget_lines: list[int] = []
        fixed_phase_quota_lines: list[int] = []
        malformed_tuple_assignment_lines: list[int] = []
        mutable_float_state: set[str] = set()
        float_comparisons: list[int] = []
        assigns_remaining_steps = False
        has_return = False
        print_calls = 0
        extra_state: set[str] = set()
        unsupported_helper_calls: set[str] = set()
        unsupported_parser_calls: set[str] = set()
        parser_on_generated_methods: set[str] = set()
        generated_string_methods: set[str] = set()
        suffix_string_methods: set[str] = set()
        old_api_calls: set[str] = set()
        repair_helper_calls: set[str] = set()
        helper_calls: set[str] = set()
        helper_parser_confusions: set[str] = set()
        unguarded_constrained_calls: set[str] = set()
        complete_branch_constrained_lines: list[int] = []
        constrain_before_complete_lines: list[int] = []
        bad_bias_helper_lines: list[int] = []
        uses_natural_left_delimiter = False
        uses_natural_right_delimiter = False
        uses_split_prefix_policy = False
        single_right_close_terminal_lines: list[int] = []
        forced_left_delimiter_lines: list[int] = []
        forced_right_delimiter_lines: list[int] = []
        low_final_ready_lines: list[int] = []
        late_budget_answer_pressure_lines: list[int] = []
        low_reason_nudge_lines: list[int] = []
        parser_readiness_early_open_lines: list[int] = []
        phase_break_open_lines: list[int] = []
        state_only_open_transition_lines: list[int] = []
        natural_open_plain_fallback_lines: list[int] = []
        negative_index_lines: list[int] = []
        premature_not_can_constrain_lines: list[int] = []
        natural_plain_constrained_lines: list[int] = []
        natural_completion_blind_right_helper_lines: list[int] = []
        sequential_helper_without_budget_lines: list[int] = []
        generated_join_lines: list[int] = []
        stray_expression_lines: list[int] = []
        continue_lines: list[int] = []
        budget_only_open_lines: list[int] = []
        spider_long_freeform_lines: list[int] = []
        bad_prompt_arg_calls: list[tuple[str, int]] = []
        if_count = 0
        first_helper_call_line = 0
        first_helper_call_attr = ""
        while_nodes = [node for node in ast.walk(tree) if isinstance(node, ast.While)]
        append_helper_methods = {
            "AppendUnconstrainedStep",
            "AppendUnconstrainedAllowLeftDelimiterStep",
            "AppendUnconstrainedNudgeLeftDelimiterStep",
            "AppendConstrainedStep",
            "AppendConstrainedOrRightDelimiterStep",
            "AppendForcedToken",
            "AppendLeftDelimiter",
            "AppendRightDelimiter",
        }
        generated_updating_helper_methods = append_helper_methods | {
            "OpenConstrainedSpan",
            "CloseConstrainedSpan",
            "AppendConstrainedToken",
        }
        constrained_helper_methods = {
            "ConstrainedStep",
            "ConstrainedOrRightDelimiterStep",
            "AppendConstrainedStep",
            "AppendConstrainedOrRightDelimiterStep",
            "AdaptiveConstrainedStep",
            "GroupBoostedConstrainedStep",
            "PenalizedConstrainedStep",
        }
        split_prefix_step_methods = {
            "OpenConstrainedSpan",
            "CloseConstrainedSpan",
            "AdaptiveConstrainedStep",
            "GroupBoostedConstrainedStep",
            "PenalizedConstrainedStep",
        }
        unconstrained_helper_methods = {
            "UnconstrainedStep",
            "UnconstrainedAllowLeftDelimiterStep",
            "UnconstrainedBiasLeftDelimiterStep",
            "UnconstrainedNudgeLeftDelimiterStep",
            "AppendUnconstrainedStep",
            "AppendUnconstrainedAllowLeftDelimiterStep",
            "AppendUnconstrainedNudgeLeftDelimiterStep",
        }
        forced_helper_methods = {
            "ForcedTokenStep",
            "AppendForcedToken",
            "AppendLeftDelimiter",
            "AppendRightDelimiter",
        }
        step_consuming_helper_methods = (
            append_helper_methods
            | constrained_helper_methods
            | unconstrained_helper_methods
            | forced_helper_methods
        )
        prompt_arg_required_methods = {
            "UnconstrainedStep",
            "UnconstrainedAllowLeftDelimiterStep",
            "UnconstrainedBiasLeftDelimiterStep",
            "UnconstrainedNudgeLeftDelimiterStep",
            "ConstrainedStep",
            "ConstrainedOrRightDelimiterStep",
            "AppendUnconstrainedStep",
            "AppendUnconstrainedAllowLeftDelimiterStep",
            "AppendUnconstrainedNudgeLeftDelimiterStep",
            "AppendConstrainedStep",
            "AppendConstrainedOrRightDelimiterStep",
            "AdaptiveConstrainedStep",
            "GroupBoostedConstrainedStep",
            "PenalizedConstrainedStep",
        }

        OLD_API = {
            "ExpressiveStep", "ConstrainedAnswerStep", "FinalizeDelimitedAnswer",
            "InsideDelimitedWindow", "CompletedDelimitedAnswer", "DelimitedAnswerValid",
            "ConstrainedWindowValid", "GetDelimitedContent", "DelimitersInLM",
            "DelimitersInLMAlways",
        }
        REPAIR_HELPERS = {
            "RollbackToValidPrefix",
            "FindLongestValidSpan",
            "ExtractAllValidSpans",
            "RepairByRetry",
        }

        def _is_name(node: ast.AST, expected: str) -> bool:
            return isinstance(node, ast.Name) and node.id == expected

        def _is_generated_token_sequence(node: ast.AST) -> bool:
            if _is_name(node, "generated"):
                return True
            return (
                isinstance(node, ast.Subscript)
                and _is_name(node.value, "generated")
            )

        def _has_generated_suffix_call(node: ast.AST) -> bool:
            return (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "helpers"
                and node.func.attr == "LongestValidSuffix"
                and len(node.args) >= 1
                and _is_name(node.args[0], "generated")
            )

        def _condition_has_constrain_guard(test: ast.AST) -> bool:
            for inner in ast.walk(test):
                if (
                    isinstance(inner, ast.Call)
                    and isinstance(inner.func, ast.Attribute)
                    and isinstance(inner.func.value, ast.Name)
                    and inner.func.value.id == "helpers"
                    and inner.func.attr == "CanConstrain"
                    and len(inner.args) >= 1
                    and _is_name(inner.args[0], "generated")
                ):
                    return True
            return False

        def _condition_has_complete_guard(test: ast.AST) -> bool:
            for inner in ast.walk(test):
                if (
                    isinstance(inner, ast.Call)
                    and isinstance(inner.func, ast.Attribute)
                    and isinstance(inner.func.value, ast.Name)
                    and inner.func.value.id == "parser"
                    and inner.func.attr == "IsCompletePrefix"
                    and len(inner.args) >= 1
                    and _has_generated_suffix_call(inner.args[0])
                ):
                    return True
                if (
                    isinstance(inner, ast.Call)
                    and isinstance(inner.func, ast.Attribute)
                    and isinstance(inner.func.value, ast.Name)
                    and inner.func.value.id == "helpers"
                    and inner.func.attr == "IsComplete"
                    and len(inner.args) >= 1
                    and _is_name(inner.args[0], "generated")
                ):
                    return True
            return False

        def _condition_has_can_constrain_guard(test: ast.AST) -> bool:
            for inner in ast.walk(test):
                if (
                    isinstance(inner, ast.Call)
                    and isinstance(inner.func, ast.Attribute)
                    and isinstance(inner.func.value, ast.Name)
                    and inner.func.value.id == "helpers"
                    and inner.func.attr == "CanConstrain"
                    and len(inner.args) >= 1
                    and _is_name(inner.args[0], "generated")
                ):
                    return True
            return False

        def _condition_has_not_can_constrain_guard(test: ast.AST) -> bool:
            for inner in ast.walk(test):
                if (
                    isinstance(inner, ast.UnaryOp)
                    and isinstance(inner.op, ast.Not)
                    and _condition_has_can_constrain_guard(inner.operand)
                ):
                    return True
            return False

        def _condition_has_not_complete_guard(test: ast.AST) -> bool:
            for inner in ast.walk(test):
                if (
                    isinstance(inner, ast.UnaryOp)
                    and isinstance(inner.op, ast.Not)
                    and _condition_has_complete_guard(inner.operand)
                ):
                    return True
            return False

        def _condition_has_answer_step_cap(test: ast.AST) -> bool:
            for inner in ast.walk(test):
                if not isinstance(inner, ast.Compare):
                    continue
                parts = [inner.left, *inner.comparators]
                names = {
                    part.id.lower()
                    for part in parts
                    if isinstance(part, ast.Name)
                }
                if not any("answer" in name or "constrained" in name for name in names):
                    continue
                if any(isinstance(op, (ast.Lt, ast.LtE)) for op in inner.ops):
                    return True
            return False

        min_required_reason_steps = 40 if require_natural_delimiters else 0
        min_required_answer_steps = 0

        def _is_reason_budget_name(name: str) -> bool:
            lowered = name.lower()
            return (
                ("reason" in lowered or "setup" in lowered or "prelude" in lowered)
                and any(marker in lowered for marker in ("min", "max", "limit", "budget", "target", "threshold"))
            )

        def _is_answer_budget_name(name: str) -> bool:
            lowered = name.lower()
            return (
                ("answer" in lowered or "constrained" in lowered)
                and any(marker in lowered for marker in ("min", "limit", "budget", "target", "threshold"))
            )

        def _is_fixed_phase_quota_name(name: str) -> bool:
            lowered = name.lower()
            return (
                any(marker in lowered for marker in ("min", "minimum", "max", "limit", "quota", "target", "budget", "threshold"))
                and any(
                    marker in lowered
                    for marker in (
                        "reason",
                        "setup",
                        "prelude",
                        "wrap",
                        "answer",
                        "final",
                        "constrained",
                        "search",
                        "scratch",
                        "span",
                    )
                )
            )

        def _is_bad_fixed_phase_quota(name: str, node: ast.AST | None) -> bool:
            if not _is_literal_int(node):
                return False
            value = node.value
            lowered = name.lower()
            if any(marker in lowered for marker in ("reason", "setup", "prelude")):
                return value < 40
            if "wrap" in lowered:
                return value < 4
            if any(marker in lowered for marker in ("answer", "final", "constrained", "search")):
                return value < 6
            if any(marker in lowered for marker in ("scratch", "span")):
                return value <= 2
            return False

        def _is_spider_freeform_state_name(name: str) -> bool:
            lowered = name.lower()
            return any(marker in lowered for marker in ("freeform", "lead", "intro", "preface", "preamble"))

        def _condition_has_low_reason_final_threshold(test: ast.AST) -> bool:
            for inner in ast.walk(test):
                if not isinstance(inner, ast.Compare):
                    continue
                parts = [inner.left, *inner.comparators]
                names = {
                    part.id.lower()
                    for part in parts
                    if isinstance(part, ast.Name)
                }
                if not any("reason" in name or "signal" in name or "token" in name for name in names):
                    continue
                constants = [
                    part.value
                    for part in parts
                    if isinstance(part, ast.Constant) and isinstance(part.value, int)
                ]
                if not constants or min(constants) >= 40:
                    continue
                if any(isinstance(op, (ast.Gt, ast.GtE, ast.Lt, ast.LtE)) for op in inner.ops):
                    return True
            return False

        def _condition_has_low_setup_or_reason_threshold(test: ast.AST, *, threshold: int) -> bool:
            for inner in ast.walk(test):
                if not isinstance(inner, ast.Compare):
                    continue
                parts = [inner.left, *inner.comparators]
                names = {
                    part.id.lower()
                    for part in parts
                    if isinstance(part, ast.Name)
                }
                if not any(
                    marker in name
                    for name in names
                    for marker in ("reason", "setup", "prelude")
                ):
                    continue
                constants = [
                    part.value
                    for part in parts
                    if isinstance(part, ast.Constant) and isinstance(part.value, int)
                ]
                if constants and min(constants) < threshold:
                    return True
            return False

        def _condition_has_parser_readiness_trigger(test: ast.AST) -> bool:
            readiness_helpers = {
                "CanConstrain",
                "IsComplete",
                "MinStepsToComplete",
                "ParserDistanceToComplete",
                "ValidContinuationCount",
            }
            readiness_parser_methods = {
                "IsCompletePrefix",
                "ParserDistanceToComplete",
                "ValidContinuationCount",
            }
            for inner in ast.walk(test):
                if (
                    isinstance(inner, ast.Call)
                    and isinstance(inner.func, ast.Attribute)
                    and isinstance(inner.func.value, ast.Name)
                ):
                    if inner.func.value.id == "helpers" and inner.func.attr in readiness_helpers:
                        return True
                    if inner.func.value.id == "parser" and inner.func.attr in readiness_parser_methods:
                        return True
            return False

        def _condition_is_opening_context(test: ast.AST) -> bool:
            opening_values = {
                "open",
                "opening",
                "nudge",
                "nudging",
                "seek",
                "seeking",
                "answer",
                "answeropen",
                "answeropening",
            }
            for inner in ast.walk(test):
                if isinstance(inner, ast.Name):
                    lowered = inner.id.lower()
                    if "open_attempt" in lowered or "nudge_attempt" in lowered or "seek_step" in lowered:
                        return True
                if not isinstance(inner, ast.Compare):
                    continue
                parts = [inner.left, *inner.comparators]
                names = {
                    part.id.lower()
                    for part in parts
                    if isinstance(part, ast.Name)
                }
                values = {
                    part.value.replace("_", "").lower()
                    for part in parts
                    if isinstance(part, ast.Constant) and isinstance(part.value, str)
                }
                if (
                    any(name in {"phase", "stage", "state", "mode"} for name in names)
                    and any(value in opening_values for value in values)
                ):
                    return True
            return False

        def _descendant_is_in_if_body(descendant: ast.AST, if_node: ast.If) -> bool:
            """True when descendant executes in if_node.body, not in its else/elif branch."""
            current = descendant
            while current in parent_map:
                parent = parent_map[current]
                if parent is if_node:
                    return current in if_node.body
                current = parent
            return False

        def _is_answer_pressure_name(name: str) -> bool:
            normalized = name.replace("_", "").lower()
            return (
                "answerpressure" in normalized
                or "finalpressure" in normalized
                or "answerready" in normalized
                or "finalready" in normalized
                or "seekspan" in normalized
                or "shouldopen" in normalized
                or "openanswer" in normalized
            )

        def _condition_has_tiny_remaining_budget_trigger(test: ast.AST) -> bool:
            for inner in ast.walk(test):
                if (
                    isinstance(inner, ast.UnaryOp)
                    and isinstance(inner.op, ast.Not)
                    and isinstance(inner.operand, ast.Call)
                    and isinstance(inner.operand.func, ast.Attribute)
                    and isinstance(inner.operand.func.value, ast.Name)
                    and inner.operand.func.value.id == "helpers"
                    and inner.operand.func.attr == "HasBudget"
                    and len(inner.operand.args) >= 2
                    and _is_name(inner.operand.args[0], "stepsLeft")
                    and isinstance(inner.operand.args[1], ast.Constant)
                    and isinstance(inner.operand.args[1].value, int)
                    and inner.operand.args[1].value < 16
                ):
                    return True
                if not isinstance(inner, ast.Compare):
                    continue
                compare_parts = [inner.left, *inner.comparators]
                for left, op, right in zip(compare_parts, inner.ops, compare_parts[1:]):
                    if isinstance(left, ast.Name) and left.id == "stepsLeft" and isinstance(right, ast.Constant):
                        if isinstance(right.value, int) and right.value < 16 and isinstance(op, (ast.Lt, ast.LtE)):
                            return True
                    if isinstance(right, ast.Name) and right.id == "stepsLeft" and isinstance(left, ast.Constant):
                        if isinstance(left.value, int) and left.value < 16 and isinstance(op, (ast.Gt, ast.GtE)):
                            return True
            return False

        def _is_verified_span_counter_name(name: str) -> bool:
            normalized = name.replace("_", "").lower()
            if "token" in normalized or "step" in normalized:
                return False
            span_markers = ("closed", "count", "emitted", "verified", "scratch", "done", "complete")
            return (
                ("span" in normalized and any(marker in normalized for marker in span_markers))
                or ("scratch" in normalized and any(marker in normalized for marker in ("closed", "count", "emitted", "done")))
                or ("verified" in normalized and any(marker in normalized for marker in ("closed", "count", "emitted", "done")))
                or ("mini" in normalized and any(marker in normalized for marker in ("closed", "count", "emitted", "done")))
            )

        def _is_open_span_state_name(name: str) -> bool:
            normalized = name.replace("_", "").lower()
            if "nudge" in normalized or "free" in normalized or "closed" in normalized:
                return False
            return (
                normalized in {"phase", "stage", "state"}
                or "inspan" in normalized
                or "insidespan" in normalized
                or "insideconstrained" in normalized
                or "inconstrained" in normalized
                or "spanopen" in normalized
                or "openspan" in normalized
                or "answeropen" in normalized
                or "constrainedmode" in normalized
                or "spanmode" in normalized
                or ("phase" in normalized and "span" in normalized)
            )

        def _is_final_span_state_name(name: str) -> bool:
            normalized = name.replace("_", "").lower()
            return (
                "final" in normalized
                or "answerready" in normalized
                or "finalready" in normalized
                or "finalspan" in normalized
                or "isfinal" in normalized
                or "doneafterfinal" in normalized
            )

        def _is_scratch_span_state_name(name: str) -> bool:
            normalized = name.replace("_", "").lower()
            return (
                ("scratch" in normalized and any(marker in normalized for marker in ("mode", "ready", "open", "span", "phase", "intent")))
                or normalized in {"scratchmode", "scratchphase", "scratchready", "scratchintent", "opening_scratch_span"}
            )

        def _condition_handles_right_delimiter_token(test: ast.AST) -> bool:
            for inner in ast.walk(test):
                if (
                    isinstance(inner, ast.Call)
                    and isinstance(inner.func, ast.Attribute)
                    and isinstance(inner.func.value, ast.Name)
                    and inner.func.value.id == "helpers"
                    and inner.func.attr == "EndsWithRightDelimiter"
                    and len(inner.args) >= 1
                    and _is_name(inner.args[0], "generated")
                ):
                    return True
                if not isinstance(inner, ast.Compare):
                    continue
                parts = [inner.left, *inner.comparators]
                has_next_token = any(isinstance(part, ast.Name) and part.id == "next_token" for part in parts)
                has_right_delimiter = any(
                    isinstance(part, ast.Name) and part.id in {"RightDelimiter", "SpacedRightDelimiter"}
                    for part in parts
                ) or any(
                    isinstance(part, ast.Constant) and part.value in {">>", " >>"}
                    for part in parts
                )
                if has_next_token and has_right_delimiter:
                    return True
            return False

        def _assigned_state_names(statements: list[ast.stmt]) -> set[str]:
            names: set[str] = set()
            for statement in statements:
                for inner in ast.walk(statement):
                    if isinstance(inner, ast.Assign):
                        for target in inner.targets:
                            if isinstance(target, ast.Name):
                                names.add(target.id)
                    if isinstance(inner, ast.AnnAssign) and isinstance(inner.target, ast.Name):
                        names.add(inner.target.id)
                    if isinstance(inner, ast.AugAssign) and isinstance(inner.target, ast.Name):
                        names.add(inner.target.id)
            return names

        def _state_used_in_conditions(name: str) -> bool:
            for condition_holder in [node for node in ast.walk(tree) if isinstance(node, (ast.If, ast.While))]:
                if any(isinstance(inner, ast.Name) and inner.id == name for inner in ast.walk(condition_holder.test)):
                    return True
            return False

        def _is_literal_int(node: ast.AST | None) -> bool:
            return isinstance(node, ast.Constant) and isinstance(node.value, int)

        def _condition_has_span_counter_threshold(test: ast.AST, span_names: set[str]) -> bool:
            for inner in ast.walk(test):
                if not isinstance(inner, ast.Compare):
                    continue
                names = [
                    part.id
                    for part in [inner.left, *inner.comparators]
                    if isinstance(part, ast.Name) and part.id in span_names
                ]
                if not names:
                    continue
                ints = [
                    part.value
                    for part in [inner.left, *inner.comparators]
                    if isinstance(part, ast.Constant) and isinstance(part.value, int)
                ]
                if ints and max(ints) >= 2:
                    return True
            return False

        def _condition_mentions_final_state(test: ast.AST, final_names: set[str]) -> bool:
            return any(
                isinstance(inner, ast.Name) and inner.id in final_names
                for inner in ast.walk(test)
            )

        def _condition_mentions_is_complete(test: ast.AST) -> bool:
            text = ast.unparse(test)
            if "helpers.IsComplete(generated)" not in text:
                return False
            # Do not treat explicit negated guards as completion branches.
            if "not helpers.IsComplete(generated)" in text:
                return False
            return True

        def _condition_has_stepsleft_threshold(test: ast.AST) -> bool:
            for inner in ast.walk(test):
                if not isinstance(inner, ast.Compare):
                    continue
                parts = [inner.left, *inner.comparators]
                has_stepsleft = any(isinstance(part, ast.Name) and part.id == "stepsLeft" for part in parts)
                if not has_stepsleft:
                    continue
                has_int = any(isinstance(part, ast.Constant) and isinstance(part.value, int) for part in parts)
                if not has_int:
                    continue
                if any(isinstance(op, (ast.Lt, ast.LtE, ast.Gt, ast.GtE)) for op in inner.ops):
                    return True
            return False

        def _condition_has_stepsleft_positive_guard(test: ast.AST) -> bool:
            for inner in ast.walk(test):
                if not isinstance(inner, ast.Compare):
                    continue
                compare_parts = [inner.left, *inner.comparators]
                for left, op, right in zip(compare_parts, inner.ops, compare_parts[1:]):
                    if isinstance(left, ast.Name) and left.id == "stepsLeft" and isinstance(right, ast.Constant):
                        if isinstance(right.value, int):
                            if isinstance(op, ast.Gt) and right.value >= 0:
                                return True
                            if isinstance(op, ast.GtE) and right.value >= 1:
                                return True
                    if isinstance(right, ast.Name) and right.id == "stepsLeft" and isinstance(left, ast.Constant):
                        if isinstance(left.value, int):
                            if isinstance(op, ast.Lt) and left.value >= 0:
                                return True
                            if isinstance(op, ast.LtE) and left.value >= 1:
                                return True
            return False

        def _condition_mentions_open_intent_signal(test: ast.AST) -> bool:
            for inner in ast.walk(test):
                if not isinstance(inner, ast.Name):
                    continue
                lowered = inner.id.lower()
                if any(
                    marker in lowered
                    for marker in (
                        "final",
                        "answer",
                        "ready",
                        "scratch",
                        "span",
                        "closed",
                        "cue",
                        "signal",
                        "pressure",
                        "phase",
                        "state",
                    )
                ):
                    return True
            return False

        def _append_left_is_budget_only(node: ast.AST) -> bool:
            current = node
            while current in parent_map:
                current = parent_map[current]
                if not isinstance(current, ast.If):
                    continue
                test = current.test
                if not _condition_has_stepsleft_threshold(test):
                    continue
                if _condition_has_complete_guard(test):
                    continue
                if _condition_mentions_open_intent_signal(test):
                    continue
                return True
            return False

        def _has_ancestor_while(node: ast.AST) -> bool:
            current = node
            while current in parent_map:
                current = parent_map[current]
                if isinstance(current, ast.While):
                    return True
            return False

        def _has_ancestor_complete_guard(node: ast.AST) -> bool:
            current = node
            while current in parent_map:
                current = parent_map[current]
                if isinstance(current, (ast.If, ast.While)) and _condition_has_complete_guard(current.test):
                    return True
            return False

        def _has_can_constrain_only_ancestor(node: ast.AST) -> bool:
            current = node
            while current in parent_map:
                current = parent_map[current]
                if not isinstance(current, (ast.If, ast.While)):
                    continue
                if not _condition_has_can_constrain_guard(current.test):
                    continue
                return not _condition_has_complete_guard(current.test)
            return False

        def _contains_helper_step(statements: list[ast.stmt]) -> bool:
            for statement in statements:
                for inner in ast.walk(statement):
                    if (
                        isinstance(inner, ast.Call)
                        and isinstance(inner.func, ast.Attribute)
                        and isinstance(inner.func.value, ast.Name)
                        and inner.func.value.id == "helpers"
                        and (
                            inner.func.attr in append_helper_methods
                            or inner.func.attr in constrained_helper_methods
                            or inner.func.attr in unconstrained_helper_methods
                            or inner.func.attr in forced_helper_methods
                        )
                    ):
                        return True
            return False

        def _is_step_consuming_helper_call(node: ast.AST) -> bool:
            return (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "helpers"
                and node.func.attr in step_consuming_helper_methods
            )

        def _first_step_consuming_helper_line(statement: ast.stmt) -> int | None:
            for inner in ast.walk(statement):
                if _is_step_consuming_helper_call(inner):
                    return getattr(inner, "lineno", getattr(statement, "lineno", 0))
            return None

        def _unguarded_step_lines_after_prior(
            statement: ast.stmt,
            *,
            stepsleft_guarded: bool = False,
        ) -> list[int]:
            if isinstance(statement, ast.If):
                guarded = stepsleft_guarded or _condition_has_stepsleft_positive_guard(statement.test)
                lines: list[int] = []
                for branch_statement in [*statement.body, *statement.orelse]:
                    lines.extend(
                        _unguarded_step_lines_after_prior(
                            branch_statement,
                            stepsleft_guarded=guarded,
                        )
                    )
                return lines
            if isinstance(statement, ast.While):
                guarded = stepsleft_guarded or _condition_has_stepsleft_positive_guard(statement.test)
                lines = []
                for branch_statement in [*statement.body, *statement.orelse]:
                    lines.extend(
                        _unguarded_step_lines_after_prior(
                            branch_statement,
                            stepsleft_guarded=guarded,
                        )
                    )
                return lines
            if stepsleft_guarded:
                return []
            line = _first_step_consuming_helper_line(statement)
            return [line] if line is not None else []

        def _collect_sequential_helper_without_budget_guard_lines(
            statements: list[ast.stmt],
        ) -> list[int]:
            lines: list[int] = []
            seen_step_on_this_path = False
            for statement in statements:
                if seen_step_on_this_path:
                    lines.extend(_unguarded_step_lines_after_prior(statement))
                if isinstance(statement, ast.If):
                    lines.extend(_collect_sequential_helper_without_budget_guard_lines(statement.body))
                    lines.extend(_collect_sequential_helper_without_budget_guard_lines(statement.orelse))
                elif isinstance(statement, ast.While):
                    lines.extend(_collect_sequential_helper_without_budget_guard_lines(statement.body))
                    lines.extend(_collect_sequential_helper_without_budget_guard_lines(statement.orelse))
                if _first_step_consuming_helper_line(statement) is not None:
                    seen_step_on_this_path = True
            return lines

        def _contains_break_or_return(statements: list[ast.stmt]) -> bool:
            return any(isinstance(inner, (ast.Break, ast.Return)) for statement in statements for inner in ast.walk(statement))

        def _contains_state_assignment(statements: list[ast.stmt]) -> bool:
            for statement in statements:
                for inner in ast.walk(statement):
                    if isinstance(inner, ast.Assign):
                        for target in inner.targets:
                            if isinstance(target, ast.Name) and target.id not in {
                                "generated", "stepsLeft", "next_token", "new_steps",
                            }:
                                return True
                    if isinstance(inner, ast.AugAssign) and isinstance(inner.target, ast.Name):
                        if inner.target.id not in {"generated", "stepsLeft", "next_token", "new_steps"}:
                            return True
            return False

        def _branch_assigns_opening_state(statements: list[ast.stmt]) -> bool:
            opening_values = {"open", "opening", "nudge", "span", "answer", "final"}
            for statement in statements:
                for inner in ast.walk(statement):
                    targets: list[ast.Name] = []
                    value: ast.AST | None = None
                    if isinstance(inner, ast.Assign):
                        targets.extend(target for target in inner.targets if isinstance(target, ast.Name))
                        value = inner.value
                    elif isinstance(inner, ast.AnnAssign) and isinstance(inner.target, ast.Name):
                        targets.append(inner.target)
                        value = inner.value
                    else:
                        continue
                    for target in targets:
                        target_name = target.id.lower()
                        target_is_phase = target_name in {"phase", "stage", "state", "mode"}
                        target_is_answer_pressure = _is_answer_pressure_name(target.id)
                        if (
                            target_is_phase
                            and isinstance(value, ast.Constant)
                            and isinstance(value.value, str)
                            and value.value.replace("_", "").lower() in opening_values
                        ):
                            return True
                        if (
                            target_is_answer_pressure
                            and isinstance(value, ast.Constant)
                            and value.value in {True, 1}
                        ):
                            return True
            return False

        def _top_level_if_branches(if_node: ast.If) -> list[tuple[ast.AST, list[ast.stmt]]]:
            branches: list[tuple[ast.AST, list[ast.stmt]]] = [(if_node.test, if_node.body)]
            current = if_node
            while len(current.orelse) == 1 and isinstance(current.orelse[0], ast.If):
                current = current.orelse[0]
                branches.append((current.test, current.body))
            if current.orelse:
                branches.append((current.test, current.orelse))
            return branches

        def _collect_nonprogress_if_branches(
            statements: list[ast.stmt],
            *,
            progress_already: bool = False,
        ) -> list[int]:
            offenders: list[int] = []
            progress_seen = progress_already
            for statement in statements:
                if not isinstance(statement, ast.If):
                    if _contains_helper_step([statement]) or _contains_break_or_return([statement]):
                        progress_seen = True
                    continue
                for _branch_test, branch_body in _top_level_if_branches(statement):
                    if (
                        branch_body
                        and not progress_seen
                        and not _contains_helper_step(branch_body)
                        and not _contains_break_or_return(branch_body)
                        and _contains_state_assignment(branch_body)
                    ):
                        offenders.append(getattr(branch_body[0], "lineno", getattr(statement, "lineno", 0)))
                    # Recurse into nested if-chains so one helper call on another path
                    # does not mask a non-consuming sub-branch.
                    offenders.extend(
                        _collect_nonprogress_if_branches(
                            branch_body,
                            progress_already=progress_seen,
                        )
                    )
            return offenders

        for while_node in while_nodes:
            for stmt_idx, statement in enumerate(while_node.body):
                if isinstance(statement, ast.Break):
                    top_level_break_lines.append(getattr(statement, "lineno", 0))
                    continue
                if not isinstance(statement, ast.If):
                    continue
                current_if = statement
                while len(current_if.orelse) == 1 and isinstance(current_if.orelse[0], ast.If):
                    current_if = current_if.orelse[0]
                has_final_else = bool(current_if.orelse)
                if not has_final_else:
                    trailing_statements = while_node.body[stmt_idx + 1 :]
                    trailing_progress = (
                        _contains_helper_step(trailing_statements)
                        or _contains_break_or_return(trailing_statements)
                    )
                    if not trailing_progress:
                        dangling_if_chain_lines.append(getattr(statement, "lineno", 0))
                branches = _top_level_if_branches(statement)
                seen_open_constrain_line = 0
                seen_complete_branch = False
                for branch_test, branch_body in branches:
                    if (
                        require_natural_delimiters
                        and branch_body
                        and not _contains_helper_step(branch_body)
                        and not _contains_break_or_return(branch_body)
                        and _branch_assigns_opening_state(branch_body)
                    ):
                        state_only_open_transition_lines.append(
                            getattr(branch_body[0], "lineno", getattr(statement, "lineno", 0))
                        )
                    if (
                        require_natural_delimiters
                        and branch_body
                        and not _contains_helper_step(branch_body)
                        and _contains_break_or_return(branch_body)
                        and _branch_assigns_opening_state(branch_body)
                    ):
                        phase_break_open_lines.append(
                            getattr(branch_body[0], "lineno", getattr(statement, "lineno", 0))
                        )
                    if (
                        seen_open_constrain_line
                        and _condition_has_complete_guard(branch_test)
                    ):
                        constrain_before_complete_lines.append(seen_open_constrain_line)
                        break
                    if (
                        _condition_has_can_constrain_guard(branch_test)
                        and not _condition_has_complete_guard(branch_test)
                        and not _condition_has_not_complete_guard(branch_test)
                        and not _condition_has_answer_step_cap(branch_test)
                        and _contains_helper_step(branch_body)
                    ):
                        seen_open_constrain_line = getattr(branch_test, "lineno", getattr(statement, "lineno", 0))
                    if (
                        (
                            os.environ.get("CSD_REQUIRE_NATURAL_DELIMITERS", "").strip() in {"1", "true", "True"}
                            or spider_force_single_sql_span
                        )
                        and not seen_complete_branch
                        and _condition_has_not_can_constrain_guard(branch_test)
                        and _contains_break_or_return(branch_body)
                    ):
                        premature_not_can_constrain_lines.append(
                            getattr(branch_test, "lineno", getattr(statement, "lineno", 0))
                        )
                    if _condition_has_complete_guard(branch_test):
                        seen_complete_branch = True
                for _test, branch_body in _top_level_if_branches(statement):
                    if (
                        branch_body
                        and not _contains_helper_step(branch_body)
                        and not _contains_break_or_return(branch_body)
                        and _contains_state_assignment(branch_body)
                    ):
                        nondecreasing_else_lines.append(getattr(branch_body[0], "lineno", getattr(statement, "lineno", 0)))
                nondecreasing_else_lines.extend(_collect_nonprogress_if_branches(statement.body))

        if os.environ.get("CSD_REQUIRE_NATURAL_DELIMITERS", "").strip() in {"1", "true", "True"} or spider_force_single_sql_span:
            for if_node in [node for node in ast.walk(tree) if isinstance(node, ast.If)]:
                seen_complete_branch = False
                for branch_test, branch_body in _top_level_if_branches(if_node):
                    if (
                        not seen_complete_branch
                        and _condition_has_not_can_constrain_guard(branch_test)
                        and _contains_break_or_return(branch_body)
                    ):
                        premature_not_can_constrain_lines.append(
                            getattr(branch_test, "lineno", getattr(if_node, "lineno", 0))
                        )
                    if _condition_has_complete_guard(branch_test):
                        seen_complete_branch = True

        for node in ast.walk(tree):
            if isinstance(node, ast.Expr):
                value = node.value
                # Bare expression statements such as `inside` are usually truncated code
                # that survives Python parsing but fails in transpilation/verification.
                if (
                    isinstance(value, ast.Name)
                    or isinstance(value, ast.Attribute)
                    or isinstance(value, ast.Subscript)
                    or (
                        isinstance(value, ast.Constant)
                        and not isinstance(value.value, str)
                    )
                ):
                    stray_expression_lines.append(getattr(node, "lineno", 0))
            if isinstance(node, ast.Continue):
                continue_lines.append(getattr(node, "lineno", 0))
            if (
                isinstance(node, ast.Subscript)
                and isinstance(node.slice, ast.UnaryOp)
                and isinstance(node.slice.op, ast.USub)
            ):
                negative_index_lines.append(getattr(node, "lineno", 0))
            if isinstance(node, ast.If):
                if require_natural_delimiters and _has_ancestor_while(node):
                    for _branch_test, branch_body in _top_level_if_branches(node):
                        if (
                            branch_body
                            and not _contains_helper_step(branch_body)
                            and _contains_break_or_return(branch_body)
                            and _branch_assigns_opening_state(branch_body)
                        ):
                            phase_break_open_lines.append(
                                getattr(branch_body[0], "lineno", getattr(node, "lineno", 0))
                            )
                if _condition_mentions_is_complete(node.test):
                    for statement in node.body:
                        for inner in ast.walk(statement):
                            if (
                                isinstance(inner, ast.Call)
                                and isinstance(inner.func, ast.Attribute)
                                and isinstance(inner.func.value, ast.Name)
                                and inner.func.value.id == "helpers"
                                and inner.func.attr in {"AppendConstrainedStep", "ConstrainedStep"}
                            ):
                                complete_branch_constrained_lines.append(getattr(inner, "lineno", 0))
                if_count += 1
                if (
                    isinstance(parent_map.get(node), ast.While)
                    and node.orelse
                    and not (len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If))
                ):
                    if (
                        not _contains_helper_step(node.orelse)
                        and not _contains_break_or_return(node.orelse)
                        and _contains_state_assignment(node.orelse)
                    ):
                        nondecreasing_else_lines.append(getattr(node.orelse[0], "lineno", getattr(node, "lineno", 0)))
            if isinstance(node, ast.Return):
                has_return = True
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if (
                    node.func.attr == "join"
                    and len(node.args) == 1
                    and _is_generated_token_sequence(node.args[0])
                ):
                    generated_join_lines.append(getattr(node, "lineno", 0))
                if isinstance(node.func.value, ast.Name) and node.func.value.id == "helpers":
                    attr = node.func.attr
                    line = getattr(node, "lineno", 0)
                    if first_helper_call_line == 0 or (line and line < first_helper_call_line):
                        first_helper_call_line = line
                        first_helper_call_attr = attr
                    if attr in prompt_arg_required_methods and not node.keywords:
                        if len(node.args) == 0 or not _is_name(node.args[0], "prompt"):
                            bad_prompt_arg_calls.append((attr, line))
                    helper_calls.add(attr)
                    if node.keywords:
                        keyword_helper_calls.add(attr)
                    if attr in OLD_API:
                        old_api_calls.add(attr)
                    elif attr in REPAIR_HELPERS:
                        repair_helper_calls.add(attr)
                    elif attr not in self.ALLOWED_HELPER_METHODS:
                        unsupported_helper_calls.add(attr)
                        if attr in self.ALLOWED_PARSER_METHODS:
                            helper_parser_confusions.add(attr)
                    if attr in constrained_helper_methods:
                        constrained_step_calls += 1
                        if attr in {"AdaptiveConstrainedStep", "GroupBoostedConstrainedStep", "PenalizedConstrainedStep"}:
                            uses_split_prefix_policy = True
                        if (
                            os.environ.get("CSD_REQUIRE_NATURAL_DELIMITERS", "").strip() in {"1", "true", "True"}
                            and attr in {"ConstrainedStep", "AppendConstrainedStep"}
                        ):
                            natural_plain_constrained_lines.append(getattr(node, "lineno", 0))
                        if attr in {"ConstrainedOrRightDelimiterStep", "AppendConstrainedOrRightDelimiterStep"}:
                            uses_natural_right_delimiter = True
                            if (
                                os.environ.get("CSD_REQUIRE_NATURAL_DELIMITERS", "").strip() in {"1", "true", "True"}
                                and _has_can_constrain_only_ancestor(node)
                            ):
                                natural_completion_blind_right_helper_lines.append(getattr(node, "lineno", 0))
                        if attr in {"ConstrainedStep", "AppendConstrainedStep"}:
                            current = node
                            guarded = False
                            while current in parent_map:
                                current = parent_map[current]
                                if isinstance(current, (ast.If, ast.While)) and _condition_has_constrain_guard(current.test):
                                    guarded = True
                                    break
                            if not guarded:
                                unguarded_constrained_calls.add(attr)
                    if attr in split_prefix_step_methods:
                        split_prefix_step_calls += 1
                        uses_split_prefix_policy = True
                    if attr in forced_helper_methods:
                        forced_token_calls += 1
                    if attr in unconstrained_helper_methods:
                        unconstrained_calls += 1
                        unconstrained_lines.append(getattr(node, "lineno", 0))
                        if (
                            require_natural_delimiters
                            and attr in {"UnconstrainedStep", "AppendUnconstrainedStep"}
                        ):
                            current = node
                            while current in parent_map:
                                current = parent_map[current]
                                if isinstance(current, ast.While):
                                    break
                                if (
                                    isinstance(current, ast.If)
                                    and _condition_is_opening_context(current.test)
                                    and _descendant_is_in_if_body(node, current)
                                ):
                                    natural_open_plain_fallback_lines.append(getattr(node, "lineno", 0))
                                    break
                        if attr in {
                            "UnconstrainedAllowLeftDelimiterStep",
                            "UnconstrainedBiasLeftDelimiterStep",
                            "UnconstrainedNudgeLeftDelimiterStep",
                            "AppendUnconstrainedAllowLeftDelimiterStep",
                            "AppendUnconstrainedNudgeLeftDelimiterStep",
                        }:
                            uses_natural_left_delimiter = True
                    if attr == "OpenConstrainedSpan":
                        uses_split_prefix_policy = True
                        emits_left_delimiter = True
                        left_delimiter_lines.append(getattr(node, "lineno", 0))
                    elif attr == "CloseConstrainedSpan":
                        uses_split_prefix_policy = True
                        emits_right_delimiter = True
                        right_delimiter_lines.append(getattr(node, "lineno", 0))
                    elif attr == "AppendConstrainedToken":
                        uses_split_prefix_policy = True
                    if attr == "AppendLeftDelimiter":
                        emits_left_delimiter = True
                        left_delimiter_lines.append(getattr(node, "lineno", 0))
                        forced_left_delimiter_lines.append(getattr(node, "lineno", 0))
                        if _append_left_is_budget_only(node):
                            budget_only_open_lines.append(getattr(node, "lineno", 0))
                        if not _has_ancestor_while(node):
                            delimiter_calls_outside_loop.add(attr)
                    elif attr == "AppendRightDelimiter":
                        emits_right_delimiter = True
                        right_delimiter_lines.append(getattr(node, "lineno", 0))
                        forced_right_delimiter_lines.append(getattr(node, "lineno", 0))
                        if not _has_ancestor_while(node):
                            delimiter_calls_outside_loop.add(attr)
                        if not _has_ancestor_complete_guard(node):
                            unguarded_right_delimiter_calls.add(attr)
                    elif attr == "ForcedTokenStep" and len(node.args) >= 3:
                        if _is_name(node.args[2], "LeftDelimiter"):
                            emits_left_delimiter = True
                            left_delimiter_lines.append(getattr(node, "lineno", 0))
                            forced_left_delimiter_lines.append(getattr(node, "lineno", 0))
                            if not _has_ancestor_while(node):
                                delimiter_calls_outside_loop.add(attr)
                        elif _is_name(node.args[2], "RightDelimiter"):
                            emits_right_delimiter = True
                            right_delimiter_lines.append(getattr(node, "lineno", 0))
                            forced_right_delimiter_lines.append(getattr(node, "lineno", 0))
                            if not _has_ancestor_while(node):
                                delimiter_calls_outside_loop.add(attr)
                            if not _has_ancestor_complete_guard(node):
                                unguarded_right_delimiter_calls.add("ForcedTokenStep(RightDelimiter)")
                    elif attr == "AppendForcedToken" and len(node.args) >= 2:
                        if _is_name(node.args[1], "LeftDelimiter"):
                            emits_left_delimiter = True
                            left_delimiter_lines.append(getattr(node, "lineno", 0))
                            forced_left_delimiter_lines.append(getattr(node, "lineno", 0))
                            if not _has_ancestor_while(node):
                                delimiter_calls_outside_loop.add(attr)
                        elif _is_name(node.args[1], "RightDelimiter"):
                            emits_right_delimiter = True
                            right_delimiter_lines.append(getattr(node, "lineno", 0))
                            forced_right_delimiter_lines.append(getattr(node, "lineno", 0))
                            if not _has_ancestor_while(node):
                                delimiter_calls_outside_loop.add(attr)
                            if not _has_ancestor_complete_guard(node):
                                unguarded_right_delimiter_calls.add("AppendForcedToken(RightDelimiter)")
                if isinstance(node.func.value, ast.Name) and node.func.value.id == "parser":
                    attr = node.func.attr
                    if attr not in self.ALLOWED_PARSER_METHODS:
                        unsupported_parser_calls.add(attr)
                    if (
                        node.args
                        and isinstance(node.args[0], ast.Name)
                        and node.args[0].id == "generated"
                    ):
                        parser_on_generated_methods.add(attr)
                if isinstance(node.func.value, ast.Name) and node.func.value.id == "generated":
                    attr = node.func.attr
                    if attr in {"startswith", "endswith", "strip", "lstrip", "rstrip"}:
                        generated_string_methods.add(attr)
                if (
                    isinstance(node.func.value, ast.Call)
                    and isinstance(node.func.value.func, ast.Attribute)
                    and isinstance(node.func.value.func.value, ast.Name)
                    and node.func.value.func.value.id == "helpers"
                    and node.func.value.func.attr == "LongestValidSuffix"
                ):
                    attr = node.func.attr
                    if attr in {"startswith", "endswith", "strip", "lstrip", "rstrip"}:
                        suffix_string_methods.add(attr)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print":
                print_calls += 1
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                attr = node.func.attr
                if attr in {
                    "AppendUnconstrainedNudgeLeftDelimiterStep",
                    "UnconstrainedNudgeLeftDelimiterStep",
                    "AppendUnconstrainedAllowLeftDelimiterStep",
                    "UnconstrainedAllowLeftDelimiterStep",
                }:
                    current = node
                    while current in parent_map:
                        current = parent_map[current]
                        if isinstance(current, ast.If) and _condition_has_low_reason_final_threshold(current.test):
                            low_reason_nudge_lines.append(getattr(node, "lineno", 0))
                            break
            if isinstance(node, ast.Compare):
                if any(isinstance(part, ast.Constant) and isinstance(part.value, float) for part in [node.left, *node.comparators]):
                    float_comparisons.append(getattr(node, "lineno", 0))
                if min_required_reason_steps > 0:
                    compare_parts = [node.left, *node.comparators]
                    for left, right in zip(compare_parts, compare_parts[1:]):
                        if (
                            isinstance(left, ast.Name)
                            and ("reason" in left.id.lower() or "setup" in left.id.lower() or "prelude" in left.id.lower())
                            and isinstance(right, ast.Constant)
                            and isinstance(right.value, int)
                            and right.value < min_required_reason_steps
                        ):
                            insufficient_reason_budget_lines.append(getattr(node, "lineno", 0))
                        if (
                            isinstance(right, ast.Name)
                            and ("reason" in right.id.lower() or "setup" in right.id.lower() or "prelude" in right.id.lower())
                            and isinstance(left, ast.Constant)
                            and isinstance(left.value, int)
                            and left.value < min_required_reason_steps
                        ):
                            insufficient_reason_budget_lines.append(getattr(node, "lineno", 0))
                if min_required_answer_steps > 0:
                    compare_parts = [node.left, *node.comparators]
                    for left, right in zip(compare_parts, compare_parts[1:]):
                        if (
                            isinstance(left, ast.Name)
                            and ("answer" in left.id.lower() or "constrained" in left.id.lower())
                            and isinstance(right, ast.Constant)
                            and isinstance(right.value, int)
                            and right.value < min_required_answer_steps
                        ):
                            insufficient_answer_budget_lines.append(getattr(node, "lineno", 0))
                if spider_force_single_sql_span:
                    compare_parts = [node.left, *node.comparators]
                    for left, right in zip(compare_parts, compare_parts[1:]):
                        if (
                            isinstance(left, ast.Name)
                            and _is_spider_freeform_state_name(left.id)
                            and isinstance(right, ast.Constant)
                            and isinstance(right.value, int)
                            and right.value > 3
                        ):
                            spider_long_freeform_lines.append(getattr(node, "lineno", 0))
                        if (
                            isinstance(right, ast.Name)
                            and _is_spider_freeform_state_name(right.id)
                            and isinstance(left, ast.Constant)
                            and isinstance(left.value, int)
                            and left.value > 3
                        ):
                            spider_long_freeform_lines.append(getattr(node, "lineno", 0))
                        if (
                            isinstance(right, ast.Name)
                            and ("answer" in right.id.lower() or "constrained" in right.id.lower())
                            and isinstance(left, ast.Constant)
                            and isinstance(left.value, int)
                            and left.value < min_required_answer_steps
                        ):
                            insufficient_answer_budget_lines.append(getattr(node, "lineno", 0))
            if (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and isinstance(node.value.func.value, ast.Name)
                and node.value.func.value.id == "helpers"
                and node.value.func.attr == "ForcedTokenStep"
            ):
                bare_forced_token_calls += 1
            if (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and isinstance(node.value.func.value, ast.Name)
                and node.value.func.value.id == "helpers"
                and node.value.func.attr in append_helper_methods
            ):
                bare_append_helper_calls += 1
            if isinstance(node, ast.Assign):
                if (
                    isinstance(node.value, ast.Constant)
                    and node.value.value is None
                ):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and "checkpoint" in target.id.lower():
                            none_checkpoint_assign_lines.append(getattr(node, "lineno", 0))
                final_ready_assignment = any(
                    isinstance(target, ast.Name)
                    and target.id.lower() in {
                        "final_ready",
                        "answer_ready",
                        "cue_signal",
                        "final_cue",
                    }
                    for target in node.targets
                )
                if (
                    final_ready_assignment
                    and isinstance(node.value, ast.Constant)
                    and node.value.value == 1
                ):
                    current = node
                    while current in parent_map:
                        current = parent_map[current]
                        if isinstance(current, ast.If) and _condition_has_low_reason_final_threshold(current.test):
                            low_final_ready_lines.append(getattr(node, "lineno", 0))
                            break
                answer_pressure_assignment = any(
                    isinstance(target, ast.Name)
                    and _is_answer_pressure_name(target.id)
                    for target in node.targets
                )
                if (
                    (final_ready_assignment or answer_pressure_assignment)
                    and isinstance(node.value, ast.Constant)
                    and node.value.value in {True, 1}
                ):
                    current = node
                    while current in parent_map:
                        current = parent_map[current]
                        if (
                            isinstance(current, ast.If)
                            and _condition_has_parser_readiness_trigger(current.test)
                            and _condition_has_low_setup_or_reason_threshold(current.test, threshold=40)
                        ):
                            parser_readiness_early_open_lines.append(getattr(node, "lineno", 0))
                            break
                if answer_pressure_assignment:
                    if (
                        isinstance(node.value, ast.Constant)
                        and node.value.value is True
                    ):
                        current = node
                        while current in parent_map:
                            current = parent_map[current]
                            if (
                                isinstance(current, ast.If)
                                and _condition_has_tiny_remaining_budget_trigger(current.test)
                            ):
                                late_budget_answer_pressure_lines.append(getattr(node, "lineno", 0))
                                break
                    elif _condition_has_tiny_remaining_budget_trigger(node.value):
                        late_budget_answer_pressure_lines.append(getattr(node, "lineno", 0))
                if len(node.targets) == 1 and isinstance(node.targets[0], ast.Tuple):
                    if not isinstance(node.value, (ast.Call, ast.Tuple)):
                        malformed_tuple_assignment_lines.append(getattr(node, "lineno", 0))
                if (
                    len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == "generated"
                    and isinstance(node.value, ast.BinOp)
                    and isinstance(node.value.op, ast.Add)
                    and isinstance(node.value.left, ast.Name)
                    and node.value.left.id == "generated"
                    and isinstance(node.value.right, ast.List)
                ):
                    appends_generated = True
                if (
                    len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Attribute)
                    and isinstance(node.value.func.value, ast.Name)
                    and node.value.func.value.id == "helpers"
                    and node.value.func.attr in generated_updating_helper_methods
                ):
                    append_helper_wrong_targets.add(node.value.func.attr)
                if (
                    len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Tuple)
                    and isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Attribute)
                    and isinstance(node.value.func.value, ast.Name)
                    and node.value.func.value.id == "helpers"
                    and node.value.func.attr in generated_updating_helper_methods
                ):
                    target_names = {
                        elt.id
                        for elt in node.targets[0].elts
                        if isinstance(elt, ast.Name)
                    }
                    if "generated" in target_names:
                        appends_generated = True
                    else:
                        append_helper_wrong_targets.add(node.value.func.attr)
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "stepsLeft":
                        if not (isinstance(node.value, ast.Name) and node.value.id == "new_steps"):
                            manual_stepsleft_mutations.append(getattr(node, "lineno", 0))
                    if isinstance(target, ast.Name) and target.id == "remainingSteps":
                        assigns_remaining_steps = True
                    if isinstance(target, ast.Name) and target.id not in {
                        "generated", "stepsLeft", "next_token", "new_steps",
                    }:
                        extra_state.add(target.id)
                        if _is_fixed_phase_quota_name(target.id) and _is_bad_fixed_phase_quota(target.id, node.value):
                            fixed_phase_quota_lines.append(getattr(node, "lineno", 0))
                        if any(isinstance(inner, ast.Constant) and isinstance(inner.value, float) for inner in ast.walk(node.value)):
                            mutable_float_state.add(target.id)
                        if (
                            min_required_reason_steps > 0
                            and _is_reason_budget_name(target.id)
                            and isinstance(node.value, ast.Constant)
                            and isinstance(node.value.value, int)
                            and node.value.value < min_required_reason_steps
                        ):
                            insufficient_reason_budget_lines.append(getattr(node, "lineno", 0))
                        if (
                            min_required_answer_steps > 0
                            and _is_answer_budget_name(target.id)
                            and isinstance(node.value, ast.Constant)
                            and isinstance(node.value.value, int)
                            and node.value.value < min_required_answer_steps
                        ):
                            insufficient_answer_budget_lines.append(getattr(node, "lineno", 0))
                        if (
                            _is_reason_budget_name(target.id)
                            and isinstance(node.value, ast.Constant)
                            and isinstance(node.value.value, int)
                            and node.value.value <= 1
                        ):
                            trivial_reason_budget_lines.append(getattr(node, "lineno", 0))
                        if (
                            _is_answer_budget_name(target.id)
                            and isinstance(node.value, ast.Constant)
                            and isinstance(node.value.value, int)
                            and node.value.value <= 2
                        ):
                            trivial_answer_budget_lines.append(getattr(node, "lineno", 0))
                        if (
                            spider_force_single_sql_span
                            and _is_spider_freeform_state_name(target.id)
                            and isinstance(node.value, ast.Constant)
                            and isinstance(node.value.value, int)
                            and node.value.value > 3
                        ):
                            spider_long_freeform_lines.append(getattr(node, "lineno", 0))
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                if node.target.id == "stepsLeft":
                    manual_stepsleft_mutations.append(getattr(node, "lineno", 0))
                if node.target.id == "remainingSteps":
                    assigns_remaining_steps = True
                if node.target.id not in {"generated", "stepsLeft", "next_token", "new_steps"}:
                    extra_state.add(node.target.id)
                    if _is_fixed_phase_quota_name(node.target.id) and _is_bad_fixed_phase_quota(node.target.id, node.value):
                        fixed_phase_quota_lines.append(getattr(node, "lineno", 0))
                    if node.value is not None and any(
                        isinstance(inner, ast.Constant) and isinstance(inner.value, float)
                        for inner in ast.walk(node.value)
                    ):
                        mutable_float_state.add(node.target.id)
                    if (
                        min_required_reason_steps > 0
                        and _is_reason_budget_name(node.target.id)
                        and isinstance(node.value, ast.Constant)
                        and isinstance(node.value.value, int)
                        and node.value.value < min_required_reason_steps
                    ):
                        insufficient_reason_budget_lines.append(getattr(node, "lineno", 0))
                    if (
                        min_required_answer_steps > 0
                        and _is_answer_budget_name(node.target.id)
                        and isinstance(node.value, ast.Constant)
                        and isinstance(node.value.value, int)
                        and node.value.value < min_required_answer_steps
                    ):
                        insufficient_answer_budget_lines.append(getattr(node, "lineno", 0))
                    if (
                        _is_reason_budget_name(node.target.id)
                        and isinstance(node.value, ast.Constant)
                        and isinstance(node.value.value, int)
                        and node.value.value <= 1
                    ):
                        trivial_reason_budget_lines.append(getattr(node, "lineno", 0))
                    if (
                        _is_answer_budget_name(node.target.id)
                        and isinstance(node.value, ast.Constant)
                        and isinstance(node.value.value, int)
                        and node.value.value <= 2
                    ):
                        trivial_answer_budget_lines.append(getattr(node, "lineno", 0))
                    if (
                        spider_force_single_sql_span
                        and _is_spider_freeform_state_name(node.target.id)
                        and isinstance(node.value, ast.Constant)
                        and isinstance(node.value.value, int)
                        and node.value.value > 3
                    ):
                        spider_long_freeform_lines.append(getattr(node, "lineno", 0))
            if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
                if node.target.id == "stepsLeft":
                    manual_stepsleft_mutations.append(getattr(node, "lineno", 0))
                if node.target.id not in {"generated", "stepsLeft", "next_token", "new_steps"}:
                    extra_state.add(node.target.id)
                if (
                    node.target.id not in {"generated", "stepsLeft", "next_token", "new_steps"}
                    and any(isinstance(inner, ast.Constant) and isinstance(inner.value, float) for inner in ast.walk(node.value))
                ):
                    mutable_float_state.add(node.target.id)

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_strategy":
                sequential_helper_without_budget_lines = (
                    _collect_sequential_helper_without_budget_guard_lines(node.body)
                )
                break

        if old_api_calls:
            return (
                "The body uses the old delimiter-based API which has been replaced. "
                "Remove these calls: " + ", ".join(sorted(old_api_calls)) + ". "
                "Prefer helpers.AppendUnconstrainedStep, helpers.AppendConstrainedStep, "
                "helpers.AppendConstrainedOrRightDelimiterStep, "
                "and helpers.AppendLeftDelimiter/helpers.AppendRightDelimiter instead."
            )
        if repair_helper_calls:
            return (
                "Do not use repair/salvage helpers in synthesized strategies. "
                "They create fallback-style control flow instead of a standard final delimiter span. "
                "Remove these calls: " + ", ".join(sorted(repair_helper_calls)) + "."
            )
        if helper_parser_confusions:
            return (
                "These names are parser methods, not helper methods: "
                + ", ".join(sorted(helper_parser_confusions))
                + ". Call them on `parser` using the grammar suffix, e.g. "
                  "`parser.IsCompletePrefix(helpers.LongestValidSuffix(generated))` or "
                  "`parser.ValidContinuationCount(helpers.LongestValidSuffix(generated))`."
            )
        if unsupported_parser_calls:
            return (
                "The body calls parser methods that do not exist in the supported synthesis API: "
                + ", ".join(sorted(unsupported_parser_calls)) + "."
            )
        if parser_on_generated_methods:
            return (
                "Do not call parser methods directly on 'generated'. "
                "Use helpers.LongestValidSuffix(generated) first to get the grammar-relevant suffix. "
                "Offending methods: " + ", ".join(sorted(parser_on_generated_methods)) + "."
            )
        if generated_join_lines:
            return (
                "Do not join `generated` into a Python string or inspect it with substring/string logic; "
                "`generated` is a token list and joined-string reasoning is not verifier-friendly. "
                "Track phase/counters explicitly and use helper predicates such as "
                "`helpers.EndsWithLeftDelimiter(generated)` and `helpers.EndsWithRightDelimiter(generated)`. "
                f"First joined-string use is near line {generated_join_lines[0]}."
            )
        if generated_string_methods:
            return (
                "Do not treat 'generated' like a Python string. It is a list of tokens, so string methods like "
                + ", ".join(sorted(generated_string_methods))
                + " are invalid here. Track delimiter/phase state explicitly and emit delimiters with "
                  "`helpers.AppendLeftDelimiter(...)` / `helpers.AppendRightDelimiter(...)` instead."
            )
        if suffix_string_methods:
            return (
                "Do not treat `helpers.LongestValidSuffix(generated)` like a Python string. It also returns a list of "
                "tokens, so string methods like "
                + ", ".join(sorted(suffix_string_methods))
                + " are invalid here. Use parser predicates on that suffix instead, such as "
                  "`parser.IsCompletePrefix(helpers.LongestValidSuffix(generated))`."
            )
        if unsupported_helper_calls:
            return (
                "The body calls helper methods that do not exist in the supported synthesis API: "
                + ", ".join(sorted(unsupported_helper_calls)) + "."
            )
        if bad_prompt_arg_calls:
            call_name, line = bad_prompt_arg_calls[0]
            return (
                "Helper calls that require context must pass the function input `prompt` as the first argument. "
                f"Found `{call_name}` with a non-`prompt` first argument near line {line}."
            )
        total_step_calls = constrained_step_calls + forced_token_calls + unconstrained_calls + split_prefix_step_calls
        if total_step_calls == 0:
            return (
                "The body must call at least one step method "
                "(AppendConstrainedStep, AppendConstrainedOrRightDelimiterStep, OpenConstrainedSpan, "
                "CloseConstrainedSpan, AdaptiveConstrainedStep, AppendLeftDelimiter, "
                "AppendUnconstrainedStep, UnconstrainedNudgeLeftDelimiterStep, ForcedTokenStep, etc.)."
            )
        if constrained_step_calls == 0:
            return (
                "The body must include at least one constrained step "
                "(helpers.ConstrainedStep, helpers.ConstrainedOrRightDelimiterStep, "
                "`helpers.AdaptiveConstrainedStep(...)`, or equivalent constrained-token logic) "
                "to produce grammar-valid answer content."
            )
        if (not require_natural_delimiters) and (not emits_left_delimiter or not emits_right_delimiter):
            missing = []
            if not emits_left_delimiter:
                missing.append("LeftDelimiter")
            if not emits_right_delimiter:
                missing.append("RightDelimiter")
            return (
                "The body must emit both LeftDelimiter and RightDelimiter "
                "(via helpers.ForcedTokenStep, helpers.AppendForcedToken, helpers.AppendLeftDelimiter, "
                "or helpers.AppendRightDelimiter) so the evaluator can extract the answer from << ... >> in the output. "
                "If LeftDelimiter is missing, include the executable assignment "
                "`generated, stepsLeft = helpers.AppendLeftDelimiter(generated, stepsLeft)`. "
                "If RightDelimiter is missing, include the executable assignment "
                "`generated, stepsLeft = helpers.AppendRightDelimiter(generated, stepsLeft)`. "
                "These calls must be real statements, not rationale text or comments. "
                "Missing: " + ", ".join(missing) + "."
            )
        if has_return or assigns_remaining_steps:
            return (
                "Do not return from the generated body or assign `remainingSteps`; "
                "the surrounding template sets `remainingSteps = stepsLeft` and returns for you."
            )
        if print_calls:
            return "Do not call print() in the generated strategy body; communicate through generated tokens only."
        if keyword_helper_calls:
            return (
                "Helper calls must use positional arguments only because the transpiler does not lower helper keyword arguments. "
                "Remove keywords from: " + ", ".join(sorted(keyword_helper_calls)) + "."
            )
        if mutable_float_state or float_comparisons:
            return (
                "Do not use mutable float local state or float comparisons in strategy control flow; "
                "the Dafny lowering is proof-hostile. Use integer counters for state."
            )
        if bad_bias_helper_lines:
            return (
                "helpers.UnconstrainedBiasLeftDelimiterStep requires a literal positive float bias, "
                "e.g. `helpers.UnconstrainedBiasLeftDelimiterStep(prompt, generated, 5.0, stepsLeft)`. "
                "Do not store the bias in an int variable such as `biasStrength = 3`. "
                f"First invalid bias argument is near line {bad_bias_helper_lines[0]}."
            )
        if negative_index_lines:
            return (
                "Do not use negative list indexing such as `generated[-1]`; the verifier cannot prove "
                "that it is in range. Use `generated[len(generated) - 1]` under an explicit "
                "`if len(generated) > 0:` guard instead. First negative index is near line "
                f"{negative_index_lines[0]}."
            )
        if require_natural_delimiters:
            split_prefix_open = "OpenConstrainedSpan" in helper_calls
            split_prefix_close = "CloseConstrainedSpan" in helper_calls
            split_prefix_append = "AppendConstrainedToken" in helper_calls
            split_prefix_constrained = any(
                name in helper_calls
                for name in {
                    "AdaptiveConstrainedStep",
                    "GroupBoostedConstrainedStep",
                    "PenalizedConstrainedStep",
                }
            )
            uses_split_prefix_family = (
                uses_split_prefix_policy
                and split_prefix_open
                and split_prefix_close
                and split_prefix_append
            )
            span_counter_state = {
                name for name in extra_state if _is_verified_span_counter_name(name)
            }
            open_span_state = {
                name for name in extra_state if _is_open_span_state_name(name)
            }
            right_delimiter_span_counter_updates: set[str] = set()
            for if_node in [node for node in ast.walk(tree) if isinstance(node, ast.If)]:
                if _condition_handles_right_delimiter_token(if_node.test):
                    right_delimiter_span_counter_updates.update(
                        _assigned_state_names(if_node.body) & span_counter_state
                    )
            if uses_split_prefix_policy and not uses_split_prefix_family:
                return (
                    "Split-prefix GSM policies must use the full helper family: "
                    "`OpenConstrainedSpan(...)`, `AppendConstrainedToken(...)`, and "
                    "`CloseConstrainedSpan(...)`, with durable local `inside_constrained` / "
                    "`current_constrained` state."
                )
            if uses_split_prefix_family and not split_prefix_constrained:
                return (
                    "Split-prefix GSM policies must include a constrained-token chooser such as "
                    "`AdaptiveConstrainedStep(...)`, `GroupBoostedConstrainedStep(...)`, or "
                    "`PenalizedConstrainedStep(...)` before `AppendConstrainedToken(...)`."
                )
            if (not uses_split_prefix_family) and (not uses_natural_left_delimiter or not uses_natural_right_delimiter):
                missing = []
                if not uses_natural_left_delimiter:
                    missing.append("UnconstrainedAllowLeftDelimiterStep/UnconstrainedNudgeLeftDelimiterStep")
                if not uses_natural_right_delimiter:
                    missing.append("ConstrainedOrRightDelimiterStep or AppendConstrainedOrRightDelimiterStep")
                return (
                    "This GSM run requires natural delimiter decisions rather than forced delimiter phases. "
                    "Use `helpers.UnconstrainedAllowLeftDelimiterStep(...)` or "
                    "`helpers.UnconstrainedNudgeLeftDelimiterStep(...)` after the answer-ready signal so "
                    "the LM may emit `<<` naturally, then use `helpers.ConstrainedOrRightDelimiterStep(...)` "
                    "inside the constrained span so the LM may emit `>>` naturally only after parser completion. "
                    "Missing: " + ", ".join(missing) + "."
                )
            if forced_left_delimiter_lines or forced_right_delimiter_lines:
                return (
                    "This GSM run is configured for natural delimiter decisions, so do not force delimiters "
                    "with AppendLeftDelimiter, AppendRightDelimiter, AppendForcedToken, or ForcedTokenStep. "
                    "Let `UnconstrainedAllowLeftDelimiterStep` or `UnconstrainedNudgeLeftDelimiterStep` choose `<<` / ` <<`, and let "
                    "`ConstrainedOrRightDelimiterStep` choose `>>` after parser completion. "
                    f"First forced delimiter is near line {(forced_left_delimiter_lines or forced_right_delimiter_lines)[0]}."
                )
            if not open_span_state or not any(_state_used_in_conditions(name) for name in open_span_state):
                return (
                    "Natural delimiter mode needs durable open-span state such as `phase`, `inside_span`, "
                    "or `in_span` used in branch conditions. `helpers.EndsWithLeftDelimiter(generated)` "
                    "is only true immediately after the `<<` token; if you use it as the whole span-mode "
                    "condition, the strategy leaves constrained mode after one token and may emit repeated "
                    "`<<` delimiters. Set the span state when `EndsWithLeftDelimiter` becomes true, keep "
                    "using `AppendConstrainedOrRightDelimiterStep` while that state is active, and clear it "
                    "after `EndsWithRightDelimiter`. Split-prefix policies may instead use "
                    "`inside_constrained` / `current_constrained` state with "
                    "`OpenConstrainedSpan` / `CloseConstrainedSpan`."
                )
            if natural_plain_constrained_lines and not uses_split_prefix_family:
                return (
                    "In GSM natural-delimiter mode, do not use plain `ConstrainedStep` or "
                    "`helpers.AppendConstrainedStep(...)` inside a span. Use "
                    "`helpers.ConstrainedOrRightDelimiterStep(...)` or "
                    "`helpers.AppendConstrainedOrRightDelimiterStep(...)` for constrained-span "
                    "tokens so completion can naturally close with `>>` instead of getting stuck in an "
                    f"open span. First plain constrained call is near line {natural_plain_constrained_lines[0]}."
                )
            natural_rollback_helpers = sorted(helper_calls & {"RestoreCheckpoint", "RestoreIfDead"})
            if natural_rollback_helpers:
                return (
                    "Avoid checkpoint rollback helpers in GSM natural-delimiter strategies. "
                    "`RestoreCheckpoint`/`RestoreIfDead` require extra checkpoint-length invariants and often "
                    "break the simple `|generated| + stepsLeft <= maxSteps` proof. Prefer one "
                    "step-consuming helper call per loop iteration, then use `EndsWithRightDelimiter`, "
                    "`IsDead`, `IsComplete`, and durable phase state to decide the next iteration. "
                    "Rollback helpers used: " + ", ".join(natural_rollback_helpers) + "."
                )
            if natural_completion_blind_right_helper_lines:
                return (
                    "In GSM natural-delimiter mode, do not gate "
                    "`AppendConstrainedOrRightDelimiterStep` behind only "
                    "`helpers.CanConstrain(generated)`. `CanConstrain` becomes false exactly when the "
                    "suffix is complete and `>>` is allowed, so that branch exits with an open `<< ...` "
                    "span. Either call the right-closure helper unconditionally while durable open-span "
                    "state is active, or guard it with "
                    "`helpers.IsComplete(generated) or helpers.CanConstrain(generated)`. First "
                    f"completion-blind right helper is near line {natural_completion_blind_right_helper_lines[0]}."
                )
            if premature_not_can_constrain_lines:
                return (
                    "`helpers.CanConstrain(generated)` is false when the current grammar suffix is already "
                    "complete, so a `not helpers.CanConstrain(generated): break` branch before an "
                    "`helpers.IsComplete(generated)` close/exit branch exits with an unclosed `<< ...` span. "
                    "Handle completion before fallback break logic (for example, switch out of span mode "
                    "or transition to unconstrained decoding), and call constrained-step helpers only when "
                    "`helpers.CanConstrain(generated)` is true. First premature break is near "
                    f"line {premature_not_can_constrain_lines[0]}."
                )
            if (
                not uses_split_prefix_family
                and (
                "EndsWithLeftDelimiter" not in strategy_body
                and "IsLeftDelimiterToken" not in strategy_body
                and "SpacedLeftDelimiter" not in strategy_body
                and '" <<"' not in strategy_body
                and "' <<'" not in strategy_body
                )
            ):
                return (
                    "When using natural left-delimiter helpers, handle both left-delimiter tokenizations. "
                    "Prefer `helpers.EndsWithLeftDelimiter(generated)` after append-style natural steps, "
                    "or use `helpers.IsLeftDelimiterToken(next_token)` for raw token-returning steps."
                )
            if (
                not uses_split_prefix_family
                and (
                uses_natural_right_delimiter
                and "EndsWithRightDelimiter" not in strategy_body
                and "IsRightDelimiterToken" not in strategy_body
                and "SpacedRightDelimiter" not in strategy_body
                and '" >>"' not in strategy_body
                and "' >>'" not in strategy_body
                )
            ):
                return (
                    "When using natural right-delimiter helpers, handle both right-delimiter tokenizations. "
                    "The missing spaced variant is `SpacedRightDelimiter` / `\" >>\"`. "
                    "Prefer `helpers.EndsWithRightDelimiter(generated)` after append-style constrained-or-close "
                    "steps, or use `helpers.IsRightDelimiterToken(next_token)` for raw token-returning steps."
                )
            if single_right_close_terminal_lines and not uses_split_prefix_family:
                return (
                    "Do not use a single global `sawRight` / `rightClosed` flag as the decoding-loop "
                    "terminator in GSM natural-delimiter mode. That stops after the first scratch or "
                    "subproblem span. Track closed spans/scratch spans and a separate final-ready state, "
                    "then continue free-form reasoning after non-final spans and stop only after the "
                    f"final answer span. First risky loop is near line {single_right_close_terminal_lines[0]}."
                )
            if not uses_split_prefix_family and not span_counter_state:
                return (
                    "GSM natural-delimiter strategies should track verified/scratch spans explicitly "
                    "(for example `closed_spans` or `scratch_spans`) so the first mini-expression is "
                    "not accidentally treated as the final answer. Counters like `spanTokens` only count "
                    "tokens inside one span and are not enough. Continue after non-final spans and emit "
                    "a later final span that composes the scratch values."
                )
            if not uses_split_prefix_family and not right_delimiter_span_counter_updates:
                return (
                    "When `ConstrainedOrRightDelimiterStep` naturally emits `RightDelimiter` or "
                    "`SpacedRightDelimiter`, update a real closed-span counter such as `closed_spans = "
                    "closed_spans + 1`. That lets the strategy distinguish scratch mini-expressions "
                    "from the later final answer span."
                )
            if not uses_split_prefix_family and not any(_state_used_in_conditions(name) for name in span_counter_state):
                return (
                    "Use the closed-span/scratch-span counter in branch or loop conditions so it affects "
                    "whether decoding continues after a scratch mini-expression versus stops after the "
                    "final answer span."
                )
            if phase_break_open_lines:
                return (
                    "Do not switch into an opening/span phase and immediately `break` out of the decoding "
                    "loop before emitting a helper step. That exits with plain free-form text or an "
                    "unclosed span instead of giving `UnconstrainedNudgeLeftDelimiterStep` and "
                    "`ConstrainedOrRightDelimiterStep` a chance to emit the final `<< ... >>` segment. "
                    "When a phase changes to `open`, `nudge`, `span`, or `answer`, either perform the "
                    "corresponding helper call in the same branch, or keep looping under a variant that "
                    "still consumes `stepsLeft`. First phase/break transition is near line "
                    f"{phase_break_open_lines[0]}."
                )
            if state_only_open_transition_lines:
                return (
                    "Do not switch into answer-opening state (for example by setting `answer_ready`, "
                    "`final_ready`, `phase = \"wrapup\"`, or `phase = \"open\"`) in a branch that consumes "
                    "no helper step and does not terminate. With the standard no-progress guard, that "
                    "pattern usually exits immediately with plain prose and no verified `<< ... >>` span. "
                    "When a branch enters wrap-up/open/span state, emit the corresponding helper step in "
                    "that same branch. First state-only transition is near line "
                    f"{state_only_open_transition_lines[0]}."
                )
            if parser_readiness_early_open_lines:
                return (
                    "Do not set `final_ready`/`answer_ready` from parser-distance or valid-continuation "
                    "signals after only a short setup phase. Those predicates describe grammar shape, not "
                    "semantic finality, and in GSM they tend to open on intermediate fragments such as "
                    "`2 * 30 = 60`. Use them only after a durable forty-plus-token setup/final-cue phase, "
                    "or combine them with explicit scratch-to-final state. First early parser-readiness "
                    f"assignment is near line {parser_readiness_early_open_lines[0]}."
                )
            if natural_open_plain_fallback_lines:
                return (
                    "After entering a GSM natural opening/nudge phase, do not fall back to plain "
                    "`AppendUnconstrainedStep` or `UnconstrainedStep`. That lets the LM solve in prose "
                    "without ever emitting `<<`, causing max-token no-delimiter failures. Once "
                    "`answer_ready` or `phase == \"open\"` is active, keep using "
                    "`AppendUnconstrainedNudgeLeftDelimiterStep(...)` until `helpers.EndsWithLeftDelimiter` "
                    "becomes true, or break only on a real no-progress/dead-end guard. First plain "
                    f"open-phase unconstrained fallback is near line {natural_open_plain_fallback_lines[0]}."
                )
            if low_final_ready_lines:
                return (
                    "Do not set `final_ready = 1` from a low reasoning-token threshold such as "
                    "`reason_signal >= 24`, `reason_steps > 3`, or `cue_signal = 1` after four "
                    "tokens; those policies open on the first "
                    "local arithmetic fragment. Wait for explicit final-answer cues, a scratch-to-final "
                    "transition, or a substantially later delimiter-masked reasoning/setup phase "
                    "around forty-plus tokens when relying on counters alone. "
                    "First risky assignment is "
                    f"near line {low_final_ready_lines[0]}."
                )
            if low_reason_nudge_lines:
                return (
                    "Do not start natural delimiter opening from a low reasoning-token threshold such as "
                    "`reasoning_steps >= 24` or `reason_steps > 6`. That still captures the first local "
                    "arithmetic fragment. For the late-single-final GSM pattern, keep delimiters masked "
                    "through a substantially later ordinary reasoning/setup phase, then nudge only after "
                    "roughly forty-plus reasoning/setup steps or explicit final-answer state. First risky nudge is near line "
                    f"{low_reason_nudge_lines[0]}."
                )
            if late_budget_answer_pressure_lines:
                return (
                    "Do not wait until only a tiny token budget remains before starting natural answer opening. "
                    "A rule like `not helpers.HasBudget(stepsLeft, 6)` or `stepsLeft <= 4` gives the LM too few "
                    "chances to emit `<<`, so GSM outputs often exhaust the budget with no delimited answer. "
                    "Start the answer-opening/nudge phase earlier using a moderate budget threshold "
                    "(for example around 16-32 remaining steps) combined with answer intent or scratch-to-final "
                    f"state. First late-budget pressure assignment is near line {late_budget_answer_pressure_lines[0]}."
                )
            if prefer_scratch_spans:
                final_span_state = {
                    name for name in extra_state if _is_final_span_state_name(name)
                }
                if not final_span_state or not any(_state_used_in_conditions(name) for name in final_span_state):
                    return (
                        "With `CSD_GSM_PREFER_SCRATCH_SPANS=1`, track explicit scratch-vs-final state "
                        "such as `final_ready`, `final_span`, or `answer_ready`, and use it in branch/loop "
                        "conditions so non-final spans can continue and a later span is treated as final."
                    )
                scratch_span_state = {
                    name for name in extra_state if _is_scratch_span_state_name(name)
                }
                if not scratch_span_state or not any(_state_used_in_conditions(name) for name in scratch_span_state):
                    return (
                        "With `CSD_GSM_PREFER_SCRATCH_SPANS=1`, track explicit scratch-span state "
                        "such as `scratch_mode`, `scratch_ready`, or `opening_scratch_span` so "
                        "non-final spans are represented as deliberate scratch spans instead of "
                        "anonymous local fragments."
                    )
                has_multi_span_gate = any(
                    _condition_has_span_counter_threshold(node.test, span_counter_state)
                    for node in ast.walk(tree)
                    if isinstance(node, (ast.If, ast.While))
                )
                if not has_multi_span_gate:
                    return (
                        "With `CSD_GSM_PREFER_SCRATCH_SPANS=1`, include an explicit multi-span condition "
                        "such as `closed_spans >= 2` (or equivalent) so one early closed span is not treated "
                        "as the automatic stopping point."
                    )
                right_delim_policy_split = False
                for if_node in [node for node in ast.walk(tree) if isinstance(node, ast.If)]:
                    if not _condition_handles_right_delimiter_token(if_node.test):
                        continue
                    for nested_if in [inner for inner in if_node.body if isinstance(inner, ast.If)]:
                        if (
                            _condition_has_span_counter_threshold(nested_if.test, span_counter_state)
                            or _condition_mentions_final_state(nested_if.test, final_span_state)
                        ):
                            right_delim_policy_split = True
                            break
                    if right_delim_policy_split:
                        break
                if not right_delim_policy_split:
                    return (
                        "With `CSD_GSM_PREFER_SCRATCH_SPANS=1`, do not close a span and immediately stop. "
                        "In the right-delimiter handling branch, add a scratch-vs-final decision "
                        "(for example using `closed_spans` plus `final_ready`) so non-final spans continue "
                        "and only the final span terminates."
                    )
                has_final_plus_span_signal = any(
                    (
                        _condition_mentions_final_state(node.test, final_span_state)
                        and any(isinstance(inner, ast.Name) and inner.id in span_counter_state for inner in ast.walk(node.test))
                    )
                    for node in ast.walk(tree)
                    if isinstance(node, (ast.If, ast.While))
                )
                if not has_final_plus_span_signal:
                    return (
                        "With `CSD_GSM_PREFER_SCRATCH_SPANS=1`, final-span decisions must depend on "
                        "more than raw token-count pressure. Combine final-intent state with span history "
                        "(for example `if final_ready and closed_spans >= 2:`) so final opening/termination "
                        "depends on prior scratch-span progress."
                    )
        if spider_force_single_sql_span:
            right_delim_nonterminal_lines: list[int] = []
            right_delim_followup_helper_lines: list[int] = []
            append_helpers = {
                "AppendUnconstrainedStep",
                "UnconstrainedStep",
                "AppendConstrainedStep",
                "AppendConstrainedOrRightDelimiterStep",
                "ConstrainedStep",
                "ConstrainedOrRightDelimiterStep",
                "AppendLeftDelimiter",
                "AppendRightDelimiter",
            }
            for if_node in [node for node in ast.walk(tree) if isinstance(node, ast.If)]:
                if not _condition_handles_right_delimiter_token(if_node.test):
                    continue
                branch_has_terminal = False
                branch_has_helper_followup = False
                for stmt in if_node.body:
                    if isinstance(stmt, (ast.Break, ast.Return)):
                        branch_has_terminal = True
                    for inner in ast.walk(stmt):
                        if (
                            isinstance(inner, ast.Call)
                            and isinstance(inner.func, ast.Attribute)
                            and isinstance(inner.func.value, ast.Name)
                            and inner.func.value.id == "helpers"
                            and inner.func.attr in append_helpers
                        ):
                            branch_has_helper_followup = True
                if not branch_has_terminal:
                    right_delim_nonterminal_lines.append(getattr(if_node, "lineno", 0))
                if branch_has_helper_followup:
                    right_delim_followup_helper_lines.append(getattr(if_node, "lineno", 0))
            if right_delim_nonterminal_lines:
                return (
                    "With `CSD_SPIDER_FORCE_SINGLE_SQL_SPAN=1`, once "
                    "`helpers.EndsWithRightDelimiter(generated)` is true, terminate immediately "
                    "(break/return) instead of transitioning to an after-phase. "
                    f"First non-terminal right-delimiter branch is near line {right_delim_nonterminal_lines[0]}."
                )
            if right_delim_followup_helper_lines:
                return (
                    "With `CSD_SPIDER_FORCE_SINGLE_SQL_SPAN=1`, do not emit any additional helper steps "
                    "inside a right-delimiter branch. After `>>`, stop immediately so no trailing spans can appear. "
                    f"First right-delimiter branch with helper calls is near line {right_delim_followup_helper_lines[0]}."
                )
            if premature_not_can_constrain_lines:
                return (
                    "`helpers.CanConstrain(generated)` is false when the constrained suffix is already complete. "
                    "In Spider mode, a `not helpers.CanConstrain(generated): break` branch before a completion-aware "
                    "close branch can exit with an unclosed `<< ...` span. Use a positive close-capable branch such as "
                    "`helpers.IsComplete(generated) or helpers.CanConstrain(generated)` before fallback break logic. "
                    f"First premature break is near line {premature_not_can_constrain_lines[0]}."
                )
            if uses_natural_left_delimiter:
                return (
                    "With `CSD_SPIDER_FORCE_SINGLE_SQL_SPAN=1`, use explicit delimiter helpers for Spider: "
                    "`helpers.AppendLeftDelimiter(...)` for SQL-span opening. "
                    "Do not use natural LEFT-delimiter helpers such as "
                    "`AppendUnconstrainedAllowLeftDelimiterStep` "
                    "or `AppendUnconstrainedNudgeLeftDelimiterStep`."
                )
            if "AppendLeftDelimiter" not in helper_calls:
                return (
                    "With `CSD_SPIDER_FORCE_SINGLE_SQL_SPAN=1`, include an executable call to "
                    "`helpers.AppendLeftDelimiter(generated, stepsLeft)` so Spider outputs enter "
                    "an explicit SQL answer span."
                )
            if (
                "ConstrainedOrRightDelimiterStep" not in helper_calls
                and "AppendConstrainedOrRightDelimiterStep" not in helper_calls
            ):
                return (
                    "With `CSD_SPIDER_FORCE_SINGLE_SQL_SPAN=1`, constrained SQL spans must use a "
                    "right-closure-capable helper (`ConstrainedOrRightDelimiterStep` or "
                    "`AppendConstrainedOrRightDelimiterStep`) so the model can close `>>` as soon as "
                    "the SQL prefix is complete."
                )
            if spider_long_freeform_lines:
                return (
                    "With `CSD_SPIDER_FORCE_SINGLE_SQL_SPAN=1`, keep unconstrained prelude short "
                    "(usually <= 3 steps) before entering the SQL span. "
                    f"First long-freeform threshold is near line {spider_long_freeform_lines[0]}."
                )
            if unconstrained_calls > 0:
                has_checkpoint_flow = "Checkpoint" in helper_calls and (
                    "RestoreIfDead" in helper_calls or "RestoreCheckpoint" in helper_calls
                )
                if not has_checkpoint_flow:
                    return (
                        "With `CSD_SPIDER_FORCE_SINGLE_SQL_SPAN=1`, if you call unconstrained helpers at all, "
                        "pair them with bounded rollback-style safety: take a `helpers.Checkpoint(generated)` "
                        "before risky growth and use `helpers.RestoreIfDead(...)` or "
                        "`helpers.RestoreCheckpoint(...)` on failure paths. This keeps SQL spans recoverable "
                        "to a grammar-valid state."
                    )
            if spider_force_span_at_start:
                if unconstrained_calls > 0:
                    return (
                        "With `CSD_SPIDER_FORCE_SPAN_AT_START=1`, do not call unconstrained helpers before "
                        "the SQL span. Start directly with `helpers.AppendLeftDelimiter(...)` so output begins "
                        "with `<<`."
                    )
                if first_helper_call_attr and first_helper_call_attr != "AppendLeftDelimiter":
                    return (
                        "With `CSD_SPIDER_FORCE_SPAN_AT_START=1`, the first helper call must be "
                        "`helpers.AppendLeftDelimiter(...)`. "
                        f"First helper call found: `{first_helper_call_attr}` near line {first_helper_call_line}."
                    )
        if fixed_phase_quota_lines:
            return (
                "Do not introduce tiny fixed phase-quota constants such as `min_reason_steps`, "
                "`reason_limit`, `max_answer_steps`, or `answer_budget`. For GSM, a durable delayed "
                "free-form phase is acceptable, but short quotas tend to capture the first local "
                "arithmetic fragment. Reason long enough before the final span, and close spans using parser "
                "completion plus semantic/budget signals rather than tiny token-count quotas. "
                f"First fixed phase quota is near line {fixed_phase_quota_lines[0]}."
            )
        if insufficient_reason_budget_lines:
            return (
                f"This run requires at least {min_required_reason_steps} free-form reasoning/setup steps "
                "before the first answer delimiter. Do not open << >> after a one- or two-token prelude; "
                "set min/max reasoning or setup budgets high enough for the evaluator LM to solve the problem "
                f"before entering the constrained span. First insufficient budget is near line {insufficient_reason_budget_lines[0]}."
            )
        if insufficient_answer_budget_lines:
            return (
                f"This run requires at least {min_required_answer_steps} constrained answer steps before closing. "
                "Do not close on the first complete prefix; short complete prefixes often truncate multi-digit "
                f"constants. First insufficient answer budget is near line {insufficient_answer_budget_lines[0]}."
            )
        if trivial_reason_budget_lines or trivial_answer_budget_lines:
            return (
                "Avoid trivial fixed-count GSM-style strategies such as a one-token free-form prelude followed by "
                "a single short answer span. Prefer adaptive interleaving: reason freely, optionally emit complete "
                "verified subexpression spans, continue reasoning, and finish with a final verified answer span. "
                f"First trivial budget constant is near line {(trivial_reason_budget_lines or trivial_answer_budget_lines)[0]}."
            )
        if malformed_tuple_assignment_lines:
            return (
                "Tuple assignment is only supported for helper call results or tuple literals. "
                "Complete the helper call, e.g. `generated, stepsLeft = helpers.AppendConstrainedStep(...)`, "
                f"near line {malformed_tuple_assignment_lines[0]}."
            )
        if manual_stepsleft_mutations:
            return (
                "Do not manually increment, decrement, or recompute `stepsLeft`; helper calls already consume "
                "budget and preserve the proof invariant. Only update `stepsLeft` from helper returns, such as "
                "`generated, stepsLeft = helpers.Append...(...)` or `stepsLeft = new_steps` after a raw step. "
                f"First manual mutation is near line {manual_stepsleft_mutations[0]}."
            )
        if bare_forced_token_calls:
            return (
                "helpers.ForcedTokenStep returns (next_token, new_steps) and must not be called as a bare statement. "
                "Assign its result, append next_token, and update stepsLeft = new_steps."
            )
        if bare_append_helper_calls:
            return (
                "Append* helper calls return (updated_prefix, remaining_steps) and must not be used as bare statements. "
                "Assign them back into generated/stepsLeft."
            )
        if append_helper_wrong_targets:
            return (
                "Helpers that update the generated prefix must assign the updated prefix back into "
                "`generated` (and, when applicable, update `stepsLeft` from the returned budget). "
                "Do not assign those results only into token variables like `next_token`. Offending helpers: "
                + ", ".join(sorted(append_helper_wrong_targets)) + "."
            )
        if none_checkpoint_assign_lines:
            return (
                "Do not initialize checkpoint-like state to `None`; Dafny treats checkpoint variables as token "
                "prefix sequences. Use `[]` (or `helpers.Checkpoint(generated)` once initialized) and track a "
                "separate boolean like `has_checkpoint`. First invalid assignment is near line "
                f"{none_checkpoint_assign_lines[0]}."
            )
        if sequential_helper_without_budget_lines:
            return (
                "A helper call can consume the last remaining step, so do not make a second step-consuming "
                "helper call later in the same loop iteration unless that later call is under an explicit "
                "`stepsLeft > 0` or `stepsLeft >= 1` guard. Prefer one helper append per iteration, then "
                "inspect `EndsWithLeftDelimiter`/`EndsWithRightDelimiter` and let the next loop iteration "
                f"continue. First unguarded second helper is near line {sequential_helper_without_budget_lines[0]}."
            )
        if unguarded_constrained_calls:
            return (
                "Every constrained helper call must be inside a branch or loop condition that explicitly checks "
                "`helpers.CanConstrain(generated)`. "
                "Unguarded calls: " + ", ".join(sorted(unguarded_constrained_calls)) + "."
            )
        if complete_branch_constrained_lines:
            return (
                "Do not call `helpers.AppendConstrainedStep(...)` (or raw `ConstrainedStep`) inside a "
                "`helpers.IsComplete(generated)` branch. Completeness means you should close the span "
                "(or leave constrained mode), and constrained-step there violates helper preconditions. "
                f"First offending call is near line {complete_branch_constrained_lines[0]}."
            )
        if unguarded_right_delimiter_calls and not spider_force_single_sql_span:
            return (
                "RightDelimiter emission must be inside a branch whose condition explicitly checks "
                "`helpers.IsComplete(generated)` or `parser.IsCompletePrefix(helpers.LongestValidSuffix(generated))`. Do not close the answer "
                "span unconditionally or merely because a phase variable changed. Offending calls: "
                + ", ".join(sorted(unguarded_right_delimiter_calls)) + "."
            )
        if dangling_if_chain_lines and not has_step_snapshot_guard:
            return (
                "Inside a `# decreases stepsLeft` loop, top-level `if/elif` chains must have an explicit final "
                "`else` fallback that either consumes a helper step or `break`s. Otherwise some states can loop "
                "without decreasing `stepsLeft`. First dangling chain is near line "
                f"{dangling_if_chain_lines[0]}."
            )
        if budget_only_open_lines:
            return (
                "Do not open a new constrained window based only on remaining token budget "
                "(for example `if stepsLeft <= k: AppendLeftDelimiter(...)`). That tends to reopen spans "
                "repeatedly near the end. Gate opening with explicit answer intent/state "
                "(for example `final_ready`, `closed_spans`, or answer-cue state), not raw budget alone. "
                f"First budget-only opening is near line {budget_only_open_lines[0]}."
            )
        if delimiter_calls_outside_loop:
            return (
                "Delimiter append calls must appear inside a budget-bounded decoding while loop, not after the loop "
                "where `stepsLeft` may be zero. Move these calls into explicit phase branches: "
                + ", ".join(sorted(delimiter_calls_outside_loop)) + "."
            )
        helper_lines: list[tuple[int, str]] = []
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "helpers"
            ):
                helper_lines.append((getattr(node, "lineno", 0), node.func.attr))
        constrained_lines = [line for line, attr in helper_lines if attr in constrained_helper_methods]
        if (
            not uses_split_prefix_policy
            and (
            not uses_natural_left_delimiter
            and constrained_lines
            and left_delimiter_lines
            and min(constrained_lines) < min(left_delimiter_lines)
            )
        ):
            return (
                "Constrained answer-token helpers must appear after executable LeftDelimiter emission in the method body. "
                "Do not generate constrained answer content before `helpers.AppendLeftDelimiter(...)`."
            )
        if (
            not uses_split_prefix_policy
            and (
            not spider_force_single_sql_span
            and not uses_natural_right_delimiter
            and constrained_lines
            and right_delimiter_lines
            and min(right_delimiter_lines) <= min(constrained_lines)
            )
        ):
            return (
                "RightDelimiter emission must appear after constrained answer-token helpers, and only after "
                "`helpers.IsComplete(generated)` is true."
            )
        if nondecreasing_else_lines and not has_step_snapshot_guard:
            return (
                "Branches inside a `# decreases stepsLeft` loop must either consume a helper step or `break`; "
                "do not use an `else` branch that only changes phase/state and loops again. "
                f"First non-consuming else branch is near line {nondecreasing_else_lines[0]}."
            )
        if continue_lines:
            return (
                "Do not use `continue` inside the decoding while loop. In Dafny this often breaks the "
                "`# decreases stepsLeft` proof when some continue paths do not consume a helper step. "
                f"First `continue` is near line {continue_lines[0]}."
            )
        if top_level_break_lines:
            return (
                "Do not place a bare top-level `break` directly in the decoding while loop body. "
                "Use guarded branch-local termination (for example inside explicit completion/dead-state branches), "
                "otherwise the strategy exits after one step and never reaches a verified answer span. "
                f"First top-level `break` is near line {top_level_break_lines[0]}."
            )
        if stray_expression_lines:
            return (
                "Found a stray expression statement (for example a dangling name) that usually indicates "
                "a truncated generation and later Dafny type/transpilation failures. "
                f"First stray expression is near line {stray_expression_lines[0]}."
            )
        for while_node in while_nodes:
            while_test_text = ast.unparse(while_node.test)
            normalized_while_test = while_test_text.replace("_", "").lower()
            if (
                "sawright" in normalized_while_test
                or "rightclosed" in normalized_while_test
                or "closedright" in normalized_while_test
                or "notclosed" in normalized_while_test
            ) and "final" not in normalized_while_test:
                single_right_close_terminal_lines.append(getattr(while_node, "lineno", 0))
            budget_names = {
                name.id
                for name in ast.walk(while_node.test)
                if isinstance(name, ast.Name)
            }
            if "stepsLeft" not in budget_names and "new_steps" not in budget_names:
                return (
                    "Every decoding while loop must be budget-bounded, e.g. "
                    "`while stepsLeft > 0 and ...:`."
                )
            body_lineno = getattr(while_node, "lineno", 0) - 1
            while_index = body_lineno - 1
            block_start = while_index - len(expected_invariant_block)
            if block_start < 0:
                return (
                    "The standard loop invariant block must be immediately above each decoding `while` line, "
                    "not indented inside the loop body."
                )
            preceding_block = [
                line.strip()
                for line in body_lines[block_start:while_index]
            ]
            if preceding_block != expected_invariant_block:
                return (
                    "The standard loop invariant block must be immediately above each decoding `while` line in the standard order, "
                    "not indented inside the loop body."
                )
        dafny_reserved_locals = {"opened"}
        conflicting_reserved = sorted(name for name in extra_state if name in dafny_reserved_locals)
        if conflicting_reserved:
            return (
                "Avoid local variable names that collide with Dafny reserved identifiers. "
                "Rename: " + ", ".join(conflicting_reserved) + "."
            )
        if not appends_generated:
            return (
                "The body must either append produced tokens with generated = generated + [next_token] "
                "or use an Append* helper that assigns back into generated."
            )
        if "# invariant helpers.lm == lm" not in body:
            return "The body must include the standard loop invariant `# invariant helpers.lm == lm`."
        if "# invariant helpers.parser == parser" not in body:
            return "The body must include the standard loop invariant `# invariant helpers.parser == parser`."
        if "# invariant lm.ValidTokensIdsLogits()" not in body:
            return "The body must include the standard loop invariant `# invariant lm.ValidTokensIdsLogits()`."
        if "# invariant 0 <= stepsLeft <= maxSteps" not in body:
            return "The body must include the standard loop invariant `# invariant 0 <= stepsLeft <= maxSteps`."
        if "# invariant |generated| + stepsLeft <= maxSteps" not in body:
            return "The body must include the standard loop invariant `# invariant |generated| + stepsLeft <= maxSteps`."
        if "# decreases stepsLeft" not in body:
            return "The body must include the standard decreases clause `# decreases stepsLeft` immediately above a decoding while loop."
        if os.environ.get("CSD_STRICT_COMPLETE_ORDER", "").strip() in {"1", "true", "True"} and constrain_before_complete_lines:
            return (
                "In a constrained answer phase, check "
                "`helpers.IsComplete(generated)` before an open-ended "
                "`helpers.CanConstrain(generated)` branch. "
                "Otherwise a complete expression can keep extending forever as a valid prefix and never close. "
                "Move the complete-prefix close/extend branch earlier, or add an explicit "
                "`and not helpers.IsComplete(generated)` guard on the "
                "open-ended constrained branch. "
                f"First risky branch is near line {constrain_before_complete_lines[0]}."
            )
        if len(extra_state) < 2:
            return "The body must maintain at least two extra local state variables so the strategy is not a trivial loop."
        if if_count < 2:
            return "The body needs richer control flow than a single top-level branch."
        return None

    def _ensure_nontrivial_strategy(self, strategy_body: str, *, max_repairs: int = 2) -> str:
        current = strategy_body
        trace: list[dict[str, object]] = []
        self.last_structure_repair_trace = trace
        self.last_structure_validation_summary = {}
        autofix_passes = 0
        for repair_round in range(1, max_repairs + 1):
            issue = self._structural_issue(current)
            if issue is None:
                fixed = self._autofix_python_strategy(current)
                autofix_changed = fixed != current
                if autofix_changed:
                    autofix_passes += 1
                final_issue = self._structural_issue(fixed)
                if final_issue is None:
                    self.last_structure_validation_summary = {
                        "structural_repairs": len(trace),
                        "autofix_passes": autofix_passes,
                        "autofix_changed": autofix_changed,
                    }
                    return fixed
                current = fixed
                issue = final_issue
            repair_record: dict[str, object] = {
                "round": repair_round,
                "input_strategy_length": len(current),
                "input_strategy": self._diagnostic_excerpt(current),
                "issue": issue,
            }
            system_prompt, user_prompt = build_structure_repair_prompt(
                current,
                issue,
                strategy_language=self.strategy_language,
            )
            repaired_raw = self._generate_text(system_prompt, user_prompt)
            repaired = (
                self._extract_dafny_strategy(repaired_raw)
                if self.strategy_language == "dafny"
                else self._extract_strategy(repaired_raw)
            )
            repair_record.update(
                {
                    "repair_raw_output_empty": repaired_raw == "",
                    "repair_raw_output_length": len(repaired_raw),
                    "repair_raw_output": self._diagnostic_excerpt(repaired_raw),
                    "repair_extracted_strategy_empty": repaired == "",
                    "repair_extracted_strategy_length": len(repaired),
                    "repair_extracted_strategy": self._diagnostic_excerpt(repaired),
                }
            )
            trace.append(repair_record)
            current = self._ensure_rationale_block(repaired)

        issue = self._structural_issue(current)
        if issue is None:
            fixed = self._autofix_python_strategy(current)
            autofix_changed = fixed != current
            if autofix_changed:
                autofix_passes += 1
            final_issue = self._structural_issue(fixed)
            if final_issue is None:
                self.last_structure_validation_summary = {
                    "structural_repairs": len(trace),
                    "autofix_passes": autofix_passes,
                    "autofix_changed": autofix_changed,
                }
                return fixed
            issue = final_issue
        trace.append(
            {
                "round": max_repairs + 1,
                "input_strategy_length": len(current),
                "input_strategy": self._diagnostic_excerpt(current),
                "issue": issue,
                "terminal": True,
            }
        )

        raise ValueError(
            "Generated strategy is structurally invalid. "
            f"It must contain executable decoding logic with a while loop and helper step calls. Last issue: {issue}"
        )

    def _candidate_rank(
        self,
        novelty_score: int,
        *,
        validation_summary: Optional[dict[str, object]] = None,
        rationale_repairs: Optional[int] = None,
    ) -> tuple[int, int]:
        summary = validation_summary or {}
        structural_repairs = int(summary.get("structural_repairs", 0))
        autofix_passes = int(summary.get("autofix_passes", 0))
        rationale_repair_count = (
            self.last_rationale_repair_count if rationale_repairs is None else rationale_repairs
        )
        stability_score = 100
        stability_score -= 20 * structural_repairs
        stability_score -= 8 * autofix_passes
        stability_score -= 6 * rationale_repair_count
        return stability_score, novelty_score

    def _novelty_score(self, strategy_body: str) -> int:
        body = self._body_without_rationale(strategy_body)
        try:
            wrapped = "def _strategy():\n" + textwrap.indent(body, "    ")
            tree = ast.parse(wrapped)
        except SyntaxError:
            return -10_000

        helper_calls: set[str] = set()
        extra_state: set[str] = set()
        if_count = 0
        while_count = 0
        bool_complexity = 0
        nested_if = 0

        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if isinstance(node.func.value, ast.Name) and node.func.value.id == "helpers":
                    helper_calls.add(node.func.attr)
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id not in {"generated", "answer", "stepsLeft", "next_token", "new_steps"}:
                        extra_state.add(target.id)
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                if node.target.id not in {"generated", "answer", "stepsLeft", "next_token", "new_steps"}:
                    extra_state.add(node.target.id)
            if isinstance(node, ast.If):
                if_count += 1
                if isinstance(node.test, ast.BoolOp):
                    bool_complexity += max(0, len(node.test.values) - 1)
                if any(isinstance(inner, ast.If) for inner in node.body):
                    nested_if += 1
            if isinstance(node, ast.While):
                while_count += 1
                if isinstance(node.test, ast.BoolOp):
                    bool_complexity += max(0, len(node.test.values) - 1)

        score = 0
        score += 6 * len(extra_state)
        score += 5 * if_count
        score += 4 * while_count
        score += 3 * bool_complexity
        score += 4 * nested_if
        if (
            "ConstrainedStep" in helper_calls
            or "AppendConstrainedStep" in helper_calls
            or "ConstrainedOrRightDelimiterStep" in helper_calls
            or "AppendConstrainedOrRightDelimiterStep" in helper_calls
        ):
            score += 10
        if "ConstrainedStep" in helper_calls:
            score += 18
        if "AppendConstrainedStep" in helper_calls:
            score += 20
        if "ConstrainedOrRightDelimiterStep" in helper_calls:
            score += 24
        if "AppendConstrainedOrRightDelimiterStep" in helper_calls:
            score += 26
        if "AdaptiveConstrainedStep" in helper_calls:
            score += 28
        if "GroupBoostedConstrainedStep" in helper_calls:
            score += 20
        if "PenalizedConstrainedStep" in helper_calls:
            score += 16
        if "OpenConstrainedSpan" in helper_calls:
            score += 18
        if "CloseConstrainedSpan" in helper_calls:
            score += 18
        if "AppendConstrainedToken" in helper_calls:
            score += 16
        if "LastTokenBefore" in helper_calls or "CountOccurrences" in helper_calls:
            score += 8
        if {"ForcedTokenStep", "AppendForcedToken", "AppendLeftDelimiter", "AppendRightDelimiter"} & helper_calls:
            score += 8
        if (
            "UnconstrainedStep" in helper_calls
            or "AppendUnconstrainedStep" in helper_calls
            or "UnconstrainedAllowLeftDelimiterStep" in helper_calls
            or "UnconstrainedNudgeLeftDelimiterStep" in helper_calls
            or "AppendUnconstrainedAllowLeftDelimiterStep" in helper_calls
            or "AppendUnconstrainedNudgeLeftDelimiterStep" in helper_calls
        ):
            score += 24
        if "UnconstrainedNudgeLeftDelimiterStep" in helper_calls:
            score += 28
        if (
            "UnconstrainedAllowLeftDelimiterStep" in helper_calls
            or "UnconstrainedNudgeLeftDelimiterStep" in helper_calls
            or "AppendUnconstrainedAllowLeftDelimiterStep" in helper_calls
            or "AppendUnconstrainedNudgeLeftDelimiterStep" in helper_calls
        ):
            score += 32
        if "UnconstrainedStep" in helper_calls:
            score += 14
        if "AppendUnconstrainedStep" in helper_calls:
            score += 3
        if "LongestValidSuffix" in helper_calls or "CanConstrain" in helper_calls or "IsComplete" in helper_calls:
            score += 6
        if len(helper_calls) >= 3:
            score += 8
        return score

    def _generate_valid_strategy(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        failure_context: str,
    ) -> str:
        search_attempts = max(
            1,
            int(os.environ.get("CSD_GENERATION_SEARCH_ATTEMPTS", str(self.SEARCH_ATTEMPTS))),
        )
        budgets = [
            max(self.max_new_tokens, self.MIN_STRATEGY_TOKENS),
            max(self.max_new_tokens, 320),
            max(self.max_new_tokens, 384),
            max(self.max_new_tokens, 640),
            max(self.max_new_tokens, 800),
        ]
        while len(budgets) < search_attempts:
            budgets.append(budgets[-1])
        temperatures = [
            max(self.temperature, 0.85),
            max(self.temperature, 0.65),
            min(self.temperature, 0.35),
            max(self.temperature, 0.75),
            min(self.temperature, 0.25),
        ]
        while len(temperatures) < search_attempts:
            temperatures.append(temperatures[-1])
        budgets = budgets[:search_attempts]
        temperatures = temperatures[:search_attempts]

        last_error: str | None = None
        current_system = system_prompt
        current_user = user_prompt
        valid_candidates: list[tuple[tuple[int, int], str]] = []
        rejected_candidates: list[dict[str, object]] = []
        self.last_generation_diagnostics = []
        strategy_language = getattr(self, "strategy_language", "python")

        def _rejection_history_context() -> str:
            if not rejected_candidates:
                return ""
            lines = [
                "Rejected candidate history:",
                "Before proposing the next repair, inspect this list and avoid repeating the same structural direction.",
            ]
            for item in rejected_candidates[-6:]:
                candidate = item.get("candidate", "?")
                issue = str(item.get("issue", "")).replace("\n", " ")
                if "Last issue:" in issue:
                    issue = issue.split("Last issue:", 1)[1].strip()
                if len(issue) > 500:
                    issue = issue[:500] + "..."
                strategy_excerpt = str(item.get("extracted_strategy", ""))
                helpers = sorted(set(re.findall(r"helpers\.([A-Za-z_]\w*)", strategy_excerpt)))
                helper_text = ", ".join(helpers[:8]) if helpers else "no helper calls detected"
                lines.append(f"- Candidate {candidate}: {issue} Helpers: {helper_text}.")
            return "\n".join(lines)

        for idx, (budget, temp) in enumerate(zip(budgets, temperatures), start=1):
            raw_output = self._generate_text(
                current_system,
                current_user,
                max_new_tokens=budget,
                temperature=temp,
            )
            strategy = (
                self._extract_dafny_strategy(raw_output)
                if strategy_language == "dafny"
                else self._extract_strategy(raw_output)
            )
            diagnostic: dict[str, object] = {
                "candidate": idx,
                "max_new_tokens": budget,
                "temperature": temp,
                "system_prompt_length": len(current_system),
                "system_prompt": self._diagnostic_excerpt(current_system),
                "user_prompt_length": len(current_user),
                "user_prompt": self._diagnostic_excerpt(current_user),
                "raw_output_empty": raw_output == "",
                "raw_output_length": len(raw_output),
                "raw_output": self._diagnostic_excerpt(raw_output),
                "extracted_strategy_empty": strategy == "",
                "extracted_strategy_length": len(strategy),
                "extracted_strategy": self._diagnostic_excerpt(strategy),
                "accepted": False,
            }
            try:
                self.last_structure_repair_trace = []
                strategy = self._ensure_rationale_block(strategy)
                if strategy_language == "dafny":
                    strategy = self._ensure_nontrivial_dafny_strategy(strategy)
                else:
                    strategy = self._ensure_nontrivial_strategy(strategy)
                novelty_score = self._novelty_score(strategy)
                validation_summary = dict(getattr(self, "last_structure_validation_summary", {}))
                rationale_repairs = int(getattr(self, "last_rationale_repair_count", 0))
                candidate_rank = self._candidate_rank(
                    novelty_score,
                    validation_summary=validation_summary,
                    rationale_repairs=rationale_repairs,
                )
                diagnostic["accepted"] = True
                diagnostic["novelty_score"] = novelty_score
                diagnostic["candidate_rank"] = {
                    "stability_score": candidate_rank[0],
                    "novelty_score": candidate_rank[1],
                }
                diagnostic["validation_summary"] = validation_summary
                diagnostic["rationale_repair_count"] = rationale_repairs
                diagnostic["final_strategy_length"] = len(strategy)
                diagnostic["final_strategy"] = self._diagnostic_excerpt(strategy)
                valid_candidates.append((candidate_rank, strategy))
                current_system, current_user = system_prompt, user_prompt
                continue
            except ValueError as exc:
                last_error = str(exc)
                diagnostic["issue"] = last_error
                diagnostic["structure_repair_trace"] = [
                    dict(item) for item in getattr(self, "last_structure_repair_trace", [])
                ]
                rejected_candidates.append(dict(diagnostic))
                current_system, current_user = build_structure_repair_prompt(
                    strategy
                    or raw_output
                    or (
                        "# CSD_RATIONALE_BEGIN\n"
                        "# Empty output.\n"
                        "# CSD_RATIONALE_END\n"
                        "# CSD_PROOF_SKETCH_BEGIN\n"
                        "# No executable strategy was produced.\n"
                        "# CSD_PROOF_SKETCH_END"
                    ),
                    last_error,
                    strategy_language=strategy_language,
                )
                history_context = _rejection_history_context()
                if history_context:
                    current_user += "\n\n" + history_context
                print(
                    f"  Initial generation attempt {idx} produced an invalid body; "
                    f"retrying with a stricter repair prompt ({last_error})."
                )
            finally:
                self.last_generation_diagnostics.append(diagnostic)

        if valid_candidates:
            best_rank, best_strategy = max(valid_candidates, key=lambda item: item[0])
            print(
                "  Selected the strongest structurally valid candidate "
                f"(stability={best_rank[0]}, novelty={best_rank[1]})."
            )
            return best_strategy

        detail = last_error or "invalid model output"
        raise StrategyGenerationError(f"{failure_context}: {detail}")
    
    def generate_initial(self, task_description: str) -> str:
        """
        Generate an initial strategy for the given task.
        
        Args:
            task_description: Description of what the strategy should accomplish

        Returns:
            Strategy body (Python code)
        """
        system_prompt, user_prompt = build_initial_prompt(
            task_description,
            strategy_language=self.strategy_language,
        )
        return self._generate_valid_strategy(
            system_prompt,
            user_prompt,
            failure_context="The generation model did not produce a usable initial strategy",
        )
    
    def refine_after_verification_error(
        self,
        previous_strategy: str,
        error_message: str
    ) -> str:
        """
        Generate a refined strategy after verification failure.
        
        Args:
            previous_strategy: The strategy that failed
            error_message: Dafny verification error
            
        Returns:
            New strategy body
        """
        system_prompt, user_prompt = build_verification_error_prompt(
            previous_strategy,
            error_message,
            strategy_language=self.strategy_language,
        )
        return self._generate_valid_strategy(
            system_prompt,
            user_prompt,
            failure_context="The generation model did not produce a usable verification repair",
        )
    
    def refine_after_runtime_error(
        self,
        previous_strategy: str,
        error_traceback: str
    ) -> str:
        """
        Generate a refined strategy after runtime failure.
        
        Args:
            previous_strategy: The strategy that failed
            error_traceback: Python traceback
            
        Returns:
            New strategy body
        """
        system_prompt, user_prompt = build_runtime_error_prompt(
            previous_strategy,
            error_traceback,
            strategy_language=self.strategy_language,
        )
        return self._generate_valid_strategy(
            system_prompt,
            user_prompt,
            failure_context="The generation model did not produce a usable runtime repair",
        )
    
    def refine_after_compilation_error(
        self,
        previous_strategy: str,
        error_message: str
    ) -> str:
        """
        Generate a refined strategy after compilation failure.
        
        Args:
            previous_strategy: The strategy that failed
            error_message: Dafny compilation error
            
        Returns:
            New strategy body
        """
        system_prompt, user_prompt = build_compilation_error_prompt(
            previous_strategy,
            error_message,
            strategy_language=self.strategy_language,
        )
        return self._generate_valid_strategy(
            system_prompt,
            user_prompt,
            failure_context="The generation model did not produce a usable compilation repair",
        )

    def refine_after_evaluation_failure(
        self,
        previous_strategy: str,
        evaluation_feedback: str
    ) -> str:
        """
        Generate a refined strategy after evaluation failure.

        The strategy passed verification, compilation, and runtime testing,
        but performed poorly on actual dataset evaluation (low accuracy,
        format rate, syntax rate, or semantic rate).

        Args:
            previous_strategy: The strategy that failed evaluation
            evaluation_feedback: Feedback summary from the evaluator

        Returns:
            New strategy body
        """
        system_prompt, user_prompt = build_evaluation_failure_prompt(
            previous_strategy,
            evaluation_feedback,
            strategy_language=self.strategy_language,
        )
        return self._generate_valid_strategy(
            system_prompt,
            user_prompt,
            failure_context="The generation model did not produce a usable evaluation repair",
        )

    def inject_strategy(self, strategy: str) -> str:
        """
        Inject a strategy into the template.

        Args:
            strategy: Strategy expression to inject

        Returns:
            Complete Python source code
        """
        body = textwrap.dedent(strategy).strip("\n")
        indented = textwrap.indent(body, "    ")
        start = self._template.find(self.strategy_begin_marker)
        end = self._template.find(self.strategy_end_marker)
        if start == -1 or end == -1 or end < start:
            raise ValueError(f"Strategy hole markers not found in {self.template_path}")
        end += len(self.strategy_end_marker)
        return self._template[:start] + indented + self._template[end:]
    
    def get_template(self) -> str:
        """Get the raw template content."""
        return self._template
