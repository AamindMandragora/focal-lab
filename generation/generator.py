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
from .rationale import extract_rationale


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
    
    # Path to the template file
    TEMPLATE_PATH = Path(__file__).resolve().parent / "csd" / "GeneratedAgentTemplate.py"

    # Markers delimiting the hole to replace
    STRATEGY_BEGIN_MARKER = "    # QWEN_INSERT_STRATEGY_BEGIN"
    STRATEGY_END_MARKER = "    # QWEN_INSERT_STRATEGY_END"

    # Under this budget, Qwen often truncates before emitting a full rationale + loop body.
    MIN_STRATEGY_TOKENS = 192
    SEARCH_ATTEMPTS = 8
    DIAGNOSTIC_TEXT_LIMIT = 12_000
    ALLOWED_HELPER_METHODS = {
        "UnconstrainedStep",
        "UnconstrainedAllowLeftDelimiterStep",
        "UnconstrainedBiasLeftDelimiterStep",
        "UnconstrainedNudgeLeftDelimiterStep",
        "ConstrainedStep",
        "ConstrainedOrRightDelimiterStep",
        "SoftConstrainedStep",
        "TopKConstrainedStep",
        "ExtendConstrainedStep",
        "ForcedTokenStep",
        "BudgetAwareStep",
        "AppendUnconstrainedStep",
        "AppendConstrainedStep",
        "AppendSoftConstrainedStep",
        "AppendTopKConstrainedStep",
        "AppendExtendConstrainedStep",
        "AppendBudgetAwareStep",
        "AppendForcedToken",
        "AppendLeftDelimiter",
        "AppendRightDelimiter",
        "LongestValidSuffix",
        "CanConstrain",
        "CanExtendConstrained",
        "HasBudget",
        "MinStepsToComplete",
        "SoftConstrainToGrammar",
        "IntersectWithGrammar",
        "BiasForCompletion",
        "AllValidNextTokensInLM",
        "ValidTokensIdsLogitsAlways",
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

        # Load template
        self._template = self._load_template()
    
    def _load_template(self) -> str:
        """Load the `generation/csd/GeneratedAgentTemplate.py` template."""
        if not self.TEMPLATE_PATH.exists():
            raise FileNotFoundError(
                f"Template not found at {self.TEMPLATE_PATH}. "
                "Make sure generation/csd/GeneratedAgentTemplate.py exists."
            )
        return self.TEMPLATE_PATH.read_text()
    
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

    def _ensure_rationale_block(self, strategy_body: str, *, max_repairs: int = 2) -> str:
        """
        Ensure the strategy body contains the required rationale markers.

        If missing, attempt a small number of "format repair" generations that rewrite
        the body into the required structure without changing semantics.
        """
        extracted = extract_rationale(strategy_body)
        if extracted.rationale is not None and extracted.has_markers:
            return self._normalize_rationale_block(strategy_body)

        current = strategy_body
        for _ in range(max_repairs):
            system_prompt, user_prompt = build_format_repair_prompt(current)
            repaired_raw = self._generate_text(system_prompt, user_prompt)
            repaired = self._extract_strategy(repaired_raw)
            extracted = extract_rationale(repaired)
            if extracted.rationale is not None and extracted.has_markers:
                return self._normalize_rationale_block(repaired)
            current = repaired

        raise ValueError(
            "Generated strategy is missing required rationale block markers "
            "(# CSD_RATIONALE_BEGIN ... # CSD_RATIONALE_END)."
        )

    def _body_without_rationale(self, strategy_body: str) -> str:
        extracted = extract_rationale(strategy_body)
        return extracted.body_without_rationale if extracted.has_markers else strategy_body

    def _normalize_rationale_block(self, strategy_body: str) -> str:
        lines = strategy_body.splitlines()
        begin_idx = None
        end_idx = None
        for i, line in enumerate(lines):
            if line.strip() in {"# CSD_RATIONALE_BEGIN", "// CSD_RATIONALE_BEGIN"}:
                begin_idx = i
                break
        if begin_idx is None:
            return strategy_body
        for j in range(begin_idx + 1, len(lines)):
            if lines[j].strip() in {"# CSD_RATIONALE_END", "// CSD_RATIONALE_END"}:
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
        i = 0
        while i < len(lines):
            line = lines[i]
            fixed.append(line)
            stripped = line.lstrip()
            indent = len(line) - len(stripped)
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
                            rf"^{re.escape(branch_indent)}([A-Za-z_]\w*)\s*,\s*([A-Za-z_]\w*)\s*=\s*helpers\.(?:ConstrainedStep|ConstrainedOrRightDelimiterStep|SoftConstrainedStep|TopKConstrainedStep|UnconstrainedStep|UnconstrainedAllowLeftDelimiterStep|UnconstrainedBiasLeftDelimiterStep|UnconstrainedNudgeLeftDelimiterStep|ForcedTokenStep|BudgetAwareStep)\(",
                            first_branch,
                        )
                        else_assign = re.match(
                            rf"^{re.escape(branch_indent)}([A-Za-z_]\w*)\s*,\s*([A-Za-z_]\w*)\s*=\s*helpers\.(?:ConstrainedStep|ConstrainedOrRightDelimiterStep|SoftConstrainedStep|TopKConstrainedStep|UnconstrainedStep|UnconstrainedAllowLeftDelimiterStep|UnconstrainedBiasLeftDelimiterStep|UnconstrainedNudgeLeftDelimiterStep|ForcedTokenStep|BudgetAwareStep)\(",
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
        return "\n".join(fixed)

    def _structural_issue(self, strategy_body: str) -> str | None:
        body = self._body_without_rationale(strategy_body)
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
        manual_stepsleft_mutations: list[int] = []
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
        helper_parser_confusions: set[str] = set()
        unguarded_constrained_calls: set[str] = set()
        constrain_before_complete_lines: list[int] = []
        bad_bias_helper_lines: list[int] = []
        uses_natural_left_delimiter = False
        uses_natural_right_delimiter = False
        forced_left_delimiter_lines: list[int] = []
        forced_right_delimiter_lines: list[int] = []
        if_count = 0
        while_nodes = [node for node in ast.walk(tree) if isinstance(node, ast.While)]
        append_helper_methods = {
            "AppendUnconstrainedStep",
            "AppendConstrainedStep",
            "AppendSoftConstrainedStep",
            "AppendTopKConstrainedStep",
            "AppendExtendConstrainedStep",
            "AppendBudgetAwareStep",
            "AppendForcedToken",
            "AppendLeftDelimiter",
            "AppendRightDelimiter",
        }
        constrained_helper_methods = {
            "ConstrainedStep",
            "SoftConstrainedStep",
            "TopKConstrainedStep",
            "ExtendConstrainedStep",
            "ConstrainedOrRightDelimiterStep",
            "AppendConstrainedStep",
            "AppendSoftConstrainedStep",
            "AppendTopKConstrainedStep",
            "AppendExtendConstrainedStep",
        }
        unconstrained_helper_methods = {
            "UnconstrainedStep",
            "UnconstrainedAllowLeftDelimiterStep",
            "UnconstrainedBiasLeftDelimiterStep",
            "UnconstrainedNudgeLeftDelimiterStep",
            "AppendUnconstrainedStep",
            "BudgetAwareStep",
            "AppendBudgetAwareStep",
        }
        forced_helper_methods = {
            "ForcedTokenStep",
            "AppendForcedToken",
            "AppendLeftDelimiter",
            "AppendRightDelimiter",
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
                    and inner.func.attr in {"CanConstrain", "CanExtendConstrained"}
                    and len(inner.args) >= 1
                    and _is_name(inner.args[0], "generated")
                ):
                    return True
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
            return False

        def _condition_has_can_constrain_guard(test: ast.AST) -> bool:
            for inner in ast.walk(test):
                if (
                    isinstance(inner, ast.Call)
                    and isinstance(inner.func, ast.Attribute)
                    and isinstance(inner.func.value, ast.Name)
                    and inner.func.value.id == "helpers"
                    and inner.func.attr in {"CanConstrain", "CanExtendConstrained"}
                    and len(inner.args) >= 1
                    and _is_name(inner.args[0], "generated")
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

        min_required_reason_steps = 0
        min_required_answer_steps = 0

        def _is_reason_budget_name(name: str) -> bool:
            lowered = name.lower()
            return (
                ("reason" in lowered or "setup" in lowered or "prelude" in lowered)
                and any(marker in lowered for marker in ("min", "max", "limit", "budget", "target"))
            )

        def _is_answer_budget_name(name: str) -> bool:
            lowered = name.lower()
            return (
                ("answer" in lowered or "constrained" in lowered)
                and any(marker in lowered for marker in ("min", "limit", "budget", "target"))
            )

        def _is_fixed_phase_quota_name(name: str) -> bool:
            lowered = name.lower()
            return (
                any(marker in lowered for marker in ("min", "minimum", "max", "limit", "quota", "target", "budget"))
                and any(
                    marker in lowered
                    for marker in (
                        "reason",
                        "setup",
                        "prelude",
                        "scratch",
                        "answer",
                        "final",
                        "constrained",
                        "span",
                        "search",
                    )
                )
            )

        def _is_literal_int(node: ast.AST | None) -> bool:
            return isinstance(node, ast.Constant) and isinstance(node.value, int)

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

        def _top_level_if_branches(if_node: ast.If) -> list[tuple[ast.AST, list[ast.stmt]]]:
            branches: list[tuple[ast.AST, list[ast.stmt]]] = [(if_node.test, if_node.body)]
            current = if_node
            while len(current.orelse) == 1 and isinstance(current.orelse[0], ast.If):
                current = current.orelse[0]
                branches.append((current.test, current.body))
            if current.orelse:
                branches.append((current.test, current.orelse))
            return branches

        for while_node in while_nodes:
            for statement in while_node.body:
                if not isinstance(statement, ast.If):
                    continue
                branches = _top_level_if_branches(statement)
                seen_open_constrain_line = 0
                for branch_test, branch_body in branches:
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
                for _test, branch_body in _top_level_if_branches(statement):
                    if (
                        branch_body
                        and not _contains_helper_step(branch_body)
                        and not _contains_break_or_return(branch_body)
                        and _contains_state_assignment(branch_body)
                    ):
                        nondecreasing_else_lines.append(getattr(branch_body[0], "lineno", getattr(statement, "lineno", 0)))

        for node in ast.walk(tree):
            if isinstance(node, ast.If):
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
                if isinstance(node.func.value, ast.Name) and node.func.value.id == "helpers":
                    attr = node.func.attr
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
                        current = node
                        guarded = False
                        while current in parent_map:
                            current = parent_map[current]
                            if isinstance(current, (ast.If, ast.While)) and _condition_has_constrain_guard(current.test):
                                guarded = True
                                break
                        if not guarded:
                            unguarded_constrained_calls.add(attr)
                    if attr in {"TopKConstrainedStep", "AppendTopKConstrainedStep"}:
                        if len(node.args) >= 3:
                            k_arg = node.args[2]
                            if not (
                                isinstance(k_arg, ast.Constant)
                                and isinstance(k_arg.value, int)
                                and k_arg.value == 1
                            ):
                                topk_unprovable_calls.add(attr)
                        else:
                            topk_unprovable_calls.add(attr)
                    if attr in forced_helper_methods:
                        forced_token_calls += 1
                    if attr in unconstrained_helper_methods:
                        unconstrained_calls += 1
                        unconstrained_lines.append(getattr(node, "lineno", 0))
                    if attr == "UnconstrainedBiasLeftDelimiterStep":
                        bias_arg = node.args[2] if len(node.args) >= 3 else None
                        if not (
                            isinstance(bias_arg, ast.Constant)
                            and isinstance(bias_arg.value, float)
                            and bias_arg.value > 0.0
                        ):
                            bad_bias_helper_lines.append(getattr(node, "lineno", 0))
                    if attr == "AppendLeftDelimiter":
                        emits_left_delimiter = True
                        left_delimiter_lines.append(getattr(node, "lineno", 0))
                        forced_left_delimiter_lines.append(getattr(node, "lineno", 0))
                        if not _has_ancestor_while(node):
                            delimiter_calls_outside_loop.add(attr)
                    elif attr in {
                        "UnconstrainedAllowLeftDelimiterStep",
                        "UnconstrainedBiasLeftDelimiterStep",
                        "UnconstrainedNudgeLeftDelimiterStep",
                    }:
                        uses_natural_left_delimiter = True
                        emits_left_delimiter = True
                        left_delimiter_lines.append(getattr(node, "lineno", 0))
                    elif attr == "ConstrainedOrRightDelimiterStep":
                        uses_natural_right_delimiter = True
                        emits_right_delimiter = True
                        right_delimiter_lines.append(getattr(node, "lineno", 0))
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
                    and isinstance(node.targets[0], ast.Tuple)
                    and isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Attribute)
                    and isinstance(node.value.func.value, ast.Name)
                    and node.value.func.value.id == "helpers"
                    and node.value.func.attr in append_helper_methods
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
                        if _is_fixed_phase_quota_name(target.id) and _is_literal_int(node.value):
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
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                if node.target.id == "stepsLeft":
                    manual_stepsleft_mutations.append(getattr(node, "lineno", 0))
                if node.target.id == "remainingSteps":
                    assigns_remaining_steps = True
                if node.target.id not in {"generated", "stepsLeft", "next_token", "new_steps"}:
                    extra_state.add(node.target.id)
                    if _is_fixed_phase_quota_name(node.target.id) and _is_literal_int(node.value):
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
            if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
                if node.target.id == "stepsLeft":
                    manual_stepsleft_mutations.append(getattr(node, "lineno", 0))
                if (
                    node.target.id not in {"generated", "stepsLeft", "next_token", "new_steps"}
                    and any(isinstance(inner, ast.Constant) and isinstance(inner.value, float) for inner in ast.walk(node.value))
                ):
                    mutable_float_state.add(node.target.id)

        if old_api_calls:
            return (
                "The body uses the old delimiter-based API which has been replaced. "
                "Remove these calls: " + ", ".join(sorted(old_api_calls)) + ". "
                "Prefer helpers.AppendUnconstrainedStep, helpers.AppendConstrainedStep, "
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
        total_step_calls = constrained_step_calls + forced_token_calls + unconstrained_calls
        if total_step_calls == 0:
            return (
                "The body must call at least one step method "
                "(AppendConstrainedStep, AppendLeftDelimiter, AppendUnconstrainedStep, "
                "ConstrainedStep, ForcedTokenStep, UnconstrainedStep, etc.)."
            )
        if constrained_step_calls == 0:
            return (
                "The body must include at least one constrained step "
                "(helpers.ConstrainedStep, helpers.SoftConstrainedStep, helpers.TopKConstrainedStep, "
                "or their Append* wrappers) "
                "to produce grammar-valid answer content."
            )
        if not emits_left_delimiter or not emits_right_delimiter:
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
                "the Dafny lowering is proof-hostile. Use integer counters for state, and pass a literal "
                "positive penalty directly to AppendSoftConstrainedStep when needed."
            )
        if bad_bias_helper_lines:
            return (
                "helpers.UnconstrainedBiasLeftDelimiterStep requires a literal positive float bias, "
                "e.g. `helpers.UnconstrainedBiasLeftDelimiterStep(prompt, generated, 5.0, stepsLeft)`. "
                "Do not store the bias in an int variable such as `biasStrength = 3`. "
                f"First invalid bias argument is near line {bad_bias_helper_lines[0]}."
            )
        if os.environ.get("CSD_REQUIRE_NATURAL_DELIMITERS", "").strip() in {"1", "true", "True"}:
            if not uses_natural_left_delimiter or not uses_natural_right_delimiter:
                missing = []
                if not uses_natural_left_delimiter:
                    missing.append("UnconstrainedAllowLeftDelimiterStep or UnconstrainedNudgeLeftDelimiterStep")
                if not uses_natural_right_delimiter:
                    missing.append("ConstrainedOrRightDelimiterStep")
                return (
                    "This GSM run requires natural delimiter decisions rather than forced delimiter phases. "
                    "Use `helpers.UnconstrainedAllowLeftDelimiterStep(...)` or "
                    "`helpers.UnconstrainedNudgeLeftDelimiterStep(...)` during free-form reasoning so "
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
            if "SpacedLeftDelimiter" not in strategy_body and '" <<"' not in strategy_body and "' <<'" not in strategy_body:
                return (
                    "When using UnconstrainedAllowLeftDelimiterStep, handle both left-delimiter tokenizations: "
                    "`LeftDelimiter` and `SpacedLeftDelimiter` (or the literal string `\" <<\"`). "
                    "Otherwise a natural spaced ` <<` can be missed and a later delimiter can create nested spans."
                )
        if fixed_phase_quota_lines:
            return (
                "Do not introduce fixed phase-quota constants such as `min_reason_steps`, "
                "`reason_limit`, `max_answer_steps`, or `answer_budget`. For GSM, prefer adaptive interleaving: "
                "reason freely, optionally emit complete verified spans, and close spans using parser "
                "completion plus semantic/budget signals rather than fixed token-count quotas. "
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
                "Complete the helper call, e.g. `generated, stepsLeft = helpers.AppendExtendConstrainedStep(...)`, "
                f"near line {malformed_tuple_assignment_lines[0]}."
            )
        if manual_stepsleft_mutations:
            return (
                "Do not manually increment, decrement, or recompute `stepsLeft`; helper calls already consume "
                "budget and preserve the proof invariant. Only update `stepsLeft` from helper returns, such as "
                "`generated, stepsLeft = helpers.Append...(...)` or `stepsLeft = new_steps` after a raw step. "
                f"First manual mutation is near line {manual_stepsleft_mutations[0]}."
            )
        if topk_unprovable_calls:
            return (
                "TopK constrained helpers are only verifier-friendly with a literal k of 1 unless you also prove "
                "`k <= |lm.Tokens|`. Prefer helpers.AppendConstrainedStep(...) or use "
                "`helpers.AppendTopKConstrainedStep(prompt, generated, 1, stepsLeft)`. Offending helpers: "
                + ", ".join(sorted(topk_unprovable_calls)) + "."
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
                "Append* helper calls return (updated_prefix, remaining_steps) and must be assigned back into "
                "`generated, stepsLeft`, not token variables like `next_token`. Offending helpers: "
                + ", ".join(sorted(append_helper_wrong_targets)) + "."
            )
        if unguarded_constrained_calls:
            return (
                "Every constrained helper call must be inside a branch or loop condition that explicitly checks "
                "`helpers.CanConstrain(generated)`, `helpers.CanExtendConstrained(generated)`, or "
                "`parser.IsCompletePrefix(helpers.LongestValidSuffix(generated))`. "
                "Unguarded calls: " + ", ".join(sorted(unguarded_constrained_calls)) + "."
            )
        if unguarded_right_delimiter_calls:
            return (
                "RightDelimiter emission must be inside a branch whose condition explicitly checks "
                "`parser.IsCompletePrefix(helpers.LongestValidSuffix(generated))`. Do not close the answer "
                "span unconditionally or merely because a phase variable changed. Offending calls: "
                + ", ".join(sorted(unguarded_right_delimiter_calls)) + "."
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
        if constrained_lines and left_delimiter_lines and min(constrained_lines) < min(left_delimiter_lines):
            return (
                "Constrained answer-token helpers must appear after executable LeftDelimiter emission in the method body. "
                "Do not generate constrained answer content before `helpers.AppendLeftDelimiter(...)`."
            )
        if (
            not uses_natural_right_delimiter
            and constrained_lines
            and right_delimiter_lines
            and min(right_delimiter_lines) <= min(constrained_lines)
        ):
            return (
                "RightDelimiter emission must appear after constrained answer-token helpers, and only after "
                "`parser.IsCompletePrefix(helpers.LongestValidSuffix(generated))` is true."
            )
        if nondecreasing_else_lines:
            return (
                "Branches inside a `# decreases stepsLeft` loop must either consume a helper step or `break`; "
                "do not use an `else` branch that only changes phase/state and loops again. "
                f"First non-consuming else branch is near line {nondecreasing_else_lines[0]}."
            )
        for while_node in while_nodes:
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
                "`parser.IsCompletePrefix(helpers.LongestValidSuffix(generated))` before an open-ended "
                "`helpers.CanConstrain(generated)` / `helpers.CanExtendConstrained(generated)` branch. "
                "Otherwise a complete expression can keep extending forever as a valid prefix and never close. "
                "Move the complete-prefix close/extend branch earlier, or add an explicit "
                "`and not parser.IsCompletePrefix(helpers.LongestValidSuffix(generated))` guard on the "
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
        for repair_round in range(1, max_repairs + 1):
            issue = self._structural_issue(current)
            if issue is None:
                return current
            repair_record: dict[str, object] = {
                "round": repair_round,
                "input_strategy_length": len(current),
                "input_strategy": self._diagnostic_excerpt(current),
                "issue": issue,
            }
            system_prompt, user_prompt = build_structure_repair_prompt(current, issue)
            repaired_raw = self._generate_text(system_prompt, user_prompt)
            repaired = self._extract_strategy(repaired_raw)
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
            return current
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
        ):
            score += 10
        if "ConstrainedOrRightDelimiterStep" in helper_calls:
            score += 18
        if {"ForcedTokenStep", "AppendForcedToken", "AppendLeftDelimiter", "AppendRightDelimiter"} & helper_calls:
            score += 8
        if "UnconstrainedAllowLeftDelimiterStep" in helper_calls:
            score += 24
        if "UnconstrainedBiasLeftDelimiterStep" in helper_calls:
            score += 28
        if "UnconstrainedNudgeLeftDelimiterStep" in helper_calls:
            score += 32
        if "UnconstrainedStep" in helper_calls:
            score += 14
        if "AppendUnconstrainedStep" in helper_calls:
            score += 3
        if {
            "SoftConstrainedStep",
            "TopKConstrainedStep",
            "AppendSoftConstrainedStep",
            "AppendTopKConstrainedStep",
        } & helper_calls:
            score += 12
        if "BudgetAwareStep" in helper_calls or "AppendBudgetAwareStep" in helper_calls:
            score += 10
        if "LongestValidSuffix" in helper_calls or "CanConstrain" in helper_calls:
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
        budgets = [
            max(self.max_new_tokens, self.MIN_STRATEGY_TOKENS),
            max(self.max_new_tokens, 320),
            max(self.max_new_tokens, 384),
            max(self.max_new_tokens, 640),
            max(self.max_new_tokens, 800),
        ]
        while len(budgets) < self.SEARCH_ATTEMPTS:
            budgets.append(budgets[-1])
        temperatures = [
            max(self.temperature, 0.85),
            max(self.temperature, 0.65),
            min(self.temperature, 0.35),
            max(self.temperature, 0.75),
            min(self.temperature, 0.25),
        ]
        while len(temperatures) < self.SEARCH_ATTEMPTS:
            temperatures.append(temperatures[-1])
        budgets = budgets[: self.SEARCH_ATTEMPTS]
        temperatures = temperatures[: self.SEARCH_ATTEMPTS]

        last_error: str | None = None
        current_system = system_prompt
        current_user = user_prompt
        valid_candidates: list[tuple[int, str]] = []
        self.last_generation_diagnostics = []

        for idx, (budget, temp) in enumerate(zip(budgets, temperatures), start=1):
            raw_output = self._generate_text(
                current_system,
                current_user,
                max_new_tokens=budget,
                temperature=temp,
            )
            strategy = self._extract_strategy(raw_output)
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
                strategy = self._ensure_nontrivial_strategy(strategy)
                novelty_score = self._novelty_score(strategy)
                diagnostic["accepted"] = True
                diagnostic["novelty_score"] = novelty_score
                diagnostic["final_strategy_length"] = len(strategy)
                diagnostic["final_strategy"] = self._diagnostic_excerpt(strategy)
                valid_candidates.append((novelty_score, strategy))
                current_system, current_user = system_prompt, user_prompt
                continue
            except ValueError as exc:
                last_error = str(exc)
                diagnostic["issue"] = last_error
                diagnostic["structure_repair_trace"] = [
                    dict(item) for item in getattr(self, "last_structure_repair_trace", [])
                ]
                current_system, current_user = build_structure_repair_prompt(
                    strategy or raw_output or "# CSD_RATIONALE_BEGIN\n# Empty output.\n# CSD_RATIONALE_END",
                    last_error,
                )
                print(
                    f"  Initial generation attempt {idx} produced an invalid body; "
                    f"retrying with a stricter repair prompt ({last_error})."
                )
            finally:
                self.last_generation_diagnostics.append(diagnostic)

        if valid_candidates:
            best_score, best_strategy = max(valid_candidates, key=lambda item: item[0])
            print(f"  Selected the most novel structurally valid candidate (score={best_score}).")
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
        system_prompt, user_prompt = build_initial_prompt(task_description)
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
            previous_strategy, error_message
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
            previous_strategy, error_traceback
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
            previous_strategy, error_message
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
            previous_strategy, evaluation_feedback
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
        start = self._template.find(self.STRATEGY_BEGIN_MARKER)
        end = self._template.find(self.STRATEGY_END_MARKER)
        if start == -1 or end == -1 or end < start:
            raise ValueError("Strategy hole markers not found in generation/csd/GeneratedAgentTemplate.py")
        end += len(self.STRATEGY_END_MARKER)
        return self._template[:start] + indented + self._template[end:]
    
    def get_template(self) -> str:
        """Get the raw template content."""
        return self._template
