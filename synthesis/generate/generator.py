"""
Strategy generator for CSD synthesis.

Supports HuggingFace, vLLM, OpenAI Chat Completions, and Amazon Bedrock Converse.
"""

import os
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

import torch

from synthesis.evaluate.benchmarks.common.model_utils import _configure_vllm_multiprocessing

from .prompts import (
    build_initial_prompt,
    build_verification_error_prompt,
    build_runtime_error_prompt,
    build_compilation_error_prompt,
    build_format_repair_prompt,
    build_evaluation_failure_prompt,
)
from .rationale import extract_rationale


class StrategyGenerator:
    """
    Generates Dafny CSD strategies.

    Supports local HuggingFace inference, local vLLM inference, OpenAI-hosted
    models, or Amazon Bedrock Converse (e.g. Claude where configured).
    """

    # Default model - can be overridden
    DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"

    # Path to the template file
    TEMPLATE_PATH = Path(__file__).parent.parent / "verify" / "library" / "GeneratedCSD.dfy"

    # Marker in template to replace
    STRATEGY_MARKER = "// QWEN_INSERT_STRATEGY_HERE"

    def __init__(
        self,
        model_name: Optional[str] = None,
        backend: str = "huggingface",
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        vllm_tensor_parallel_size: Optional[int] = None,
        vllm_pipeline_parallel_size: int = 1,
        vllm_gpu_memory_utilization: float = 0.8,
        vllm_max_model_len: int = 4096,
        vllm_enforce_eager: bool = True,
        api_base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """
        Initialize the strategy generator.

        Args:
            model_name: HuggingFace model name (default: Qwen2.5-Coder-7B-Instruct)
            backend: Inference backend ("huggingface", "vllm", "openai", or "bedrock")
            device: Device to run on ('cuda', 'mps', 'cpu', or None for auto)
            torch_dtype: Torch dtype for model (default: auto based on device)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p (nucleus) sampling parameter
            load_in_4bit: Load model in 4-bit quantization
            load_in_8bit: Load model in 8-bit quantization
            vllm_tensor_parallel_size: Explicit tensor parallel size for vLLM
            vllm_pipeline_parallel_size: Explicit pipeline parallel size for vLLM
            vllm_gpu_memory_utilization: GPU memory fraction reserved by vLLM
            vllm_max_model_len: Max context length passed to vLLM
            vllm_enforce_eager: Disable cudagraph/compile in vLLM for stability
            api_base_url: Optional base URL override (OpenAI: OPENAI_BASE_URL; Bedrock: BEDROCK_BASE_URL)
            api_key: Optional API key (OpenAI: OPENAI_API_KEY; Bedrock: AWS_BEARER_TOKEN_BEDROCK)
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        self.backend = backend
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit
        self.vllm_tensor_parallel_size = vllm_tensor_parallel_size
        self.vllm_pipeline_parallel_size = vllm_pipeline_parallel_size
        self.vllm_gpu_memory_utilization = vllm_gpu_memory_utilization
        self.vllm_max_model_len = vllm_max_model_len
        self.vllm_enforce_eager = vllm_enforce_eager
        self.api_base_url = api_base_url or self._default_api_base_url(backend)
        self.api_key = api_key or self._default_api_key(backend)

        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        # Auto-detect dtype
        if torch_dtype is None:
            if device == "cuda":
                torch_dtype = torch.bfloat16
            else:
                torch_dtype = torch.float32
        self.torch_dtype = torch_dtype

        # Lazy loading - model loaded on first use
        self._model = None
        self._tokenizer = None
        self._client = None
        self._vllm = None
        self._current_task_description: Optional[str] = None
        self._prompt_log_counter = 0
        self._synthesis_context: Optional[dict[str, str]] = None

        # Load template
        self._template = self._load_template()

    @staticmethod
    def _default_api_base_url(backend: str) -> Optional[str]:
        if backend == "openai":
            return os.environ.get("OPENAI_BASE_URL")
        if backend == "bedrock":
            region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"
            return os.environ.get(
                "BEDROCK_BASE_URL",
                f"https://bedrock-runtime.{region}.amazonaws.com",
            )
        return None

    @staticmethod
    def _default_api_key(backend: str) -> Optional[str]:
        if backend == "openai":
            return os.environ.get("OPENAI_API_KEY")
        if backend == "bedrock":
            return os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
        return None

    def _get_vllm_quantization_kwargs(self) -> dict:
        """Translate local quantization flags to the installed vLLM config surface."""
        if self.load_in_4bit and self.load_in_8bit:
            raise ValueError("Choose at most one of load_in_4bit or load_in_8bit.")

        if not (self.load_in_4bit or self.load_in_8bit):
            return {}

        quant_config = {
            "quant_method": "bitsandbytes",
        }
        if self.load_in_4bit:
            quant_config.update(
                {
                    "load_in_4bit": True,
                    "bnb_4bit_compute_dtype": "bfloat16",
                    "bnb_4bit_quant_type": "nf4",
                    "bnb_4bit_use_double_quant": True,
                }
            )
        else:
            quant_config.update(
                {
                    "load_in_8bit": True,
                }
            )

        return {
            "quantization": "bitsandbytes",
            "hf_overrides": {
                "quantization_config": quant_config,
            },
        }

    def _load_template(self) -> str:
        """Load the GeneratedCSD.dfy template."""
        if not self.TEMPLATE_PATH.exists():
            raise FileNotFoundError(
                f"Template not found at {self.TEMPLATE_PATH}. "
                "Make sure GeneratedCSD.dfy exists in synthesis/verify/library/."
            )
        return self.TEMPLATE_PATH.read_text()

    def set_synthesis_context(
        self,
        eval_model: str,
        dataset: str,
        max_steps: int,
        step_token_budget: int,
    ) -> None:
        """Store runtime evaluation context for inclusion in synthesis prompts."""
        self._synthesis_context = {
            "eval_model": eval_model,
            "dataset": dataset,
            "max_steps": str(max_steps),
            "step_token_budget": str(step_token_budget),
        }

    def _synthesis_context_block(self) -> str:
        if not self._synthesis_context:
            return ""
        ctx = self._synthesis_context
        return (
            "\n\n## Runtime Context\n"
            f"- Evaluation model: {ctx['eval_model']}\n"
            f"- Dataset: {ctx['dataset']}\n"
            f"- maxSteps budget: {ctx['max_steps']}\n"
            f"- stepTokenBudget: {ctx['step_token_budget']}\n"
        )
    
    def _ensure_backend_loaded(self) -> None:
        """Lazy-load the selected backend."""
        if self.backend == "openai":
            if self._client is None:
                if not self.api_key:
                    raise ValueError(
                        "OPENAI_API_KEY is required when --generation-backend=openai"
                    )
                from openai import OpenAI

                client_kwargs = {"api_key": self.api_key}
                if self.api_base_url:
                    client_kwargs["base_url"] = self.api_base_url
                self._client = OpenAI(**client_kwargs)
            return

        if self.backend == "bedrock":
            if not self.api_key:
                raise ValueError(
                    "AWS_BEARER_TOKEN_BEDROCK is required when --generation-backend=bedrock"
                )
            return

        if self.backend == "vllm":
            if self._vllm is None:
                if not self.device.startswith("cuda"):
                    raise ValueError("vLLM generation currently requires CUDA in this project.")

                _configure_vllm_multiprocessing()
                from vllm import LLM
                from vllm.transformers_utils.tokenizer import get_tokenizer

                tensor_parallel_size = self.vllm_tensor_parallel_size or max(1, torch.cuda.device_count())
                self._tokenizer = get_tokenizer(self.model_name, trust_remote_code=True)
                vllm_kwargs = self._get_vllm_quantization_kwargs()
                self._vllm = LLM(
                    model=self.model_name,
                    tokenizer=self.model_name,
                    trust_remote_code=True,
                    tensor_parallel_size=tensor_parallel_size,
                    pipeline_parallel_size=self.vllm_pipeline_parallel_size,
                    gpu_memory_utilization=self.vllm_gpu_memory_utilization,
                    max_model_len=self.vllm_max_model_len,
                    enforce_eager=self.vllm_enforce_eager,
                    **vllm_kwargs,
                )
            return

        if self.backend != "huggingface":
            raise ValueError(f"Unsupported generation backend: {self.backend}")

        if self._model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

            print(f"Loading {self.model_name}...")
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )

            # Prepare quantization config if needed
            quantization_config = None
            if self.load_in_4bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            elif self.load_in_8bit:
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)

            # Try loading on requested device, fallback to CPU on CUDA OOM
            try:
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=self.torch_dtype,
                    device_map="auto" if self.device.startswith("cuda") else (self.device if self.device != "mps" else None),
                    trust_remote_code=True,
                    quantization_config=quantization_config,
                )
                if self.device == "mps":
                    self._model = self._model.to(self.device)
                print(f"Model loaded on {self.device}")
            except RuntimeError as e:
                error_str = str(e).lower()
                if self.device in ["cuda", "mps"] and ("out of memory" in error_str or "cuda" in error_str):
                    print(f"⚠️  {self.device.upper()} out of memory: {e}")
                    print(f"   Falling back to CPU (this will be slower)...")

                    # Clear CUDA cache if available
                    if self.device == "cuda":
                        torch.cuda.empty_cache()

                    # Retry on CPU
                    self.device = "cpu"
                    self.torch_dtype = torch.float32
                    self._model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        torch_dtype=self.torch_dtype,
                        trust_remote_code=True,
                    ).to(self.device)
                    print(f"Model loaded on {self.device} (CPU fallback)")
                else:
                    raise

    def _generate_text(self, system_prompt: str, user_prompt: str) -> str:
        """
        Generate text using Qwen.

        Args:
            system_prompt: System message
            user_prompt: User message

        Returns:
            Generated text
        """
        self._ensure_backend_loaded()

        system_prompt = system_prompt + self._synthesis_context_block()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        if self.backend == "openai":
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_completion_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
            )
            content = response.choices[0].message.content or ""
            output = content.strip()
            self._log_prompt_io(system_prompt, user_prompt, output)
            return output

        if self.backend == "bedrock":
            output = self._generate_bedrock(system_prompt, user_prompt)
            self._log_prompt_io(system_prompt, user_prompt, output)
            return output

        if self.backend == "vllm":
            from vllm import SamplingParams

            text = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            sampling_params = SamplingParams(
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
            )
            outputs = self._vllm.generate([text], sampling_params=sampling_params, use_tqdm=False)
            output = outputs[0].outputs[0].text.strip()
            self._log_prompt_io(system_prompt, user_prompt, output)
            return output

        # Apply chat template
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize
        inputs = self._tokenizer(text, return_tensors="pt").to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                pad_token_id=self._tokenizer.eos_token_id
            )

        # Decode only the new tokens
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        response = self._tokenizer.decode(generated, skip_special_tokens=True)

        output = response.strip()
        self._log_prompt_io(system_prompt, user_prompt, output)
        return output

    def _post_json(self, url: str, headers: dict[str, str], payload: dict) -> dict:
        max_retries = int(os.environ.get("CSD_API_MAX_RETRIES", "5"))
        retry_base_seconds = float(os.environ.get("CSD_API_RETRY_BASE_SECONDS", "20"))
        retryable_statuses = {408, 409, 429, 500, 502, 503, 504, 529}
        for attempt in range(max_retries + 1):
            request = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json", **headers},
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=300) as response:
                    body = response.read().decode("utf-8")
                break
            except urllib.error.HTTPError as exc:
                error_body = exc.read().decode("utf-8", errors="replace")
                if exc.code in retryable_statuses and attempt < max_retries:
                    sleep_seconds = retry_base_seconds * (2 ** attempt)
                    print(
                        f"[api-retry] {self.backend} HTTP {exc.code}; "
                        f"retry {attempt + 1}/{max_retries} after {sleep_seconds:.1f}s: "
                        f"{error_body[:300]}",
                        flush=True,
                    )
                    time.sleep(sleep_seconds)
                    continue
                raise RuntimeError(
                    f"{self.backend} generation API returned HTTP {exc.code}: {error_body[:1000]}"
                ) from exc
        return json.loads(body)

    def _generate_bedrock(self, system_prompt: str, user_prompt: str) -> str:
        base_url = (self.api_base_url or self._default_api_base_url("bedrock") or "").rstrip("/")
        model = urllib.parse.quote(self.model_name, safe="")
        url = f"{base_url}/model/{model}/converse"
        payload = {
            "system": [{"text": system_prompt}],
            "messages": [
                {
                    "role": "user",
                    "content": [{"text": user_prompt}],
                }
            ],
            "inferenceConfig": {
                "maxTokens": self.max_new_tokens,
                "temperature": self.temperature,
                "topP": self.top_p,
            },
        }
        data = self._post_json(
            url,
            {
                "Authorization": f"Bearer {self.api_key or ''}",
                "Accept": "application/json",
            },
            payload,
        )
        parts = data.get("output", {}).get("message", {}).get("content") or []
        return "".join(part.get("text", "") for part in parts).strip()

    def _log_prompt_io(self, system_prompt: str, user_prompt: str, output: str) -> None:
        """Optionally persist exact prompt/response records for debugging."""
        log_dir = os.environ.get("CSD_PROMPT_LOG_DIR")
        if not log_dir:
            return

        self._prompt_log_counter += 1
        path = Path(log_dir)
        path.mkdir(parents=True, exist_ok=True)
        record = {
            "index": self._prompt_log_counter,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "backend": self.backend,
            "model": self.model_name,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "output": output,
        }
        with (path / "prompt_io.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def _extract_strategy(self, raw_output: str) -> str:
        """
        Extract the strategy expression from Qwen's output.

        Handles cases where Qwen includes extra text, code blocks, etc.

        Args:
            raw_output: Raw text from Qwen

        Returns:
            Cleaned strategy expression
        """
        # Remove markdown code blocks if present (handles both complete and truncated blocks)
        # First try complete code block
        code_block_pattern = r"```(?:dafny)?\s*([\s\S]*?)```"
        match = re.search(code_block_pattern, raw_output)
        if match:
            raw_output = match.group(1)
        else:
            # Handle truncated code blocks (no closing fence due to token limit)
            truncated_pattern = r"^```(?:dafny)?\s*([\s\S]*)$"
            match = re.search(truncated_pattern, raw_output.strip())
            if match:
                raw_output = match.group(1)

        # Remove leading/trailing whitespace
        strategy = raw_output.strip()

        # Heuristic repair: Dafny uses `+` for sequence concatenation; `++` is invalid.
        # Replace `++` when it is used like an operator (surrounded by optional whitespace).
        # This avoids touching tokens like "C++" inside words.
        strategy = re.sub(r"\s*\+\+\s*", " + ", strategy)

        # If it looks like a full function/method definition, extract just the body.
        # We match the *final* '}' so nested braces inside the body are preserved.
        lowered = strategy.lower()
        if ("function" in lowered or "method" in lowered) and "{" in strategy:
            brace_match = re.search(r"\{([\s\S]*)\}\s*$", strategy)
            if brace_match:
                strategy = brace_match.group(1).strip()

        # Ensure the body ends in a reasonable terminator.
        # - Single statements should end with ';'
        # - Block bodies may end with '}' (e.g., if/else/while blocks)
        if strategy:
            last_char = strategy.rstrip()[-1]
            if last_char not in {";", "}"}:
                strategy = strategy.rstrip() + ";"

        return strategy

    def _ensure_rationale_block(
        self,
        strategy_body: str,
        *,
        max_repairs: int = 2,
        search_memory: str = "",
        allowed_helpers: list[str] | None = None,
    ) -> str:
        """
        Ensure the strategy body contains the required rationale markers.

        If missing, attempt a small number of "format repair" generations that rewrite
        the body into the required structure without changing semantics.
        """
        extracted = extract_rationale(strategy_body)
        if extracted.rationale is not None and extracted.has_markers:
            return strategy_body

        current = strategy_body
        for _ in range(max_repairs):
            system_prompt, user_prompt = build_format_repair_prompt(
                current,
                search_memory,
                allowed_helpers=allowed_helpers,
            )
            repaired_raw = self._generate_text(system_prompt, user_prompt)
            repaired = self._extract_strategy(repaired_raw)
            extracted = extract_rationale(repaired)
            if extracted.rationale is not None and extracted.has_markers:
                return repaired
            current = repaired

        return "// CSD_RATIONALE_BEGIN\n// (Auto-injected rationale)\n// CSD_RATIONALE_END\n" + current

    def generate_initial(
        self,
        task_description: str,
        allowed_helpers: list[str] | None = None,
    ) -> str:
        """
        Generate an initial strategy for the given task.

        Args:
            task_description: Description of what the strategy should accomplish

        Returns:
            Strategy expression (Dafny code)
        """
        self._current_task_description = task_description
        system_prompt, user_prompt = build_initial_prompt(
            task_description,
            allowed_helpers=allowed_helpers,
        )
        raw_output = self._generate_text(system_prompt, user_prompt)
        strategy = self._extract_strategy(raw_output)
        return self._ensure_rationale_block(strategy, allowed_helpers=allowed_helpers)

    def refine_after_verification_error(
        self,
        previous_strategy: str,
        error_message: str,
        behavioral_context: str = "",
        structured_feedback: str = "",
        error_history: str = "",
        strategy_context: str = "",
        search_memory: str = "",
        allowed_helpers: list[str] | None = None,
    ) -> str:
        """
        Generate a refined strategy after verification failure.

        Args:
            previous_strategy: The strategy that failed
            error_message: Dafny verification error

        Returns:
            New strategy expression
        """
        task_description = self._current_task_description or "Unknown task"
        system_prompt, user_prompt = build_verification_error_prompt(
            task_description,
            previous_strategy,
            error_message,
            behavioral_context,
            structured_feedback,
            error_history,
            strategy_context,
            search_memory,
            allowed_helpers=allowed_helpers,
        )
        raw_output = self._generate_text(system_prompt, user_prompt)
        strategy = self._extract_strategy(raw_output)
        return self._ensure_rationale_block(
            strategy,
            search_memory=search_memory,
            allowed_helpers=allowed_helpers,
        )

    def refine_after_runtime_error(
        self,
        previous_strategy: str,
        error_traceback: str,
        search_memory: str = "",
        allowed_helpers: list[str] | None = None,
    ) -> str:
        """
        Generate a refined strategy after runtime failure.

        Args:
            previous_strategy: The strategy that failed
            error_traceback: Python traceback

        Returns:
            New strategy expression
        """
        task_description = self._current_task_description or "Unknown task"
        system_prompt, user_prompt = build_runtime_error_prompt(
            previous_strategy,
            error_traceback,
            task_description,
            search_memory,
            allowed_helpers=allowed_helpers,
        )
        raw_output = self._generate_text(system_prompt, user_prompt)
        strategy = self._extract_strategy(raw_output)
        return self._ensure_rationale_block(
            strategy,
            search_memory=search_memory,
            allowed_helpers=allowed_helpers,
        )

    def refine_after_compilation_error(
        self,
        previous_strategy: str,
        error_message: str,
        search_memory: str = "",
        allowed_helpers: list[str] | None = None,
    ) -> str:
        """
        Generate a refined strategy after compilation failure.

        Args:
            previous_strategy: The strategy that failed
            error_message: Dafny compilation error

        Returns:
            New strategy expression
        """
        system_prompt, user_prompt = build_compilation_error_prompt(
            previous_strategy,
            error_message,
            search_memory,
            allowed_helpers=allowed_helpers,
        )
        raw_output = self._generate_text(system_prompt, user_prompt)
        strategy = self._extract_strategy(raw_output)
        return self._ensure_rationale_block(
            strategy,
            search_memory=search_memory,
            allowed_helpers=allowed_helpers,
        )

    def refine_after_evaluation_failure(
        self,
        previous_strategy: str,
        evaluation_feedback: str,
        evaluation_history: str = "",
        working_hypothesis: str = "",
        search_memory: str = "",
        allowed_helpers: list[str] | None = None,
    ) -> str:
        """
        Generate a refined strategy after evaluation failure.

        The strategy passed verification, compilation, and runtime testing,
        but performed poorly on actual dataset evaluation (low accuracy,
        format rate, syntax rate, or semantic rate).

        Args:
            previous_strategy: The strategy that failed evaluation
            evaluation_feedback: Feedback summary from the evaluator
            evaluation_history: Recent evaluated attempts and outcomes
            working_hypothesis: Compact strategy lineage and balanced-best state

        Returns:
            New strategy expression
        """
        task_description = self._current_task_description or "Unknown task"
        system_prompt, user_prompt = build_evaluation_failure_prompt(
            task_description,
            previous_strategy,
            evaluation_feedback,
            evaluation_history,
            working_hypothesis,
            search_memory,
            allowed_helpers=allowed_helpers,
        )
        raw_output = self._generate_text(system_prompt, user_prompt)
        strategy = self._extract_strategy(raw_output)
        return self._ensure_rationale_block(
            strategy,
            search_memory=search_memory,
            allowed_helpers=allowed_helpers,
        )


    def inject_strategy(self, strategy: str) -> str:
        """
        Inject a strategy into the template.

        Args:
            strategy: Strategy expression to inject

        Returns:
            Complete Dafny source code
        """
        return self._template.replace(self.STRATEGY_MARKER, strategy)

    def get_template(self) -> str:
        """Get the raw template content."""
        return self._template
