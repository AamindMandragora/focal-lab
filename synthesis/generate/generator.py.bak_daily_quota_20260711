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
        anthropic_thinking: str = "auto",
        anthropic_thinking_budget_tokens: int = 4096,
        anthropic_effort: str = "xhigh",
        anthropic_thinking_display: str = "summarized",
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
            anthropic_thinking: Anthropic thinking mode: auto, off, adaptive, or enabled.
            anthropic_thinking_budget_tokens: Manual thinking budget for models that still accept it.
            anthropic_effort: Anthropic adaptive-thinking effort level.
            anthropic_thinking_display: Whether Anthropic should return summarized or omitted thinking.
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
        self.anthropic_thinking = anthropic_thinking
        self.anthropic_thinking_budget_tokens = anthropic_thinking_budget_tokens
        self.anthropic_effort = anthropic_effort
        self.anthropic_thinking_display = anthropic_thinking_display

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
        self._summary_client = None
        self._summary_anthropic_client = None

        # Load template
        self._template = self._load_template()

    @staticmethod
    def _default_api_base_url(backend: str) -> Optional[str]:
        if backend == "openai":
            return os.environ.get("OPENAI_BASE_URL")
        if backend == "bedrock":
            return os.environ.get("BEDROCK_BASE_URL")
        if backend == "gemini":
            return os.environ.get(
                "GEMINI_BASE_URL",
                "https://generativelanguage.googleapis.com/v1beta",
            )
        if backend == "vertex":
            location = (
                os.environ.get("VERTEX_AI_LOCATION")
                or os.environ.get("GOOGLE_CLOUD_LOCATION")
                or os.environ.get("GOOGLE_VERTEX_LOCATION")
                or "global"
            )
            if location == "global":
                return os.environ.get("VERTEX_AI_BASE_URL", "https://aiplatform.googleapis.com/v1")
            return os.environ.get(
                "VERTEX_AI_BASE_URL",
                f"https://{location}-aiplatform.googleapis.com/v1",
            )
        return None

    @staticmethod
    def _bedrock_runtime_base_url() -> str:
        region = (
            os.environ.get("AWS_REGION")
            or os.environ.get("AWS_DEFAULT_REGION")
            or "us-east-1"
        )
        return os.environ.get(
            "BEDROCK_BASE_URL",
            f"https://bedrock-runtime.{region}.amazonaws.com",
        )

    @staticmethod
    def _default_api_key(backend: str) -> Optional[str]:
        if backend == "openai":
            return os.environ.get("OPENAI_API_KEY")
        if backend == "bedrock":
            return os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
        if backend == "anthropic":
            return os.environ.get("ANTHROPIC_API_KEY")
        if backend == "gemini":
            return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if backend == "vertex":
            return os.environ.get("VERTEX_AI_ACCESS_TOKEN")
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

        if self.backend == "anthropic":
            if self._client is None:
                if not self.api_key:
                    raise ValueError(
                        "ANTHROPIC_API_KEY is required when --generation-backend=anthropic"
                    )
                from anthropic import Anthropic
                client_kwargs = {"api_key": self.api_key}
                if self.api_base_url:
                    client_kwargs["base_url"] = self.api_base_url
                self._client = Anthropic(**client_kwargs)
            return

        if self.backend == "gemini":
            if not self.api_key:
                raise ValueError(
                    "GEMINI_API_KEY or GOOGLE_API_KEY is required when --generation-backend=gemini"
                )
            return

        if self.backend == "vertex":
            return

        if self.backend == "vllm":
            if self._vllm is None:
                if not self.device.startswith("cuda"):
                    raise ValueError("vLLM generation currently requires CUDA in this project.")

                _configure_vllm_multiprocessing()
                from vllm import LLM
                from vllm.transformers_utils.tokenizer import get_tokenizer

                from synthesis.evaluate.benchmarks.common.model_utils import (
                    resolve_vllm_tensor_parallel_size,
                )

                tensor_parallel_size = resolve_vllm_tensor_parallel_size(self.vllm_tensor_parallel_size)
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

    def _is_opus47(self) -> bool:
        return "claude-opus-4-7" in self.model_name

    def _anthropic_thinking_kwargs(self) -> dict[str, object]:
        mode = self.anthropic_thinking
        if mode == "auto":
            mode = "adaptive" if self._is_opus47() else "off"
        if mode == "off":
            return {}
        if mode not in {"adaptive", "enabled"}:
            raise ValueError(
                "anthropic_thinking must be one of: auto, off, adaptive, enabled"
            )
        if self.anthropic_thinking_display not in {"omitted", "summarized"}:
            raise ValueError(
                "anthropic_thinking_display must be 'omitted' or 'summarized'"
            )

        if mode == "adaptive":
            allowed_efforts = {"low", "medium", "high", "xhigh", "max"}
            if self.anthropic_effort not in allowed_efforts:
                raise ValueError(
                    "anthropic_effort must be one of: low, medium, high, xhigh, max"
                )
            return {
                "thinking": {
                    "type": "adaptive",
                    "display": self.anthropic_thinking_display,
                },
                "output_config": {"effort": self.anthropic_effort},
            }

        if self._is_opus47():
            raise ValueError(
                "claude-opus-4-7 does not support manual Anthropic thinking "
                "with budget_tokens; use anthropic_thinking='adaptive'."
            )
        if self.anthropic_thinking_budget_tokens < 1024:
            raise ValueError("anthropic_thinking_budget_tokens must be at least 1024")
        if self.anthropic_thinking_budget_tokens >= self.max_new_tokens:
            raise ValueError(
                "anthropic_thinking_budget_tokens must be less than max_new_tokens"
            )
        return {
            "thinking": {
                "type": "enabled",
                "budget_tokens": self.anthropic_thinking_budget_tokens,
                "display": self.anthropic_thinking_display,
            }
        }

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
            request_kwargs = {
                "model": self.model_name,
                "messages": messages,
                "max_completion_tokens": self.max_new_tokens,
            }
            if not self.model_name.startswith("gpt-5"):
                request_kwargs.update(
                    {
                        "temperature": self.temperature,
                        "top_p": self.top_p,
                    }
                )
            response = self._client.chat.completions.create(**request_kwargs)
            content = response.choices[0].message.content or ""
            output = content.strip()
            self._log_prompt_io(system_prompt, user_prompt, output)
            return output

        if self.backend == "bedrock":
            output = self._generate_bedrock(system_prompt, user_prompt)
            self._log_prompt_io(system_prompt, user_prompt, output)
            return output

        if self.backend == "anthropic":
            # Anthropic API takes `system` as a top-level arg, not a message.
            # Claude Opus 4.7 rejects custom sampling params and manual thinking
            # budgets, so extended thinking is enabled through adaptive thinking.
            request_kwargs = {
                "model": self.model_name,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_prompt}],
                "max_tokens": self.max_new_tokens,
            }
            request_kwargs.update(self._anthropic_thinking_kwargs())
            # Streaming required by the SDK for requests whose estimated
            # runtime exceeds 10 minutes (true with high max_tokens + adaptive
            # thinking). get_final_message() returns the same Message object
            # that non-streaming create() would have returned.
            with self._client.messages.stream(**request_kwargs) as stream:
                response = stream.get_final_message()
            # response.content is a list of content blocks; join text blocks.
            parts = []
            for block in response.content:
                text = getattr(block, "text", None)
                if text:
                    parts.append(text)
            output = "".join(parts).strip()
            self._log_prompt_io(system_prompt, user_prompt, output)
            return output

        if self.backend == "gemini":
            output = self._generate_gemini(system_prompt, user_prompt)
            self._log_prompt_io(system_prompt, user_prompt, output)
            return output

        if self.backend == "vertex":
            output = self._generate_vertex(system_prompt, user_prompt)
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

    def _post_json(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict,
        max_retries: Optional[int] = None,
        retryable_statuses: Optional[set[int]] = None,
    ) -> dict:
        if max_retries is None:
            max_retries = int(os.environ.get("CSD_API_MAX_RETRIES", "5"))
        retry_base_seconds = float(os.environ.get("CSD_API_RETRY_BASE_SECONDS", "20"))
        if retryable_statuses is None:
            retryable_statuses = {408, 409, 429, 500, 502, 503, 504, 529}
        for attempt in range(max_retries + 1):
            request = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json", "Connection": "close", **headers},
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
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                # Network-level failures (read timeout, connection reset) carry no
                # HTTP status, so the branch above never sees them — retry these
                # too, or one transient blip kills the whole synthesis run
                # (observed 2026-06-11: a single Bedrock read timeout ended a
                # run at attempt 18/20).
                if attempt < max_retries:
                    sleep_seconds = retry_base_seconds * (2 ** attempt)
                    print(
                        f"[api-retry] {self.backend} network error ({exc}); "
                        f"retry {attempt + 1}/{max_retries} after {sleep_seconds:.1f}s",
                        flush=True,
                    )
                    time.sleep(sleep_seconds)
                    continue
                raise RuntimeError(
                    f"{self.backend} generation API failed at network level after "
                    f"{max_retries} retries: {exc}"
                ) from exc
        return json.loads(body)

    @staticmethod
    def _dedupe_nonempty(values: list[Optional[str]]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for value in values:
            if not value or value in seen:
                continue
            seen.add(value)
            result.append(value)
        return result

    @staticmethod
    def _is_quota_exhausted_error(exc: BaseException) -> bool:
        status = getattr(exc, "status_code", None)
        text = " ".join(
            str(part)
            for part in (
                exc,
                getattr(exc, "response_body", ""),
                getattr(exc, "body", ""),
            )
            if part
        ).lower()
        return status == 429 or "resource_exhausted" in text or "quota" in text

    def _gemini_api_keys(self, primary: Optional[str]) -> list[str]:
        backups = [
            os.environ.get(f"GEMINI_API_KEY_BACKUP_{idx}")
            for idx in range(1, 10)
        ]
        return self._dedupe_nonempty([
            primary,
            os.environ.get("GEMINI_API_KEY"),
            os.environ.get("GOOGLE_API_KEY"),
            *backups,
        ])

    def _generate_bedrock(self, system_prompt: str, user_prompt: str) -> str:
        client = getattr(self, "_client", None)
        if client is not None and hasattr(client, "converse"):
            data = client.converse(
                modelId=self.model_name,
                system=[{"text": system_prompt}],
                messages=[{"role": "user", "content": [{"text": user_prompt}]}],
                inferenceConfig={"maxTokens": self.max_new_tokens},
            )
            parts = data.get("output", {}).get("message", {}).get("content") or []
            return "".join(part.get("text", "") for part in parts).strip()

        base_url = (self.api_base_url or self._bedrock_runtime_base_url()).rstrip("/")
        model = urllib.parse.quote(self.model_name, safe="")
        url = f"{base_url}/model/{model}/converse"
        payload = {
            "system": [{"text": system_prompt}],
            "messages": [{"role": "user", "content": [{"text": user_prompt}]}],
            "inferenceConfig": {
                "maxTokens": self.max_new_tokens,
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

    def _generate_gemini(self, system_prompt: str, user_prompt: str) -> str:
        base_url = (self.api_base_url or "https://generativelanguage.googleapis.com/v1beta").rstrip("/")
        model = urllib.parse.quote(self.model_name, safe="")
        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {
                "maxOutputTokens": self.max_new_tokens,
                "temperature": self.temperature,
                "topP": self.top_p,
            },
        }
        keys = self._gemini_api_keys(self.api_key)
        if not keys:
            keys = [""]
        last_exc: Optional[BaseException] = None
        for idx, api_key in enumerate(keys):
            key = urllib.parse.quote(api_key, safe="")
            url = f"{base_url}/models/{model}:generateContent?key={key}"
            try:
                if len(keys) == 1:
                    data = self._post_json(url, {}, payload)
                else:
                    data = self._post_json(url, {}, payload, max_retries=0, retryable_statuses=set())
                break
            except Exception as exc:
                last_exc = exc
                if idx + 1 < len(keys) and self._is_quota_exhausted_error(exc):
                    continue
                raise
        else:
            assert last_exc is not None
            raise last_exc
        candidates = data.get("candidates") or []
        if not candidates:
            return ""
        parts = candidates[0].get("content", {}).get("parts") or []
        return "".join(part.get("text", "") for part in parts).strip()

    def _vertex_project(self) -> str:
        project = (
            os.environ.get("VERTEX_AI_PROJECT")
            or os.environ.get("GOOGLE_CLOUD_PROJECT")
            or os.environ.get("GCP_PROJECT")
        )
        if not project:
            raise RuntimeError(
                "VERTEX_AI_PROJECT or GOOGLE_CLOUD_PROJECT is required for Vertex AI generation"
            )
        return project

    def _vertex_location(self) -> str:
        return (
            os.environ.get("VERTEX_AI_LOCATION")
            or os.environ.get("GOOGLE_CLOUD_LOCATION")
            or os.environ.get("GOOGLE_VERTEX_LOCATION")
            or "global"
        )

    def _vertex_access_token(self) -> str:
        token = self.api_key or os.environ.get("VERTEX_AI_ACCESS_TOKEN")
        if token:
            return token
        try:
            import google.auth
            from google.auth.transport.requests import Request
        except ImportError as exc:
            raise RuntimeError(
                "Vertex AI generation requires VERTEX_AI_ACCESS_TOKEN or google-auth "
                "with Application Default Credentials."
            ) from exc
        credentials, _project = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        credentials.refresh(Request())
        token = getattr(credentials, "token", None)
        if not token:
            raise RuntimeError("Application Default Credentials did not provide an access token")
        return token

    def _vertex_auth_headers(self) -> dict[str, str]:
        api_key = (
            os.environ.get("VERTEX_AI_API_KEY")
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
        )
        if api_key:
            return {"x-goog-api-key": api_key}
        return {"Authorization": f"Bearer {self._vertex_access_token()}"}

    def _vertex_generate_content(
        self,
        model_name: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
    ) -> str:
        base_url = (self.api_base_url or self._default_api_base_url("vertex") or "").rstrip("/")
        project = urllib.parse.quote(self._vertex_project(), safe="")
        location = urllib.parse.quote(self._vertex_location(), safe="")
        model = urllib.parse.quote(model_name, safe="")
        url = (
            f"{base_url}/projects/{project}/locations/{location}/"
            f"publishers/google/models/{model}:generateContent"
        )
        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": self.temperature,
                "topP": self.top_p,
            },
        }
        headers = self._vertex_auth_headers()
        primary_key = headers.get("x-goog-api-key")
        if primary_key:
            keys = self._gemini_api_keys(primary_key)
            last_exc: Optional[BaseException] = None
            for idx, api_key in enumerate(keys):
                try:
                    if len(keys) == 1:
                        data = self._post_json(url, {"x-goog-api-key": api_key}, payload)
                    else:
                        data = self._post_json(
                            url,
                            {"x-goog-api-key": api_key},
                            payload,
                            max_retries=0,
                            retryable_statuses=set(),
                        )
                    break
                except Exception as exc:
                    last_exc = exc
                    if idx + 1 < len(keys) and self._is_quota_exhausted_error(exc):
                        continue
                    raise
            else:
                assert last_exc is not None
                raise last_exc
        else:
            data = self._post_json(url, headers, payload)
        candidates = data.get("candidates") or []
        if not candidates:
            return ""
        parts = candidates[0].get("content", {}).get("parts") or []
        return "".join(part.get("text", "") for part in parts).strip()

    def _generate_vertex(self, system_prompt: str, user_prompt: str) -> str:
        return self._vertex_generate_content(
            self.model_name,
            system_prompt,
            user_prompt,
            self.max_new_tokens,
        )

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

    def summarize_rationale_claim(self, rationale: str) -> str:
        """Summarize an attempt rationale into one empirical branch claim.

        This uses a separate small hosted model so synthesis prompts get the
        rationale's causal claim without blunt text truncation. If the summary
        backend is unavailable, return the full rationale rather than silently
        losing context.
        """
        rationale = rationale.strip()
        if not rationale:
            return ""
        backend = os.environ.get("CSD_RATIONALE_SUMMARY_BACKEND", "anthropic").strip().lower()
        if backend in {"", "off", "none", "disabled"}:
            return rationale
        try:
            return self._summarize_rationale_claim_with_backend(rationale, backend, fallback=False)
        except Exception as exc:
            fallback_backend = os.environ.get(
                "CSD_RATIONALE_SUMMARY_FALLBACK_BACKEND",
                "anthropic",
            ).strip().lower()
            if fallback_backend in {"", "off", "none", "disabled"}:
                print(f"[rationale-summary] using full rationale; summary failed: {exc}", flush=True)
                return rationale
            try:
                print(
                    f"[rationale-summary] primary {backend} summary failed; "
                    f"trying {fallback_backend}: {exc}",
                    flush=True,
                )
                return self._summarize_rationale_claim_with_backend(
                    rationale,
                    fallback_backend,
                    fallback=True,
                )
            except Exception as fallback_exc:
                print(
                    "[rationale-summary] using full rationale; "
                    f"primary failed: {exc}; fallback failed: {fallback_exc}",
                    flush=True,
                )
            return rationale

    def _rationale_summary_messages(self, rationale: str) -> tuple[str, str]:
        system_prompt = (
            "Summarize a CSD attempt rationale as a factual empirical branch claim. "
            "Preserve the causal hypothesis and concrete changed knobs. "
            "Do not add advice, evaluation, or facts not present in the rationale. "
            "Return one sentence, at most 35 words."
        )
        return system_prompt, f"Rationale:\n{rationale}"

    def _summarize_rationale_claim_with_backend(
        self,
        rationale: str,
        backend: str,
        *,
        fallback: bool,
    ) -> str:
        if backend == "openai":
            return self._summarize_rationale_claim_openai(rationale, fallback=fallback)
        if backend == "anthropic":
            return self._summarize_rationale_claim_anthropic(rationale, fallback=fallback)
        if backend == "bedrock":
            return self._summarize_rationale_claim_bedrock(rationale, fallback=fallback)
        if backend == "gemini":
            return self._summarize_rationale_claim_gemini(rationale, fallback=fallback)
        if backend == "vertex":
            return self._summarize_rationale_claim_vertex(rationale, fallback=fallback)
        raise ValueError(f"unsupported rationale summary backend: {backend}")

    def _summarize_rationale_claim_bedrock(self, rationale: str, *, fallback: bool = False) -> str:
        token = (
            os.environ.get("CSD_RATIONALE_SUMMARY_FALLBACK_API_KEY")
            if fallback
            else os.environ.get("CSD_RATIONALE_SUMMARY_API_KEY")
        ) or os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
        if not token:
            raise RuntimeError("AWS_BEARER_TOKEN_BEDROCK is not set")
        model = (
            os.environ.get("CSD_RATIONALE_SUMMARY_FALLBACK_MODEL")
            if fallback
            else os.environ.get("CSD_RATIONALE_SUMMARY_MODEL")
        ) or "us.anthropic.claude-haiku-4-5-20251001-v1:0"
        base_url = (
            (
                os.environ.get("CSD_RATIONALE_SUMMARY_FALLBACK_BASE_URL")
                if fallback
                else os.environ.get("CSD_RATIONALE_SUMMARY_BASE_URL")
            )
            or self._bedrock_runtime_base_url()
            or ""
        ).rstrip("/")
        system_prompt, user_prompt = self._rationale_summary_messages(rationale)
        url = f"{base_url}/model/{urllib.parse.quote(model, safe='')}/converse"
        payload = {
            "system": [{"text": system_prompt}],
            "messages": [{"role": "user", "content": [{"text": user_prompt}]}],
            "inferenceConfig": {
                "maxTokens": int(os.environ.get("CSD_RATIONALE_SUMMARY_MAX_TOKENS", "96")),
            },
        }
        data = self._post_json(
            url,
            {"Authorization": f"Bearer {token}", "Accept": "application/json"},
            payload,
        )
        parts = data.get("output", {}).get("message", {}).get("content") or []
        content = "".join(part.get("text", "") for part in parts)
        return self._clean_rationale_summary(content)

    def _summarize_rationale_claim_openai(self, rationale: str, *, fallback: bool = False) -> str:
        if fallback:
            raise ValueError("openai fallback is not configured by default")
        api_key = os.environ.get("CSD_RATIONALE_SUMMARY_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        if self._summary_client is None:
            from openai import OpenAI

            self._summary_client = OpenAI(
                api_key=api_key,
                base_url=os.environ.get("CSD_RATIONALE_SUMMARY_BASE_URL")
                or os.environ.get("OPENAI_BASE_URL"),
            )
        model = (
            os.environ.get("CSD_RATIONALE_SUMMARY_MODEL")
            or os.environ.get("OPENAI_RATIONALE_SUMMARY_MODEL")
            or "chat-latest"
        )
        system_prompt, user_prompt = self._rationale_summary_messages(rationale)
        response = self._summary_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_completion_tokens=int(os.environ.get("CSD_RATIONALE_SUMMARY_MAX_TOKENS", "96")),
        )
        return self._clean_rationale_summary(response.choices[0].message.content or "")

    def _summarize_rationale_claim_anthropic(self, rationale: str, *, fallback: bool = False) -> str:
        api_key = (
            os.environ.get("CSD_RATIONALE_SUMMARY_FALLBACK_API_KEY")
            if fallback
            else os.environ.get("CSD_RATIONALE_SUMMARY_API_KEY")
        ) or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        if self._summary_anthropic_client is None:
            from anthropic import Anthropic

            client_kwargs = {"api_key": api_key}
            base_url = (
                os.environ.get("CSD_RATIONALE_SUMMARY_FALLBACK_BASE_URL")
                if fallback
                else os.environ.get("CSD_RATIONALE_SUMMARY_BASE_URL")
            ) or os.environ.get("ANTHROPIC_BASE_URL")
            if base_url:
                client_kwargs["base_url"] = base_url
            self._summary_anthropic_client = Anthropic(**client_kwargs)
        model = (
            os.environ.get("CSD_RATIONALE_SUMMARY_FALLBACK_MODEL")
            if fallback
            else os.environ.get("CSD_RATIONALE_SUMMARY_MODEL")
        ) or "claude-haiku-4-5"
        system_prompt, user_prompt = self._rationale_summary_messages(rationale)
        response = self._summary_anthropic_client.messages.create(
            model=model,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            max_tokens=int(os.environ.get("CSD_RATIONALE_SUMMARY_MAX_TOKENS", "96")),
        )
        content = "".join(
            getattr(block, "text", "") or ""
            for block in response.content
            if getattr(block, "type", None) == "text"
        )
        return self._clean_rationale_summary(content)

    def _summarize_rationale_claim_gemini(self, rationale: str, *, fallback: bool = False) -> str:
        primary_api_key = (
            os.environ.get("CSD_RATIONALE_SUMMARY_FALLBACK_API_KEY")
            if fallback
            else os.environ.get("CSD_RATIONALE_SUMMARY_API_KEY")
        ) or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        api_keys = self._gemini_api_keys(primary_api_key)
        if not api_keys:
            raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY is not set")
        model = (
            os.environ.get("CSD_RATIONALE_SUMMARY_FALLBACK_MODEL")
            if fallback
            else os.environ.get("CSD_RATIONALE_SUMMARY_MODEL")
        ) or "gemini-2.5-flash-lite"
        base_url = (
            os.environ.get("CSD_RATIONALE_SUMMARY_FALLBACK_BASE_URL")
            if fallback
            else os.environ.get("CSD_RATIONALE_SUMMARY_BASE_URL")
        ) or os.environ.get("GEMINI_BASE_URL") or "https://generativelanguage.googleapis.com/v1beta"
        system_prompt, user_prompt = self._rationale_summary_messages(rationale)
        model_path = urllib.parse.quote(model, safe="")
        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {
                "maxOutputTokens": int(os.environ.get("CSD_RATIONALE_SUMMARY_MAX_TOKENS", "96")),
            },
        }
        last_exc: Optional[BaseException] = None
        for idx, api_key in enumerate(api_keys):
            key = urllib.parse.quote(api_key, safe="")
            try:
                if len(api_keys) == 1:
                    data = self._post_json(
                        f"{base_url.rstrip('/')}/models/{model_path}:generateContent?key={key}",
                        {},
                        payload,
                    )
                else:
                    data = self._post_json(
                        f"{base_url.rstrip('/')}/models/{model_path}:generateContent?key={key}",
                        {},
                        payload,
                        max_retries=0,
                        retryable_statuses=set(),
                    )
                break
            except Exception as exc:
                last_exc = exc
                if idx + 1 < len(api_keys) and self._is_quota_exhausted_error(exc):
                    continue
                raise
        else:
            assert last_exc is not None
            raise last_exc
        candidates = data.get("candidates") or []
        if not candidates:
            raise RuntimeError("empty Gemini rationale summary")
        parts = candidates[0].get("content", {}).get("parts") or []
        content = "".join(part.get("text", "") for part in parts)
        return self._clean_rationale_summary(content)

    def _summarize_rationale_claim_vertex(self, rationale: str, *, fallback: bool = False) -> str:
        model = (
            os.environ.get("CSD_RATIONALE_SUMMARY_FALLBACK_MODEL")
            if fallback
            else os.environ.get("CSD_RATIONALE_SUMMARY_MODEL")
        ) or "gemini-2.5-flash-lite"
        system_prompt, user_prompt = self._rationale_summary_messages(rationale)
        output = self._vertex_generate_content(
            model,
            system_prompt,
            user_prompt,
            int(os.environ.get("CSD_RATIONALE_SUMMARY_MAX_TOKENS", "96")),
        )
        return self._clean_rationale_summary(output)

    @staticmethod
    def _clean_rationale_summary(content: str) -> str:
        summary = " ".join(content.strip().split())
        if not summary:
            raise RuntimeError("empty rationale summary")
        return summary

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

        # Drop any prose preamble above the rationale block markers.
        marker_idx = strategy.find("// CSD_RATIONALE_BEGIN")
        if marker_idx > 0:
            strategy = strategy[marker_idx:]

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
        previous_accuracy: float,
        previous_syntax_rate: float,
        num_examples: int,
        goal_accuracy: float,
        goal_syntax_rate: float,
        evaluation_feedback: str,
        best_strategy: str | None = None,
        best_accuracy: float | None = None,
        best_syntax_rate: float | None = None,
        search_memory: str = "",
        allowed_helpers: list[str] | None = None,
        eval_max_seconds_per_example: float | None = None,
        mode_examples: str = "",
        attempt_outcome_ledger: str = "",
    ) -> str:
        """Generate a refined strategy after evaluation failure.

        The model sees the previous attempt's code and scores, the goal,
        the evaluation feedback, and — only when the previous attempt
        regressed from a better-scoring earlier attempt — the best-so-far
        strategy as a positive anchor. It may also see a compact empirical
        outcome ledger so it has global search context without replaying full
        prior strategy bodies.
        """
        task_description = self._current_task_description or "Unknown task"
        system_prompt, user_prompt = build_evaluation_failure_prompt(
            task_description=task_description,
            previous_strategy=previous_strategy,
            previous_accuracy=previous_accuracy,
            previous_syntax_rate=previous_syntax_rate,
            num_examples=num_examples,
            goal_accuracy=goal_accuracy,
            goal_syntax_rate=goal_syntax_rate,
            evaluation_feedback=evaluation_feedback,
            best_strategy=best_strategy,
            best_accuracy=best_accuracy,
            best_syntax_rate=best_syntax_rate,
            search_memory=search_memory,
            allowed_helpers=allowed_helpers,
            eval_max_seconds_per_example=eval_max_seconds_per_example,
            mode_examples=mode_examples,
            attempt_outcome_ledger=attempt_outcome_ledger,
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
