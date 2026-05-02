"""
Strategy generator for CSD synthesis.

Supports both local HuggingFace models and OpenAI-compatible chat APIs.
"""

import os
import re
from pathlib import Path
from typing import Optional

import torch

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

    Supports either local HuggingFace inference or an OpenAI-compatible API.
    """

    # Default model - can be overridden
    DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
    
    # Path to the template file
    TEMPLATE_PATH = Path(__file__).parent.parent / "dafny" / "GeneratedCSD.dfy"
    
    # Marker in template to replace
    STRATEGY_MARKER = "// QWEN_INSERT_STRATEGY_HERE"

    def __init__(
        self,
        model_name: Optional[str] = None,
        backend: str = "huggingface",
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        api_base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """
        Initialize the strategy generator.
        
        Args:
            model_name: HuggingFace model name (default: Qwen2.5-Coder-7B-Instruct)
            backend: Inference backend ("huggingface" or "openai")
            device: Device to run on ('cuda', 'mps', 'cpu', or None for auto)
            torch_dtype: Torch dtype for model (default: auto based on device)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p (nucleus) sampling parameter
            load_in_4bit: Load model in 4-bit quantization
            load_in_8bit: Load model in 8-bit quantization
            api_base_url: Optional base URL for an OpenAI-compatible API
            api_key: Optional API key (falls back to environment)
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        self.backend = backend
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit
        self.api_base_url = api_base_url or os.environ.get("OPENAI_BASE_URL")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        
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

        # Load template
        self._template = self._load_template()
    
    def _load_template(self) -> str:
        """Load the GeneratedCSD.dfy template."""
        if not self.TEMPLATE_PATH.exists():
            raise FileNotFoundError(
                f"Template not found at {self.TEMPLATE_PATH}. "
                "Make sure GeneratedCSD.dfy exists in the dafny/ directory."
            )
        return self.TEMPLATE_PATH.read_text()
    
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

        # Format as chat messages
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
            return content.strip()

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

        return response.strip()
    
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

    def _ensure_rationale_block(self, strategy_body: str, *, max_repairs: int = 2) -> str:
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
            system_prompt, user_prompt = build_format_repair_prompt(current)
            repaired_raw = self._generate_text(system_prompt, user_prompt)
            repaired = self._extract_strategy(repaired_raw)
            extracted = extract_rationale(repaired)
            if extracted.rationale is not None and extracted.has_markers:
                return repaired
            current = repaired

        return "// CSD_RATIONALE_BEGIN\n// (Auto-injected rationale)\n// CSD_RATIONALE_END\n" + current
    
    def generate_initial(self, task_description: str) -> str:
        """
        Generate an initial strategy for the given task.

        Args:
            task_description: Description of what the strategy should accomplish

        Returns:
            Strategy expression (Dafny code)
        """
        system_prompt, user_prompt = build_initial_prompt(task_description)
        raw_output = self._generate_text(system_prompt, user_prompt)
        strategy = self._extract_strategy(raw_output)
        return self._ensure_rationale_block(strategy)
    
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
            New strategy expression
        """
        system_prompt, user_prompt = build_verification_error_prompt(
            previous_strategy, error_message
        )
        raw_output = self._generate_text(system_prompt, user_prompt)
        strategy = self._extract_strategy(raw_output)
        return self._ensure_rationale_block(strategy)
    
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
            New strategy expression
        """
        system_prompt, user_prompt = build_runtime_error_prompt(
            previous_strategy, error_traceback
        )
        raw_output = self._generate_text(system_prompt, user_prompt)
        strategy = self._extract_strategy(raw_output)
        return self._ensure_rationale_block(strategy)
    
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
            New strategy expression
        """
        system_prompt, user_prompt = build_compilation_error_prompt(
            previous_strategy, error_message
        )
        raw_output = self._generate_text(system_prompt, user_prompt)
        strategy = self._extract_strategy(raw_output)
        return self._ensure_rationale_block(strategy)

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
            New strategy expression
        """
        system_prompt, user_prompt = build_evaluation_failure_prompt(
            previous_strategy, evaluation_feedback
        )
        raw_output = self._generate_text(system_prompt, user_prompt)
        strategy = self._extract_strategy(raw_output)
        return self._ensure_rationale_block(strategy)

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
