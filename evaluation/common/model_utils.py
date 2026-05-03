"""
Model loading and management utilities for CSD evaluation.

Provides optimized model loading for CUDA/CPU with proper device handling.
"""

from __future__ import annotations

import math
import os
import random
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set precision before any torch operations to avoid TensorFloat32 warning
torch.set_float32_matmul_precision('high')

# GSM delimiter strings to add explicitly to the vocabulary.
# The Qwen tokenizer produces ' <<' (id 1115, with leading space) for << in context,
# and '>>' (id 2452) for >>. We ensure both ' <<' and ' >>' are in vocab, plus '>>'
# as a fallback in case ' >>' is not a single BPE token.
GSM_DELIMITER_STRINGS = [" <<", " >>", ">>"]

# SQL scaffolding tokens that should not be penalized even when they are not
# explicitly present in the prompt text. These keep the query structure viable.
SQL_SCAFFOLD_KEYWORDS = frozenset({
    "select", "from", "where", "join", "on", "and", "or",
    "group", "by", "having", "order", "limit", "as", "distinct",
    "max", "min", "count", "sum", "avg", "in", "not", "like", "between",
    "inner", "left", "right", "full", "outer", "union", "all", "is", "null",
})


def _hf_offline_enabled() -> bool:
    """True when HuggingFace model/tokenizer loaders should stay offline."""
    return any(os.environ.get(name, "").strip() in {"1", "true", "True"} for name in (
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
    ))


def _is_hf_connection_error(exc: Exception) -> bool:
    """Best-effort detection for DNS/network lookup failures in HF loading."""
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


def _load_tokenizer(model_name: str, **kwargs):
    """Load a tokenizer, falling back to local-files-only on network failure."""
    load_kwargs = dict(kwargs)
    if _hf_offline_enabled():
        load_kwargs["local_files_only"] = True
    try:
        return AutoTokenizer.from_pretrained(model_name, **load_kwargs)
    except Exception as exc:
        if load_kwargs.get("local_files_only") or not _is_hf_connection_error(exc):
            raise
        print("  HuggingFace network lookup failed; retrying tokenizer load from local cache only.")
        load_kwargs["local_files_only"] = True
        return AutoTokenizer.from_pretrained(model_name, **load_kwargs)


def _load_causal_lm(**kwargs):
    """Load a causal LM, falling back to local-files-only on network failure."""
    load_kwargs = dict(kwargs)
    if _hf_offline_enabled():
        load_kwargs["local_files_only"] = True
    try:
        return AutoModelForCausalLM.from_pretrained(**load_kwargs)
    except Exception as exc:
        if load_kwargs.get("local_files_only") or not _is_hf_connection_error(exc):
            raise
        print("  HuggingFace network lookup failed; retrying model load from local cache only.")
        load_kwargs["local_files_only"] = True
        return AutoModelForCausalLM.from_pretrained(**load_kwargs)


def _decode_single_token(tokenizer, token_id: int) -> str:
    """Decode one token ID without cleanup so token strings stay stable."""
    return tokenizer.decode([token_id], clean_up_tokenization_spaces=False)


def _dedupe_token_ids_by_decoded_string(tokenizer, token_ids: list[int]) -> tuple[list[int], int]:
    """
    Keep only the first HF token ID for each decoded token string.

    The verified LM invariant requires logical tokens to be unique strings, but
    HuggingFace vocabularies often contain multiple IDs that decode to the same
    surface form. Collapsing duplicates here preserves a stable logical vocab
    while still letting us use the original HF IDs for logits lookup.
    """
    unique_ids: list[int] = []
    seen_tokens: set[str] = set()
    dropped = 0
    for token_id in token_ids:
        token = _decode_single_token(tokenizer, token_id)
        if token in seen_tokens:
            dropped += 1
            continue
        seen_tokens.add(token)
        unique_ids.append(token_id)
    return unique_ids, dropped


def _valid_tokens_ids_logits_py(tokens: list[str], ids: list[int], logits: list[float]) -> bool:
    """Python-native equivalent of LM.ValidTokensIdsLogits with linear-time uniqueness."""
    return (
        len(tokens) == len(ids) == len(logits)
        and len(ids) > 0
        and ids[0] == 0
        and all(i == ids[i] for i in range(len(ids)))
        and len(set(tokens)) == len(tokens)
        and all(-1e9 <= logit <= 1e9 for logit in logits)
    )


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "")
    if not raw:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _collect_prompt_sql_bias_tokens(prompt_text: str) -> set[str]:
    """Collect prompt-grounded SQL token strings (schema/question identifiers and literals)."""
    text = _extract_spider_grounding_text(prompt_text).strip()
    if not text:
        return set()
    items: set[str] = set()
    for marker in ('"', "'", ' "', " '"):
        items.add(marker)
    for m in re.finditer(r"[A-Za-z_][A-Za-z0-9_]*", text):
        token = m.group(0)
        lower = token.lower()
        items.add(token)
        items.add(lower)
        items.add(" " + token)
        items.add(" " + lower)
        items.add(f'"{token}"')
        items.add(f'"{lower}"')
        items.add(f' "{token}"')
        items.add(f' "{lower}"')
    for m in re.finditer(r'"([^"]+)"', text):
        phrase = m.group(1).strip()
        if phrase:
            items.add(phrase)
            items.add(" " + phrase)
            items.add(f'"{phrase}"')
            items.add(f' "{phrase}"')
    return items


def _extract_spider_grounding_text(prompt_text: str) -> str:
    """
    Extract Spider grounding text (actual question/schema) and avoid worked examples.

    The Spider prompt template includes demonstration patterns with fixed literals
    (e.g., specific states). Prompt-token boosting should not over-weight those
    demonstration literals.
    """
    text = prompt_text or ""
    if "You are a text-to-SQL system for the Spider benchmark." not in text:
        return text

    lines = text.splitlines()
    headings = {
        "Database:",
        "Question:",
        "Database schema:",
        "Columns by table (copy names exactly):",
    }
    stop_headings = {
        "Example final format:",
        "SQL:",
        "Example pattern",
    }

    sections: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if any(line.startswith(stop) for stop in stop_headings):
            break
        if any(line.startswith(h) for h in headings):
            section_lines = [line]
            i += 1
            while i < len(lines):
                nxt = lines[i].strip()
                if not nxt:
                    break
                if any(nxt.startswith(h) for h in headings) or any(nxt.startswith(stop) for stop in stop_headings):
                    break
                section_lines.append(nxt)
                i += 1
            sections.append("\n".join(section_lines))
            continue
        i += 1

    if sections:
        # Use the last seen grounding block to prioritize the current example.
        return "\n".join(sections[-4:])
    return text


def _normalize_prompt_bias_token(token: str) -> str:
    """Normalize token text for fuzzy prompt-grounding checks."""
    t = (token or "").strip().lower()
    if len(t) >= 2 and ((t[0] == '"' and t[-1] == '"') or (t[0] == "'" and t[-1] == "'")):
        t = t[1:-1].strip()
    return t


def _token_is_sql_scaffold(token: str) -> bool:
    """True for SQL/operator tokens we should keep unpenalized."""
    t = (token or "").strip()
    if not t:
        return True
    low = t.lower()
    if low in SQL_SCAFFOLD_KEYWORDS:
        return True
    if re.fullmatch(r"[(),.*=<>!+\-/%]+", t):
        return True
    if re.fullmatch(r"\d+(?:\.\d+)?", t):
        return True
    if re.fullmatch(r"t\d+", low):
        return True
    if low in {'"', "'", "`"}:
        return True
    return False


def _sample_from_masked_logits(
    logits: list[float],
    rng: random.Random,
    temperature: float,
    top_p: float,
) -> int:
    """Sample one index from masked logits (masked entries should be <= -1e8)."""
    candidates: list[tuple[int, float]] = [(i, v) for i, v in enumerate(logits) if math.isfinite(v) and v > -1e8]
    if not candidates:
        return max(range(len(logits)), key=lambda i: logits[i])
    if len(candidates) == 1:
        return candidates[0][0]

    temp = max(1e-4, temperature)
    max_logit = max(v for _, v in candidates)
    weighted: list[tuple[int, float]] = []
    total = 0.0
    for idx, value in candidates:
        w = math.exp((value - max_logit) / temp)
        if not math.isfinite(w) or w <= 0.0:
            w = 0.0
        weighted.append((idx, w))
        total += w
    if total <= 0.0:
        return max((idx for idx, _ in candidates), key=lambda i: logits[i])

    weighted.sort(key=lambda it: it[1], reverse=True)
    keep: list[tuple[int, float]] = []
    running = 0.0
    p_cut = min(max(top_p, 0.05), 1.0)
    for idx, w in weighted:
        keep.append((idx, w))
        running += w
        if running / total >= p_cut and len(keep) >= 1:
            break

    keep_total = sum(w for _, w in keep)
    if keep_total <= 0.0:
        return keep[0][0]
    r = rng.random() * keep_total
    acc = 0.0
    for idx, w in keep:
        acc += w
        if acc >= r:
            return idx
    return keep[-1][0]


def get_model_input_device(model) -> torch.device:
    """
    Find the device where the model's embedding layer resides.
    
    For models with hf_device_map (multi-GPU via accelerate), this determines
    where input tensors should be placed.
    
    Args:
        model: A HuggingFace model instance
        
    Returns:
        The torch.device for input tensors
    """
    # For models with hf_device_map (multi-GPU via accelerate)
    if hasattr(model, 'hf_device_map') and model.hf_device_map:
        # Look for embedding layer in device map
        for key, device in model.hf_device_map.items():
            if 'embed' in key.lower():
                return torch.device(f"cuda:{device}" if isinstance(device, int) else device)
        # Fallback to first device in map
        first_device = next(iter(model.hf_device_map.values()))
        return torch.device(f"cuda:{first_device}" if isinstance(first_device, int) else first_device)
    
    # For single-device models
    return next(model.parameters()).device


def get_max_input_length(model, tokenizer) -> int:
    """
    Choose a safe max input length for the model/tokenizer.
    
    Args:
        model: A HuggingFace model instance
        tokenizer: A HuggingFace tokenizer instance
        
    Returns:
        Maximum input length in tokens
    """
    max_len = None
    if hasattr(model, "config") and getattr(model.config, "max_position_embeddings", None):
        max_len = int(model.config.max_position_embeddings)
    tok_max = getattr(tokenizer, "model_max_length", None)
    if tok_max and tok_max < 1_000_000:
        max_len = min(max_len, int(tok_max)) if max_len else int(tok_max)
    return max_len or 4096


def _resolve_token_ids(tokenizer, vocab_size: int | None, token_ids=None) -> list[int]:
    """Resolve the constrained vocabulary, defaulting to the full tokenizer."""
    if token_ids is not None:
        return list(token_ids)

    full_vocab_size = len(tokenizer)
    if vocab_size is None or vocab_size <= 0:
        return list(range(full_vocab_size))
    return list(range(min(vocab_size, full_vocab_size)))


def _device_map_for_selected_device(device: str):
    """Return a HuggingFace device_map that honors the selected CUDA device."""
    if os.environ.get("CSD_EVAL_DEVICE_MAP_AUTO", "").strip().lower() in {"1", "true", "yes", "on"}:
        return "auto"
    if device == "cuda":
        return {"": "cuda:0"}
    return {"": device}


def create_huggingface_lm(
    model_name: str,
    device: str,
    vocab_size: int | None,
    VerifiedDecoderAgent,
    _dafny,
    token_ids=None,
    extra_token_strings: list[str] | None = None,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    add_gsm_delimiter_tokens: bool = False,
):
    """
    Create a HuggingFace LM wrapped with a Dafny-compatible interface.

    Args:
        model_name: HuggingFace model identifier
        device: Device to use ("cuda", "cpu", etc.)
        vocab_size: Size of constrained vocabulary. `None` or <= 0 uses the full tokenizer.
        VerifiedDecoderAgent: Imported Dafny module for LM interface
        _dafny: Dafny runtime module
        token_ids: Optional list of token IDs for constrained vocabulary
        load_in_4bit: Whether to load in 4-bit quantization
        load_in_8bit: Whether to load in 8-bit quantization

    Returns:
        A Dafny-compatible LM wrapper
    """
    prec_str = "FP16"
    if load_in_4bit: prec_str = "4-bit"
    elif load_in_8bit: prec_str = "8-bit"
    
    print(f"Loading model: {model_name} on {device}... ({prec_str})")
    tokenizer = _load_tokenizer(model_name, trust_remote_code=True)

    if device.startswith("cuda"):
        kwargs = {
            "pretrained_model_name_or_path": model_name,
            "trust_remote_code": True,
            "device_map": _device_map_for_selected_device(device),
        }
        
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        elif load_in_8bit:
            kwargs["load_in_8bit"] = True
        else:
            kwargs["torch_dtype"] = torch.float16
            
        model = _load_causal_lm(**kwargs)
        input_device = get_model_input_device(model)
        print(f"Model loaded on {input_device}")
    else:
        # CPU fallback
        model = _load_causal_lm(
            pretrained_model_name_or_path=model_name,
            trust_remote_code=True,
            torch_dtype=torch.float32,
        )
        input_device = torch.device("cpu")

    model.eval()

    token_ids = _resolve_token_ids(tokenizer, vocab_size, token_ids=token_ids)
    if token_ids:
        if add_gsm_delimiter_tokens:
            # Add GSM delimiter tokens: ' <<' (id≈1115) is likely already in default 2000-token vocab,
            # but ' >>' and '>>' (id≈2452) are typically outside. Add whichever encode as single tokens.
            existing_ids = set(token_ids)
            for s in GSM_DELIMITER_STRINGS:
                ids = tokenizer.encode(s, add_special_tokens=False)
                if len(ids) == 1:
                    tid = ids[0]
                    if tid not in existing_ids:
                        token_ids.append(tid)
                        existing_ids.add(tid)
                        print(f"  Added GSM delimiter token {repr(s)} (id={tid}) to vocabulary.")
        if extra_token_strings:
            existing_ids = set(token_ids)
            for s in extra_token_strings:
                if not s:
                    continue
                for tid in tokenizer.encode(s, add_special_tokens=False):
                    if tid not in existing_ids:
                        token_ids.append(tid)
                        existing_ids.add(tid)

    token_ids, dropped_duplicates = _dedupe_token_ids_by_decoded_string(tokenizer, token_ids)
    if dropped_duplicates:
        print(f"  Deduplicated {dropped_duplicates} decoded token string(s) from vocabulary.")

    tokens_dafny = _dafny.SeqWithoutIsStrInference(
        [_dafny.Seq(_decode_single_token(tokenizer, tid)) for tid in token_ids]
    )

    class HuggingFaceLM(VerifiedDecoderAgent.LM):
        """Wrapper that bridges HuggingFace models to the Dafny LM interface."""
        
        # Token IDs that must not be chosen as the first output token.
        _FORBID_FIRST_STRINGS = frozenset({"<<", "<", " <<", " << ", " >>", ">>", "$"})

        def __init__(self, hf_model, hf_tokenizer, tokens, tids, dev):
            super().__init__()
            self.model = hf_model
            self.tokenizer = hf_tokenizer
            self._Tokens = tokens
            self._token_ids = tids
            self._input_device = dev
            self._max_input_len = get_max_input_length(hf_model, hf_tokenizer)
            self.instruction_text = ""
            self.Logits = _dafny.Array(None, len(tids))
            for i in range(len(tids)):
                self.Logits[i] = _dafny.BigRational(0)
            # Store full logits for unconstrained generation
            self._full_logits = None
            self._sample_constrained = _env_flag("CSD_CONSTRAINED_SAMPLING", False)
            self._sample_temperature = _env_float("CSD_CONSTRAINED_TEMPERATURE", 0.35)
            self._sample_top_p = _env_float("CSD_CONSTRAINED_TOP_P", 0.9)
            self._prompt_bias_cache_key = ""
            self._prompt_bias_tokens = set()
            self._prompt_bias_normalized = set()
            self._prompt_bias_text_lower = ""
            self._spider_prompt_token_bonus = _env_float("CSD_SPIDER_PROMPT_TOKEN_BONUS", 0.0)
            self._spider_prompt_token_bonus_enabled = self._spider_prompt_token_bonus > 0.0
            self._spider_prompt_token_penalty = _env_float("CSD_SPIDER_PROMPT_TOKEN_PENALTY", 0.0)
            self._spider_prompt_token_penalty_enabled = self._spider_prompt_token_penalty > 0.0
            seed = os.environ.get("CSD_CONSTRAINED_SAMPLING_SEED", "").strip()
            self._sampling_rng = random.Random(int(seed)) if seed else random.Random()
            # Cache token IDs forbidden as first output token.
            vocab_size = hf_tokenizer.vocab_size if hasattr(hf_tokenizer, "vocab_size") else len(hf_tokenizer)
            self._forbid_first_token_ids = set()
            allow_leading_delimiter = os.environ.get("CSD_ALLOW_LEADING_DELIMITER", "").strip().lower() in {
                "1", "true", "yes", "on"
            }
            for vid in range(min(vocab_size, 200000)):  # cap for speed
                try:
                    s = _decode_single_token(hf_tokenizer, vid)
                    if s in HuggingFaceLM._FORBID_FIRST_STRINGS and not allow_leading_delimiter:
                        self._forbid_first_token_ids.add(vid)
                    elif not s:  # forbid tokens that decode to empty string
                        self._forbid_first_token_ids.add(vid)
                except Exception:
                    pass
            # Forbid EOS/pad as first token so the model cannot immediately stop (which produced empty output)
            eos_id = getattr(hf_tokenizer, "eos_token_id", None)
            if eos_id is not None:
                self._forbid_first_token_ids.add(int(eos_id))
            pad_id = getattr(hf_tokenizer, "pad_token_id", None)
            if pad_id is not None and pad_id != eos_id:
                self._forbid_first_token_ids.add(int(pad_id))

        def _to_str(self, obj):
            """Convert a Dafny object (potentially a Seq of chars) to a Python string."""
            if isinstance(obj, str):
                return obj
            try:
                # Dafny Seqs of chars can be converted by joining their elements
                return "".join(obj[i] for i in range(len(obj)))
            except:
                return str(obj)

        def ResetForNewExample(self) -> None:
            """Clear per-example transient LM state between evaluations."""
            self.instruction_text = ""
            self._full_logits = None
            self._first_token_choice = False
            self._prompt_bias_cache_key = ""
            self._prompt_bias_tokens = set()
            self._prompt_bias_normalized = set()
            self._prompt_bias_text_lower = ""
            zero = _dafny.BigRational(0)
            for i in range(self.Logits.length(0)):
                self.Logits[i] = zero

        def _refresh_prompt_bias_tokens(self):
            if not (self._spider_prompt_token_bonus_enabled or self._spider_prompt_token_penalty_enabled):
                self._prompt_bias_tokens = set()
                self._prompt_bias_normalized = set()
                self._prompt_bias_text_lower = ""
                self._prompt_bias_cache_key = self.instruction_text
                return
            if self._prompt_bias_cache_key == self.instruction_text:
                return
            self._prompt_bias_cache_key = self.instruction_text
            self._prompt_bias_tokens = _collect_prompt_sql_bias_tokens(self.instruction_text)
            self._prompt_bias_normalized = {
                _normalize_prompt_bias_token(tok) for tok in self._prompt_bias_tokens
                if _normalize_prompt_bias_token(tok)
            }
            self._prompt_bias_text_lower = (self.instruction_text or "").lower()

        def GenerateLogits(self, input_prefix):
            """Compute logits for the next token given a prefix."""
            import os
            debug = os.environ.get('CSD_MASK_DEBUG', '').lower() in ('1', 'true', 'yes')

            # Correctly handle Dafny sequences which might contain char sequences
            prefix_parts = []
            for i in range(len(input_prefix)):
                prefix_parts.append(self._to_str(input_prefix[i]))
            prefix_text = "".join(prefix_parts)
            
            full_prompt = self.instruction_text + prefix_text
            
            if debug and len(input_prefix) <= 5:
                print(f"    [GENERATE DEBUG] Step {len(input_prefix)} prompt tail:\n...{full_prompt[-200:]}")
                if len(input_prefix) == 0:
                    print(f"    [GENERATE DEBUG] Full initial prompt length: {len(full_prompt)}")

            inputs = self.tokenizer(
                full_prompt,
                return_tensors="pt",
                add_special_tokens=False,
            )
            if inputs["input_ids"].shape[-1] > self._max_input_len:
                inputs["input_ids"] = inputs["input_ids"][:, -self._max_input_len:]
                if "attention_mask" in inputs:
                    inputs["attention_mask"] = inputs["attention_mask"][:, -self._max_input_len:]
            inputs = inputs.to(self._input_device)

            with torch.no_grad():
                # Get logits and immediately move to CPU to avoid cross-device issues
                output = self.model(**inputs)
                logits = output.logits[0, -1, :].float().cpu()
            
            # Forbid delimiter + EOS as first output token so model outputs plain text before the formula
            self._first_token_choice = len(input_prefix) == 0
            if self._first_token_choice and getattr(self, "_forbid_first_token_ids", None):
                for vid in self._forbid_first_token_ids:
                    if vid < logits.shape[0]:
                        logits[vid] = float("-inf")

            # Store full logits for unconstrained generation
            self._full_logits = logits

            # BigRational cannot represent ±inf; clamp so forbidden tokens stay worst
            _LOGIT_FORBIDDEN = -1e9
            for i, tid in enumerate(self._token_ids):
                v = float(logits[tid].item())
                if not math.isfinite(v):
                    v = _LOGIT_FORBIDDEN
                self.Logits[i] = _dafny.BigRational(v)

        def ChooseNextToken(self):
            """Return the token with the highest logit score (constrained to vocab)."""
            import os
            debug = os.environ.get('CSD_MASK_DEBUG', '').lower() in ('1', 'true', 'yes')
            logits_debug = os.environ.get("CSD_LOGITS_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}
            logits_top_k = int(os.environ.get("CSD_LOGITS_TOP_K", "8") or "8")
            if self.Logits.length(0) <= 0:
                raise RuntimeError("No logits are available for constrained selection")
            has_unmasked = False
            for i in range(self.Logits.length(0)):
                if float(self.Logits[i]) > -1e8:
                    has_unmasked = True
                    break
            if not has_unmasked:
                raise RuntimeError("No unmasked token is available after constrained masking")
            self._refresh_prompt_bias_tokens()

            def _is_prompt_grounded(token: str) -> bool:
                if not self._prompt_bias_tokens:
                    return False
                if token in self._prompt_bias_tokens:
                    return True
                norm = _normalize_prompt_bias_token(token)
                if not norm:
                    return False
                if norm in self._prompt_bias_normalized:
                    return True
                # Back off to substring grounding for partial BPE pieces.
                return len(norm) >= 3 and norm in self._prompt_bias_text_lower

            def _score(i: int) -> float:
                base = float(self.Logits[i])
                if not (math.isfinite(base) and base > -1e8):
                    return base
                token = self._to_str(self._Tokens[i])
                grounded = _is_prompt_grounded(token)
                if (
                    self._spider_prompt_token_bonus_enabled
                    and grounded
                ):
                    base += self._spider_prompt_token_bonus
                if (
                    self._spider_prompt_token_penalty_enabled
                    and not grounded
                    and not _token_is_sql_scaffold(token)
                ):
                    base -= self._spider_prompt_token_penalty
                return base

            if self._sample_constrained:
                logits = [_score(i) for i in range(self.Logits.length(0))]
                best_idx = _sample_from_masked_logits(
                    logits=logits,
                    rng=self._sampling_rng,
                    temperature=self._sample_temperature,
                    top_p=self._sample_top_p,
                )
                best_val = _score(best_idx)
            else:
                best_idx, best_val = 0, _score(0)
                for i in range(1, self.Logits.length(0)):
                    val = _score(i)
                    if val > best_val:
                        best_val, best_idx = val, i

            chosen_token = self._Tokens[best_idx]
            if logits_debug:
                scored: list[tuple[int, float]] = []
                for i in range(self.Logits.length(0)):
                    v = float(self.Logits[i])
                    if math.isfinite(v) and v > -1e8:
                        scored.append((i, v))
                scored.sort(key=lambda x: x[1], reverse=True)
                top = scored[:max(1, logits_top_k)]
                debug_parts: list[str] = []
                for i, v in top:
                    tok = self._to_str(self._Tokens[i]).replace("\n", "\\n")
                    debug_parts.append(f"{repr(tok)}:{v:.2f}")
                print(f"    [LOGITS DEBUG] top={'; '.join(debug_parts)}")
            if debug:
                # Convert Dafny Seq to string for display
                try:
                    token_str = ''.join(chosen_token[i] for i in range(len(chosen_token)))
                except:
                    token_str = str(chosen_token)
                print(f"    [CHOOSE DEBUG] Best idx={best_idx}, logit={best_val:.2f}, token={repr(token_str)}")

            return chosen_token
        
        def ChooseNextTokenUnconstrained(self):
            """Return the token with the highest logit score from FULL vocabulary."""
            import os
            debug = os.environ.get('CSD_MASK_DEBUG', '').lower() in ('1', 'true', 'yes')

            if self._full_logits is None:
                raise RuntimeError("Must call GenerateLogits before ChooseNextTokenUnconstrained")
            logits = self._full_logits.clone()
            best_idx = int(logits.argmax().item())
            token_text = _decode_single_token(self.tokenizer, best_idx)
            # If this was the first token and we got empty/EOS (mask failed or tokenizer quirk), take next-best
            if getattr(self, "_first_token_choice", False):
                self._first_token_choice = False
                eos_str = (
                    _decode_single_token(self.tokenizer, self.tokenizer.eos_token_id)
                    if getattr(self.tokenizer, "eos_token_id", None) is not None
                    else ""
                )
                for _ in range(50):
                    if token_text and token_text.strip() and token_text != eos_str:
                        break
                    logits[best_idx] = float("-inf")
                    best_idx = int(logits.argmax().item())
                    token_text = _decode_single_token(self.tokenizer, best_idx)
            
            if debug:
                print(f"    [UNCONSTRAINED DEBUG] chosen_token={repr(token_text)}")

            return _dafny.Seq(token_text)
        
        def MaskTokensExcept(self, valid_tokens, debug=False):
            """Mask all tokens except those in valid_tokens.
            
            This implementation follows the Dafny specification strictly.
            When this is the first token (_first_token_choice), we also exclude EOS and
            empty-decoding tokens so strategies that use ConstrainedStep first still get real text.
            """
            import os
            debug = debug or os.environ.get('CSD_MASK_DEBUG', '').lower() in ('1', 'true', 'yes')

            # Helper to convert Dafny Seq to string
            def seq_to_str(seq):
                try:
                    return ''.join(seq)
                except TypeError:
                    try:
                        return ''.join(seq[i] for i in range(len(seq)))
                    except:
                        return str(seq)

            # Get the set of valid token strings
            valid_set = set()
            for i in range(len(valid_tokens)):
                valid_set.add(seq_to_str(valid_tokens[i]))

            # Spider/sql mode: avoid pure-whitespace constrained loops when meaningful tokens exist.
            if os.environ.get("CSD_DISALLOW_CONSTRAINED_WHITESPACE", "").strip().lower() in {"1", "true", "yes", "on"}:
                non_ws = {tok for tok in valid_set if tok.strip() != ""}
                if non_ws:
                    valid_set = non_ws

            # First token: exclude EOS and delimiter so we don't produce blank output (some strategies use ConstrainedStep first)
            if getattr(self, "_first_token_choice", False):
                self._first_token_choice = False
                eos_str = self.tokenizer.decode([self.tokenizer.eos_token_id]) if getattr(self.tokenizer, "eos_token_id", None) is not None else ""
                allow_leading_delimiter = os.environ.get("CSD_ALLOW_LEADING_DELIMITER", "").strip().lower() in {
                    "1", "true", "yes", "on"
                }
                forbid = {"", eos_str}
                if not allow_leading_delimiter:
                    forbid |= set(HuggingFaceLM._FORBID_FIRST_STRINGS)
                reduced = valid_set - forbid
                if reduced:
                    valid_set = reduced

            # Mask everything not in the valid set
            masked_val = _dafny.BigRational(-1000000000, 1) # -1e9
            
            masked_count = 0
            for i in range(self.Logits.length(0)):
                token_str = seq_to_str(self._Tokens[i])
                if token_str not in valid_set:
                    self.Logits[i] = masked_val
                    masked_count += 1
            
            if debug:
                print(f"    [MASK DEBUG] Masked {masked_count} tokens, {len(valid_set)} remain valid.")

    return HuggingFaceLM(model, tokenizer, tokens_dafny, token_ids, input_device)


def create_huggingface_lm_native(
    model_name: str,
    device: str,
    vocab_size: int | None,
    VerifiedAgentSynthesis,
    token_ids=None,
    extra_token_strings: list[str] | None = None,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    add_gsm_delimiter_tokens: bool = False,
):
    """
    Create a HuggingFace LM wrapped with a Python-native (non-Dafny) interface.

    Uses plain Python lists for Tokens/Ids/Logits so the strategy code runs
    directly without the Dafny runtime.
    """
    prec_str = "FP16"
    if load_in_4bit:
        prec_str = "4-bit"
    elif load_in_8bit:
        prec_str = "8-bit"

    print(f"Loading model (native): {model_name} on {device}... ({prec_str})")
    tokenizer = _load_tokenizer(model_name, trust_remote_code=True)

    if device.startswith("cuda"):
        kwargs = {
            "pretrained_model_name_or_path": model_name,
            "trust_remote_code": True,
            "device_map": _device_map_for_selected_device(device),
        }
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        elif load_in_8bit:
            kwargs["load_in_8bit"] = True
        else:
            kwargs["torch_dtype"] = torch.float16
        model = _load_causal_lm(**kwargs)
        input_device = get_model_input_device(model)
    else:
        model = _load_causal_lm(
            pretrained_model_name_or_path=model_name,
            trust_remote_code=True,
            torch_dtype=torch.float32,
        )
        input_device = torch.device("cpu")

    model.eval()

    token_ids = _resolve_token_ids(tokenizer, vocab_size, token_ids=token_ids)
    if token_ids:
        if add_gsm_delimiter_tokens:
            existing_ids = set(token_ids)
            for s in GSM_DELIMITER_STRINGS:
                ids = tokenizer.encode(s, add_special_tokens=False)
                if len(ids) == 1:
                    tid = ids[0]
                    if tid not in existing_ids:
                        token_ids.append(tid)
                        existing_ids.add(tid)
                        print(f"  Added GSM delimiter token {repr(s)} (id={tid}) to vocabulary.")
        if extra_token_strings:
            existing_ids = set(token_ids)
            for s in extra_token_strings:
                if not s:
                    continue
                for tid in tokenizer.encode(s, add_special_tokens=False):
                    if tid not in existing_ids:
                        token_ids.append(tid)
                        existing_ids.add(tid)

    token_ids, dropped_duplicates = _dedupe_token_ids_by_decoded_string(tokenizer, token_ids)
    if dropped_duplicates:
        print(f"  Deduplicated {dropped_duplicates} decoded token string(s) from vocabulary.")

    tokens = [_decode_single_token(tokenizer, tid) for tid in token_ids]
    max_input_len = get_max_input_length(model, tokenizer)

    class HuggingFaceLMNative(VerifiedAgentSynthesis.LM):
        """HuggingFace model wrapped with Python-native (non-Dafny) LM interface."""

        _FORBID_FIRST_STRINGS = frozenset({"<<", "<", " <<", " << ", " >>", ">>", "$"})

        def __init__(self):
            # Bypass parent __init__ (sets 2-token dummy vocab); set full vocab directly
            self.Tokens = tokens
            self.Ids = list(range(len(tokens)))
            self.Logits = [0.0] * len(tokens)
            self._Tokens = self.Tokens  # alias for evaluator compat
            self._token_ids = token_ids
            self._token_index = {token: i for i, token in enumerate(tokens)}
            self._input_device = input_device
            self._max_input_len = max_input_len
            self.tokenizer = tokenizer
            self.instruction_text = ""
            self._full_logits = None
            self._first_token_choice = False
            self._sample_constrained = _env_flag("CSD_CONSTRAINED_SAMPLING", False)
            self._sample_temperature = _env_float("CSD_CONSTRAINED_TEMPERATURE", 0.35)
            self._sample_top_p = _env_float("CSD_CONSTRAINED_TOP_P", 0.9)
            self._prompt_bias_cache_key = ""
            self._prompt_bias_tokens = set()
            self._prompt_bias_normalized = set()
            self._prompt_bias_text_lower = ""
            self._spider_prompt_token_bonus = _env_float("CSD_SPIDER_PROMPT_TOKEN_BONUS", 0.0)
            self._spider_prompt_token_bonus_enabled = self._spider_prompt_token_bonus > 0.0
            self._spider_prompt_token_penalty = _env_float("CSD_SPIDER_PROMPT_TOKEN_PENALTY", 0.0)
            self._spider_prompt_token_penalty_enabled = self._spider_prompt_token_penalty > 0.0
            seed = os.environ.get("CSD_CONSTRAINED_SAMPLING_SEED", "").strip()
            self._sampling_rng = random.Random(int(seed)) if seed else random.Random()
            vocab_sz = tokenizer.vocab_size if hasattr(tokenizer, "vocab_size") else len(tokenizer)
            self._forbid_first_token_ids: set = set()
            allow_leading_delimiter = os.environ.get("CSD_ALLOW_LEADING_DELIMITER", "").strip().lower() in {
                "1", "true", "yes", "on"
            }
            for vid in range(min(vocab_sz, 200000)):
                try:
                    s = _decode_single_token(tokenizer, vid)
                    if (s in HuggingFaceLMNative._FORBID_FIRST_STRINGS and not allow_leading_delimiter) or not s:
                        self._forbid_first_token_ids.add(vid)
                except Exception:
                    pass
            eos_id = getattr(tokenizer, "eos_token_id", None)
            if eos_id is not None:
                self._forbid_first_token_ids.add(int(eos_id))
            pad_id = getattr(tokenizer, "pad_token_id", None)
            if pad_id is not None and pad_id != eos_id:
                self._forbid_first_token_ids.add(int(pad_id))

        def ValidTokensIdsLogits(self) -> bool:
            return _valid_tokens_ids_logits_py(self.Tokens, self.Ids, self.Logits)

        def ValidTokensIdsLogitsAlways(self) -> None:
            assert self.ValidTokensIdsLogits()

        def TokenToId(self, token: str) -> int:
            return self._token_index[token]

        def IdToToken(self, id: int) -> str:
            return self.Tokens[id]

        def IdToLogit(self, id: int) -> float:
            return self.Logits[id]

        def TokenToLogit(self, token: str) -> float:
            return self.Logits[self._token_index[token]]

        def MaskToken(self, token: str) -> None:
            self.Logits[self._token_index[token]] = -1e9

        def MaskTokens(self, tokens_to_mask: list[str]) -> None:
            for token in tokens_to_mask:
                idx = self._token_index.get(token)
                if idx is not None:
                    self.Logits[idx] = -1e9

        def BiasToken(self, token: str, delta: float) -> None:
            idx = self._token_index[token]
            raw = self.Logits[idx] + delta
            if raw > 1e9:
                raw = 1e9
            elif raw < -1e9:
                raw = -1e9
            self.Logits[idx] = raw

        def BiasTokens(self, tokens_to_bias: list[str], delta: float) -> None:
            for token in tokens_to_bias:
                idx = self._token_index.get(token)
                if idx is None:
                    continue
                raw = self.Logits[idx] + delta
                if raw > 1e9:
                    raw = 1e9
                elif raw < -1e9:
                    raw = -1e9
                self.Logits[idx] = raw

        def ScaleToken(self, token: str, factor: float) -> None:
            idx = self._token_index[token]
            raw = self.Logits[idx] * factor
            if raw > 1e9:
                raw = 1e9
            elif raw < -1e9:
                raw = -1e9
            self.Logits[idx] = raw

        def ScaleTokens(self, tokens_to_scale: list[str], factor: float) -> None:
            for token in tokens_to_scale:
                idx = self._token_index.get(token)
                if idx is None:
                    continue
                raw = self.Logits[idx] * factor
                if raw > 1e9:
                    raw = 1e9
                elif raw < -1e9:
                    raw = -1e9
                self.Logits[idx] = raw

        def IsMasked(self, token: str) -> bool:
            return self.Logits[self._token_index[token]] <= -1e8

        def HasUnmaskedToken(self) -> bool:
            return any(v > -1e8 for v in self.Logits)

        def ResetForNewExample(self) -> None:
            """Clear per-example transient LM state between evaluations."""
            self.instruction_text = ""
            self._full_logits = None
            self._first_token_choice = False
            self._prompt_bias_cache_key = ""
            self._prompt_bias_tokens = set()
            self._prompt_bias_normalized = set()
            self._prompt_bias_text_lower = ""
            for i in range(len(self.Logits)):
                self.Logits[i] = 0.0

        def _refresh_prompt_bias_tokens(self) -> None:
            if not (self._spider_prompt_token_bonus_enabled or self._spider_prompt_token_penalty_enabled):
                self._prompt_bias_tokens = set()
                self._prompt_bias_normalized = set()
                self._prompt_bias_text_lower = ""
                self._prompt_bias_cache_key = self.instruction_text
                return
            if self._prompt_bias_cache_key == self.instruction_text:
                return
            self._prompt_bias_cache_key = self.instruction_text
            self._prompt_bias_tokens = _collect_prompt_sql_bias_tokens(self.instruction_text)
            self._prompt_bias_normalized = {
                _normalize_prompt_bias_token(tok) for tok in self._prompt_bias_tokens
                if _normalize_prompt_bias_token(tok)
            }
            self._prompt_bias_text_lower = (self.instruction_text or "").lower()

        def GenerateLogits(self, input_prefix: list) -> None:
            prefix_text = "".join(str(t) for t in input_prefix)
            full_prompt = self.instruction_text + prefix_text
            inputs = tokenizer(full_prompt, return_tensors="pt", add_special_tokens=False)
            if inputs["input_ids"].shape[-1] > self._max_input_len:
                inputs["input_ids"] = inputs["input_ids"][:, -self._max_input_len:]
                if "attention_mask" in inputs:
                    inputs["attention_mask"] = inputs["attention_mask"][:, -self._max_input_len:]
            inputs = inputs.to(self._input_device)

            self._first_token_choice = len(input_prefix) == 0
            with torch.no_grad():
                output = model(**inputs)
                logits = output.logits[0, -1, :].float().cpu()

            if self._first_token_choice and self._forbid_first_token_ids:
                for vid in self._forbid_first_token_ids:
                    if vid < logits.shape[0]:
                        logits[vid] = float("-inf")

            self._full_logits = logits
            _LOGIT_FORBIDDEN = -1e9
            for i, tid in enumerate(self._token_ids):
                v = float(logits[tid].item())
                if not math.isfinite(v):
                    v = _LOGIT_FORBIDDEN
                self.Logits[i] = v

        def ChooseNextToken(self) -> str:
            if not self.Logits:
                raise RuntimeError("No logits are available for constrained selection")
            if all((not math.isfinite(float(v))) or float(v) <= -1e8 for v in self.Logits):
                raise RuntimeError("No unmasked token is available after constrained masking")
            self._refresh_prompt_bias_tokens()

            def _is_prompt_grounded(token: str) -> bool:
                if not self._prompt_bias_tokens:
                    return False
                if token in self._prompt_bias_tokens:
                    return True
                norm = _normalize_prompt_bias_token(token)
                if not norm:
                    return False
                if norm in self._prompt_bias_normalized:
                    return True
                return len(norm) >= 3 and norm in self._prompt_bias_text_lower

            def _score(i: int) -> float:
                base = float(self.Logits[i])
                if not (math.isfinite(base) and base > -1e8):
                    return base
                token = self.Tokens[i]
                grounded = _is_prompt_grounded(token)
                if (
                    self._spider_prompt_token_bonus_enabled
                    and grounded
                ):
                    base += self._spider_prompt_token_bonus
                if (
                    self._spider_prompt_token_penalty_enabled
                    and not grounded
                    and not _token_is_sql_scaffold(token)
                ):
                    base -= self._spider_prompt_token_penalty
                return base

            if self._sample_constrained:
                best_i = _sample_from_masked_logits(
                    logits=[_score(i) for i in range(len(self.Logits))],
                    rng=self._sampling_rng,
                    temperature=self._sample_temperature,
                    top_p=self._sample_top_p,
                )
            else:
                best_i = max(range(len(self.Logits)), key=_score)
            if os.environ.get("CSD_LOGITS_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}:
                top_k = int(os.environ.get("CSD_LOGITS_TOP_K", "8") or "8")
                scored = [(i, v) for i, v in enumerate(self.Logits) if math.isfinite(v) and v > -1e8]
                scored.sort(key=lambda x: x[1], reverse=True)
                parts = []
                for i, v in scored[:max(1, top_k)]:
                    tok = str(self.Tokens[i]).replace("\n", "\\n")
                    parts.append(f"{repr(tok)}:{v:.2f}")
                print(f"    [LOGITS DEBUG] top={'; '.join(parts)}")
            return self.Tokens[best_i]

        def ChooseNextTokenUnconstrained(self) -> str:
            if self._full_logits is None:
                raise RuntimeError("Call GenerateLogits before ChooseNextTokenUnconstrained")
            logits = self._full_logits.clone()
            best_idx = int(logits.argmax().item())
            return _decode_single_token(tokenizer, best_idx)

        def MaskTokensExcept(self, valid_tokens: list) -> None:
            valid_indices = {
                self._token_index[str(token)]
                for token in valid_tokens
                if str(token) in self._token_index
            }
            if os.environ.get("CSD_DISALLOW_CONSTRAINED_WHITESPACE", "").strip().lower() in {"1", "true", "yes", "on"}:
                non_ws = {idx for idx in valid_indices if self.Tokens[idx].strip() != ""}
                if non_ws:
                    valid_indices = non_ws
            if self._first_token_choice:
                self._first_token_choice = False
                eos_str = (
                    _decode_single_token(tokenizer, tokenizer.eos_token_id)
                    if getattr(tokenizer, "eos_token_id", None) is not None
                    else ""
                )
                allow_leading_delimiter = os.environ.get("CSD_ALLOW_LEADING_DELIMITER", "").strip().lower() in {
                    "1", "true", "yes", "on"
                }
                forbid = {"", eos_str}
                if not allow_leading_delimiter:
                    forbid |= set(HuggingFaceLMNative._FORBID_FIRST_STRINGS)
                reduced = {idx for idx in valid_indices if self.Tokens[idx] not in forbid}
                if reduced:
                    valid_indices = reduced
            for i in range(len(self.Logits)):
                if i not in valid_indices:
                    self.Logits[i] = -1e9

    return HuggingFaceLMNative()
