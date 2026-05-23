"""
Model loading and management utilities for CSD evaluation.

Supports both HuggingFace and vLLM runtimes. The hot path remains tensorized:
next-token logits are captured as tensors, masking uses tensor ops, constrained
token selection is argmax over masked tensors, and unconstrained token selection
samples from the model distribution.
"""

from __future__ import annotations

import os
import multiprocessing as mp
import time
from collections import defaultdict
from contextlib import contextmanager

from typing import Any

import torch


# Per-component timing. Keyed by a short label. Values are (total_seconds, call_count).
# Printed periodically from GenerateLogits to break down where per-step time goes.
# Set CSD_DISABLE_TIMING=1 to turn off.
_TIMINGS: dict[str, list[float]] = defaultdict(lambda: [0.0, 0])
_TIMINGS_ENABLED = os.environ.get("CSD_DISABLE_TIMING", "") == ""
_TIMINGS_PRINT_EVERY = int(os.environ.get("CSD_TIMINGS_PRINT_EVERY", "10"))


@contextmanager
def _timed(label: str):
    """Accumulate wall-clock time under `label` into `_TIMINGS`."""
    if not _TIMINGS_ENABLED:
        yield
        return
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        slot = _TIMINGS[label]
        slot[0] += elapsed
        slot[1] += 1


def _print_timings_breakdown(header: str = "") -> None:
    if not _TIMINGS_ENABLED or not _TIMINGS:
        return
    total = sum(t for t, _ in _TIMINGS.values())
    if total <= 0:
        return
    lines = [f"[TIMING] {header} total={total:.2f}s"]
    for label in sorted(_TIMINGS.keys(), key=lambda k: -_TIMINGS[k][0]):
        secs, calls = _TIMINGS[label]
        pct = 100.0 * secs / total
        avg_ms = 1000.0 * secs / max(calls, 1)
        lines.append(f"  {label:<30} {secs:7.2f}s  ({pct:5.1f}%)  calls={calls:<5} avg={avg_ms:7.2f}ms")
    print("\n".join(lines), flush=True)

    # Also print parser-side timings if available. Lazy import avoids circular deps
    # and keeps this file independent of parser_utils.
    try:
        from synthesis.evaluate.benchmarks.common.parser_utils import print_parser_timings
        print_parser_timings(header=header)
    except Exception:
        pass

_RUNTIME_TOKENIZER_CACHE: dict[tuple[str, str], Any] = {}
_VLLM_ENGINE_CACHE: dict[tuple[Any, ...], tuple[Any, Any]] = {}

# vLLM's SamplingParams.logprobs controls how much of the next-token distribution
# is returned. `-1` means "full vocabulary" — for a 152k Qwen vocab this costs
# ~5-8s per step in Python-object construction + IPC alone. We only need the
# top of the distribution for argmax / masking / boost semantics: any token
# outside the top-K is effectively tail noise and gets masked to -1e9 (same
# value used by `MaskToken`), which is indistinguishable from a masked token.
# Raise this if strategies begin reporting all-masked argmaxes.
VLLM_TOPK_LOGPROBS = 1000

from transformers import AutoModelForCausalLM, AutoTokenizer


class _LogitsProxy:
    """Proxy for lm.Logits that writes through to tensor-backed storage."""

    def __init__(self, size, token_ids):
        self._size = size
        self._token_ids = token_ids
        self._constrained_tensor: torch.Tensor | None = None
        self._full_tensor: torch.Tensor | None = None

    def update_tensors(
        self,
        constrained_tensor: torch.Tensor,
        full_tensor: torch.Tensor,
    ) -> None:
        self._constrained_tensor = constrained_tensor
        self._full_tensor = full_tensor

    def __getitem__(self, idx: int):
        import _dafny

        with _timed("LogitsProxy.__getitem__"):
            if self._constrained_tensor is not None:
                return _dafny.BigRational(self._constrained_tensor[idx].item())
            return _dafny.BigRational(0)

    def __setitem__(self, idx: int, value) -> None:
        with _timed("LogitsProxy.__setitem__"):
            float_val = float(value)
            if self._constrained_tensor is not None:
                self._constrained_tensor[idx] = float_val
            if self._full_tensor is not None:
                full_id = self._token_ids[idx]
                self._full_tensor[full_id] = float_val

    def __len__(self) -> int:
        return self._size


torch.set_float32_matmul_precision("high")


def get_model_input_device(model) -> torch.device:
    """Find the device where a HuggingFace model expects inputs."""
    if hasattr(model, "hf_device_map") and model.hf_device_map:
        for key, device in model.hf_device_map.items():
            if "embed" in key.lower():
                return torch.device(f"cuda:{device}" if isinstance(device, int) else device)
        first_device = next(iter(model.hf_device_map.values()))
        return torch.device(f"cuda:{first_device}" if isinstance(first_device, int) else first_device)
    return next(model.parameters()).device


def get_max_input_length(model, tokenizer) -> int:
    """Choose a safe max input length for the runtime."""
    max_len = None
    if hasattr(model, "config") and getattr(model.config, "max_position_embeddings", None):
        max_len = int(model.config.max_position_embeddings)
    tok_max = getattr(tokenizer, "model_max_length", None)
    if tok_max and tok_max < 1_000_000:
        max_len = min(max_len, int(tok_max)) if max_len else int(tok_max)
    return max_len or 4096




def configure_vllm_multiprocessing() -> None:
    """Prefer spawn workers for vLLM to avoid CUDA re-init failures under fork."""
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    try:
        if mp.get_start_method(allow_none=True) is None:
            mp.set_start_method("spawn")
    except RuntimeError:
        # Another library may have already locked the start method.
        pass


_configure_vllm_multiprocessing = configure_vllm_multiprocessing


def load_runtime_tokenizer(model_name: str, backend: str = "huggingface"):
    """Load the tokenizer matching the requested runtime backend."""
    cache_key = (backend, model_name)
    cached = _RUNTIME_TOKENIZER_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if backend == "vllm":
        configure_vllm_multiprocessing()
        from vllm.transformers_utils.tokenizer import get_tokenizer

        tokenizer = get_tokenizer(model_name, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    _RUNTIME_TOKENIZER_CACHE[cache_key] = tokenizer
    return tokenizer


def _get_visible_devices_key() -> str:
    return os.environ.get("CUDA_VISIBLE_DEVICES", "ALL")


def _get_cached_vllm_engine(
    model_name: str,
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
    gpu_memory_utilization: float,
    max_model_len: int,
    enforce_eager: bool,
    vllm_kwargs: dict[str, Any],
):
    _configure_vllm_multiprocessing()
    from vllm import LLM

    cache_key = (
        model_name,
        _get_visible_devices_key(),
        tensor_parallel_size,
        pipeline_parallel_size,
        float(gpu_memory_utilization),
        int(max_model_len),
        bool(enforce_eager),
        repr(sorted(vllm_kwargs.items(), key=lambda item: item[0])),
    )
    cached = _VLLM_ENGINE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    print(f"Loading model: {model_name} on cuda with vLLM...")
    tokenizer = load_runtime_tokenizer(model_name, backend="vllm")
    llm = LLM(
        model=model_name,
        tokenizer=model_name,
        trust_remote_code=True,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        enforce_eager=enforce_eager,
        enable_prefix_caching=True,
        max_logprobs=-1,
        disable_log_stats=True,
        **vllm_kwargs,
    )
    _VLLM_ENGINE_CACHE[cache_key] = (llm, tokenizer)
    return llm, tokenizer


def max_cuda_devices_from_env(default: int = 1) -> int:
    """Max CUDA devices for local runs (override with VAS_MAX_CUDA_DEVICES or CSD_MAX_CUDA_DEVICES)."""
    raw = os.environ.get(
        "VAS_MAX_CUDA_DEVICES",
        os.environ.get("CSD_MAX_CUDA_DEVICES", str(default)),
    )
    try:
        return max(1, int(raw))
    except ValueError:
        return default


def resolve_vllm_tensor_parallel_size(requested: int | None = None) -> int:
    """Resolve vLLM tensor parallel size, capped by max_cuda_devices_from_env().

    When ``requested`` is None, use the env cap (``VAS_MAX_CUDA_DEVICES``, default 1) so
    ``CUDA_VISIBLE_DEVICES=2,3`` with ``VAS_MAX_CUDA_DEVICES=2`` spreads models across both GPUs.
    """
    cap = max_cuda_devices_from_env()
    tensor_parallel_size = cap if requested is None else requested
    return max(1, min(tensor_parallel_size, cap))


def limit_cuda_visible_devices(value: str | None, max_devices: int | None = None) -> str | None:
    """Keep at most ``max_devices`` entries from a comma-separated CUDA_VISIBLE_DEVICES value."""
    if not value:
        return value
    cap = max_devices if max_devices is not None else max_cuda_devices_from_env()
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if len(parts) <= cap:
        return value
    return ",".join(parts[:cap])


def visible_cuda_device_ids() -> list[str]:
    """Return CUDA device ids from CUDA_VISIBLE_DEVICES, or 0..N-1 when unset."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible:
        try:
            import torch

            return [str(i) for i in range(torch.cuda.device_count())]
        except Exception:
            return ["0"]
    return [part.strip() for part in visible.split(",") if part.strip()]


def pick_cuda_device_index_with_most_free_memory() -> int:
    """Pick the visible CUDA index with the largest free memory pool."""
    try:
        import torch

        if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
            return 0
        best_idx = 0
        best_free = -1
        for idx in range(torch.cuda.device_count()):
            free_bytes, _total_bytes = torch.cuda.mem_get_info(idx)
            if free_bytes > best_free:
                best_free = free_bytes
                best_idx = idx
        return best_idx
    except Exception:
        return 0


def narrow_cuda_visible_devices_to_index(device_index: int) -> str:
    """Restrict CUDA_VISIBLE_DEVICES to one physical id from the current visible set."""
    visible_ids = visible_cuda_device_ids()
    if not visible_ids:
        chosen = str(device_index)
    elif 0 <= device_index < len(visible_ids):
        chosen = visible_ids[device_index]
    else:
        chosen = visible_ids[0]
    os.environ["CUDA_VISIBLE_DEVICES"] = chosen
    return chosen


def clear_vllm_engine_cache() -> None:
    """Release cached vLLM engines before switching back to a generator model."""
    cached_engines = list(_VLLM_ENGINE_CACHE.values())
    _VLLM_ENGINE_CACHE.clear()

    for llm, _tokenizer in cached_engines:
        for attr_name in ("shutdown", "close"):
            maybe_shutdown = getattr(llm, attr_name, None)
            if callable(maybe_shutdown):
                try:
                    maybe_shutdown()
                except Exception:
                    pass

        engine = getattr(llm, "llm_engine", None)
        if engine is not None:
            for attr_name in ("shutdown", "close"):
                maybe_shutdown = getattr(engine, attr_name, None)
                if callable(maybe_shutdown):
                    try:
                        maybe_shutdown()
                    except Exception:
                        pass

    try:
        from vllm.distributed import destroy_distributed_environment, destroy_model_parallel

        destroy_model_parallel()
        destroy_distributed_environment()
    except Exception:
        pass

    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _get_vllm_quantization_kwargs(
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
) -> dict[str, Any]:
    """Translate project quantization flags to the installed vLLM config surface."""
    if load_in_4bit and load_in_8bit:
        raise ValueError("Choose at most one of load_in_4bit or load_in_8bit.")

    if not (load_in_4bit or load_in_8bit):
        return {}

    quant_config: dict[str, Any] = {
        "quant_method": "bitsandbytes",
    }
    if load_in_4bit:
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


class _TaskGuidanceState:
    """First-call-wins prompt guidance appended by generated CSDs."""

    MAX_GUIDANCE_CHARS = 1200
    HEADER = "Additional task guidance from CSD:"

    def __init__(self) -> None:
        self.accepted_guidance: str | None = None

    def reset(self) -> None:
        self.accepted_guidance = None

    def append(self, instruction_text: str, guidance: object) -> str:
        if self.accepted_guidance is not None:
            return instruction_text
        text = self._coerce_guidance(guidance)
        if not text:
            return instruction_text
        self.accepted_guidance = text
        separator = "\n" if instruction_text.endswith("\n") else "\n\n"
        return f"{instruction_text}{separator}{self.HEADER}\n{text}\n"

    def _coerce_guidance(self, guidance: object) -> str:
        text = str(guidance).strip()
        if not text:
            return ""
        return text[: self.MAX_GUIDANCE_CHARS]


class _TensorizedLMBase:
    """Shared tensorized behavior for Dafny LM wrappers."""

    def __init__(self, _dafny, tokenizer, tokens, tids, logits_device: torch.device | str = "cpu"):
        self._dafny = _dafny
        self.tokenizer = tokenizer
        self._Tokens = tokens
        self._token_ids = tids
        self.instruction_text = ""
        self._task_guidance = _TaskGuidanceState()
        self._logits_device = torch.device(logits_device)

        n = len(tids)
        self.Logits = _LogitsProxy(n, list(tids))
        self._logits_tensor = torch.zeros(n, dtype=torch.float32, device=self._logits_device)
        self._token_ids_tensor = torch.tensor(tids, dtype=torch.long, device=self._logits_device)
        self._full_logits: torch.Tensor | None = None
        self._generate_count = 0
        self._token_id_to_str: dict[int, str] = {}

        # Prefix-cache short-circuit state.
        self._last_full_prompt: str | None = None
        self._logits_dirty: bool = False
        self._cache_hits: int = 0

        self._token_str_to_indices = {}
        for i in range(n):
            token_str = self._to_str(tokens[i])
            self._token_str_to_indices.setdefault(token_str, []).append(i)

    def _to_str(self, obj):
        if isinstance(obj, str):
            return obj
        try:
            return "".join(obj[i] for i in range(len(obj)))
        except Exception:
            return str(obj)

    def _prefix_text(self, prefix) -> str:
        return "".join(self._to_str(prefix[i]) for i in range(len(prefix)))

    def ResetTaskGuidance(self):
        self._task_guidance.reset()

    def AppendTaskGuidance(self, guidance):
        self.instruction_text = self._task_guidance.append(
            self.instruction_text,
            self._to_str(guidance),
        )

    @property
    def task_guidance(self) -> str | None:
        return self._task_guidance.accepted_guidance

    def _token_str_from_id(self, token_id: int) -> str:
        token_id = int(token_id)
        cached = self._token_id_to_str.get(token_id)
        if cached is None:
            cached = self.tokenizer.decode([token_id])
            self._token_id_to_str[token_id] = cached
        return cached

    def _dafny_prefix_from_token_strs(self, token_strs: list[str]):
        return self._dafny.SeqWithoutIsStrInference([self._dafny.Seq(token) for token in token_strs])

    def _token_strs_from_text(self, text: str) -> list[str]:
        if not text:
            return []
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        return [self._token_str_from_id(token_id) for token_id in token_ids]

    def _build_unconstrained_chunk_result(self, token_ids, open_span_token, eos_token, max_new_tokens: int):
        if max_new_tokens <= 0:
            return self._dafny_prefix_from_token_strs([]), False, False, 0

        open_span_str = self._to_str(open_span_token)
        eos_str = self._to_str(eos_token)
        chunk_tokens: list[str] = []
        chunk_text = ""
        steps_used = 0
        stopped_on_open = False
        stopped_on_eos = False

        for raw_token_id in token_ids:
            if steps_used >= max_new_tokens:
                break
            token_str = self._token_str_from_id(int(raw_token_id))
            steps_used += 1
            if token_str == eos_str:
                stopped_on_eos = True
                break

            candidate_text = chunk_text + token_str
            open_idx = candidate_text.find(open_span_str)
            if open_idx != -1:
                prefix_text = candidate_text[:open_idx]
                chunk_tokens = self._token_strs_from_text(prefix_text)
                chunk_tokens.append(open_span_str)
                stopped_on_open = True
                break

            chunk_tokens.append(token_str)
            chunk_text = candidate_text

        return self._dafny_prefix_from_token_strs(chunk_tokens), stopped_on_open, stopped_on_eos, steps_used

    def IdToLogit(self, id_):
        with _timed("IdToLogit"):
            return self._dafny.BigRational(self._logits_tensor[id_].item())

    def MaskToken(self, token):
        with _timed("MaskToken"):
            token_id = self.TokenToId(token)
            self._logits_tensor[token_id] = -1e9
            self._logits_dirty = True

    def IsMasked(self, token):
        with _timed("IsMasked"):
            return self._logits_tensor[self.TokenToId(token)].item() == -1e9

    def _finalize_full_logits(self, full_logits: torch.Tensor) -> None:
        full_logits = full_logits.float().to(self._logits_device)
        self._full_logits = full_logits
        self._logits_tensor = full_logits[self._token_ids_tensor]
        self.Logits.update_tensors(self._logits_tensor, self._full_logits)

    def _sample_full_token_id(self) -> int:
        if self._full_logits is None:
            raise RuntimeError("Must call GenerateLogits before sampling unconstrained tokens")

        probs = torch.softmax(self._full_logits, dim=0)
        if torch.isnan(probs).any() or torch.sum(probs).item() <= 0.0:
            return int(self._full_logits.argmax().item())
        sampled_idx = int(torch.multinomial(probs, num_samples=1).item())
        return sampled_idx

    def _finalize_from_logprob_dict(self, logprob_dict: dict[int, Any]) -> None:
        # vLLM returns next-token logprobs as a Python dict. We previously fetched
        # the *full* vocab (logprobs=-1) and looped once per vocab entry — 152k
        # Python objects + a 152k-step fill loop per step, dominating runtime.
        #
        # Now we ask for top-K only (see VLLM_TOPK_LOGPROBS) and vectorize the
        # dict -> tensor conversion. Missing token IDs are filled with -1e9,
        # matching the Dafny invariant `Logits[i] >= -1e9` (used elsewhere for
        # masking). For argmax / additive-transform semantics this is equivalent
        # to "masked out" — tokens that never made it into the top-K are treated
        # as un-selectable, which is the correct semantic for constrained
        # decoding (any tail-distribution token is effectively noise anyway).
        if self._token_ids_tensor.numel() > 0:
            token_ids_max_self = int(self._token_ids_tensor.max().item())
        else:
            token_ids_max_self = 0

        if logprob_dict:
            ids_list = [int(tid) for tid in logprob_dict.keys()]
            logprobs_list = [float(info.logprob) for info in logprob_dict.values()]
            max_token_id = max(max(ids_list), token_ids_max_self)
        else:
            ids_list = []
            logprobs_list = []
            max_token_id = token_ids_max_self

        full_scores = torch.full(
            (max_token_id + 1,),
            -1e9,
            dtype=torch.float32,
            device=self._logits_device,
        )
        if ids_list:
            ids_tensor = torch.tensor(ids_list, dtype=torch.long, device=self._logits_device)
            scores_tensor = torch.tensor(logprobs_list, dtype=torch.float32, device=self._logits_device)
            full_scores.scatter_(0, ids_tensor, scores_tensor)

        self._finalize_full_logits(full_scores)

    def ChooseNextToken(self):
        with _timed("ChooseNextToken"):
            best_idx = int(self._logits_tensor.argmax().item())
            return self._Tokens[best_idx]

    def ChooseNextTokenUnconstrained(self):
        with _timed("ChooseNextTokenUnconstrained"):
            if self._full_logits is None:
                raise RuntimeError("Must call GenerateLogits before ChooseNextTokenUnconstrained")
            sampled_idx = self._sample_full_token_id()
            return self._dafny.Seq(self.tokenizer.decode([sampled_idx]))

    def _token_indices_for_token(self, token) -> list[int]:
        token_str = self._to_str(token)
        return list(self._token_str_to_indices.get(token_str, []))

    def _expand_full_mask(self, full_mask: torch.Tensor) -> torch.Tensor:
        if self._full_logits is None:
            raise RuntimeError("Must call GenerateLogits before applying parser masks")
        full_mask = full_mask.to(dtype=torch.bool, device=self._full_logits.device)
        if full_mask.numel() < self._full_logits.numel():
            padding = torch.zeros(
                self._full_logits.numel() - full_mask.numel(),
                dtype=torch.bool,
                device=self._full_logits.device,
            )
            full_mask = torch.cat((full_mask, padding))
        elif full_mask.numel() > self._full_logits.numel():
            full_mask = full_mask[: self._full_logits.numel()]
        return full_mask

    def _subset_mask_from_full_mask(self, full_mask: torch.Tensor) -> torch.Tensor:
        subset = full_mask[self._token_ids_tensor.to(full_mask.device)]
        return subset.to(dtype=torch.bool, device=self._logits_tensor.device)

    def _parser_full_mask(self, parser, prefix):
        if hasattr(parser, "_get_accept_mask_for_prefix"):
            return parser._get_accept_mask_for_prefix(prefix)
        valid_tokens = parser.ValidNextTokens(prefix)
        fallback_mask = torch.zeros(len(self._token_ids), dtype=torch.bool)
        for i in range(len(valid_tokens)):
            token_str = self._to_str(valid_tokens[i])
            indices = self._token_str_to_indices.get(token_str)
            if indices is not None:
                fallback_mask[indices] = True
        return fallback_mask

    def MaskValidNextAndEos(self, parser, prefix, eosToken):
        with _timed("MaskValidNextAndEos"):
            full_mask = self._parser_full_mask(parser, prefix)
            if full_mask.numel() == len(self._token_ids):
                subset_mask = full_mask.to(dtype=torch.bool, device=self._logits_tensor.device)
                full_mask = None
            else:
                full_mask = self._expand_full_mask(full_mask)
                subset_mask = self._subset_mask_from_full_mask(full_mask)

            eos_indices = self._token_indices_for_token(eosToken)
            if eos_indices:
                subset_mask[eos_indices] = True
                if full_mask is not None:
                    eos_full_ids = self._token_ids_tensor[eos_indices].to(full_mask.device)
                    full_mask[eos_full_ids] = True

            if torch.sum(subset_mask).item() == 0:
                raise RuntimeError("MaskValidNextAndEos found no valid next tokens including EOS")

            self._logits_tensor.masked_fill_(~subset_mask, -1e9)
            if self._full_logits is not None and full_mask is not None:
                self._full_logits.masked_fill_(~full_mask, -1e9)
            self._logits_dirty = True
            self.Logits.update_tensors(self._logits_tensor, self._full_logits)

    def BoostValidNextAndEos(self, parser, prefix, amount, eosToken):
        with _timed("BoostValidNextAndEos"):
            amount_f = float(amount)
            full_mask = self._parser_full_mask(parser, prefix)
            if full_mask.numel() == len(self._token_ids):
                subset_mask = full_mask.to(dtype=torch.bool, device=self._logits_tensor.device)
                full_mask = None
            else:
                full_mask = self._expand_full_mask(full_mask)
                subset_mask = self._subset_mask_from_full_mask(full_mask)

            eos_indices = self._token_indices_for_token(eosToken)
            if eos_indices:
                subset_mask[eos_indices] = True
                if full_mask is not None:
                    eos_full_ids = self._token_ids_tensor[eos_indices].to(full_mask.device)
                    full_mask[eos_full_ids] = True

            self._logits_tensor[subset_mask] = torch.clamp(
                self._logits_tensor[subset_mask] + amount_f, min=-1e9, max=1e9
            )
            if self._full_logits is not None and full_mask is not None:
                self._full_logits[full_mask] = torch.clamp(
                    self._full_logits[full_mask] + amount_f, min=-1e9, max=1e9
                )
            self.Logits.update_tensors(self._logits_tensor, self._full_logits)
            self._logits_dirty = True

    def MaskTokensExcept(self, valid_tokens, debug=False):
        with _timed("MaskTokensExcept"):
            accept_mask = torch.zeros(len(self._token_ids), dtype=torch.bool)
            for i in range(len(valid_tokens)):
                token_str = self._to_str(valid_tokens[i])
                indices = self._token_str_to_indices.get(token_str)
                if indices is not None:
                    accept_mask[indices] = True

            if torch.sum(accept_mask) == 0:
                valid_preview = [self._to_str(valid_tokens[i]) for i in range(min(len(valid_tokens), 10))]
                raise RuntimeError(
                    "MaskTokensExcept found no LM tokens matching the provided valid token set. "
                    f"Sample valid tokens: {valid_preview}"
                )

            if len(self._logits_tensor) > len(accept_mask):
                padding = torch.zeros(len(self._logits_tensor) - len(accept_mask), dtype=torch.bool)
                accept_mask = torch.cat((accept_mask, padding))

            self._logits_tensor.masked_fill_(~accept_mask.to(self._logits_tensor.device), -1e9)
            self._logits_dirty = True


def _build_tokens_dafny(_dafny, tokenizer, token_ids):
    return _dafny.SeqWithoutIsStrInference(
        [_dafny.Seq(tokenizer.decode([tid])) for tid in token_ids]
    )


def create_huggingface_lm(
    model_name: str,
    device: str,
    VerifiedDecoderAgent,
    _dafny,
    token_ids=None,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
):
    """Create a HuggingFace LM wrapped with a Dafny-compatible interface."""
    prec_str = "FP16"
    if load_in_4bit:
        prec_str = "4-bit"
    elif load_in_8bit:
        prec_str = "8-bit"

    print(f"Loading model: {model_name} on {device}... ({prec_str})")
    tokenizer = load_runtime_tokenizer(model_name, backend="huggingface")

    if device.startswith("cuda"):
        kwargs = {
            "pretrained_model_name_or_path": model_name,
            "trust_remote_code": True,
            "device_map": "auto",
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

        model = AutoModelForCausalLM.from_pretrained(**kwargs)
        input_device = get_model_input_device(model)
        print(f"Model loaded across {torch.cuda.device_count()} GPU(s), inputs go to {input_device}")
    elif device == "mps" and torch.backends.mps.is_available():
        # Apple Silicon path: FP16 on Metal. bitsandbytes is unsupported on MPS,
        # so load_in_4bit/load_in_8bit flags are ignored if requested here.
        if load_in_4bit or load_in_8bit:
            print("⚠️  4/8-bit quantization is not supported on MPS — loading FP16 instead.")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        model = model.to("mps")
        input_device = torch.device("mps")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float32,
        )
        input_device = torch.device("cpu")

    model.eval()

    if token_ids is None:
        token_ids = list(range(len(tokenizer)))

    tokens_dafny = _build_tokens_dafny(_dafny, tokenizer, token_ids)

    class HuggingFaceLM(_TensorizedLMBase, VerifiedDecoderAgent.LM):
        def __init__(self, hf_model, hf_tokenizer, tokens, tids, dev):
            VerifiedDecoderAgent.LM.__init__(self)
            _TensorizedLMBase.__init__(self, _dafny, hf_tokenizer, tokens, tids, logits_device=dev)
            self.model = hf_model
            self._input_device = dev
            self._max_input_len = get_max_input_length(hf_model, hf_tokenizer)

        def GenerateLogits(self, input_prefix):
            prefix_text = self._prefix_text(input_prefix)
            full_prompt = self.instruction_text + prefix_text

            # Prefix-cache short-circuit
            if full_prompt == self._last_full_prompt and not self._logits_dirty:
                self._cache_hits += 1
                return

            self._generate_count += 1
            if self._generate_count % 10 == 0:
                print(f"    [PROGRESS] GenerateLogits call #{self._generate_count}, prefix length: {len(input_prefix)}, cache_hits={self._cache_hits}")

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
                output = self.model(**inputs)
                logits = output.logits[0, -1, :]

            self._finalize_full_logits(logits)
            self._last_full_prompt = full_prompt
            self._logits_dirty = False

        def GenerateUnconstrainedChunk(self, input_prefix, maxNewTokens, openSpanToken, eosToken):
            max_new_tokens = int(maxNewTokens)
            if max_new_tokens <= 0:
                return self._build_unconstrained_chunk_result([], openSpanToken, eosToken, 0)

            prefix_text = self._prefix_text(input_prefix)
            full_prompt = self.instruction_text + prefix_text
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
            prompt_len = inputs["input_ids"].shape[-1]

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            token_ids = outputs[0, prompt_len:].tolist()
            return self._build_unconstrained_chunk_result(token_ids, openSpanToken, eosToken, max_new_tokens)

    return HuggingFaceLM(model, tokenizer, tokens_dafny, token_ids, input_device)


def create_vllm_lm(
    model_name: str,
    device: str,
    VerifiedDecoderAgent,
    _dafny,
    token_ids=None,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    tensor_parallel_size: int | None = None,
    pipeline_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.8,
    max_model_len: int = 16384,
    enforce_eager: bool = True,
):
    """Create a vLLM-backed LM wrapper with tensorized logits capture."""
    if not device.startswith("cuda"):
        raise ValueError("vLLM runtime currently requires a CUDA device in this project.")

    _configure_vllm_multiprocessing()
    from vllm import SamplingParams

    tensor_parallel_size = resolve_vllm_tensor_parallel_size(tensor_parallel_size)
    vllm_kwargs = _get_vllm_quantization_kwargs(
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
    )
    llm, tokenizer = _get_cached_vllm_engine(
        model_name=model_name,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        enforce_eager=enforce_eager,
        vllm_kwargs=vllm_kwargs,
    )

    if token_ids is None:
        token_ids = list(range(len(tokenizer)))

    tokens_dafny = _build_tokens_dafny(_dafny, tokenizer, token_ids)

    class VllmLM(_TensorizedLMBase, VerifiedDecoderAgent.LM):
        def __init__(self, engine, tok, tokens, tids):
            VerifiedDecoderAgent.LM.__init__(self)
            _TensorizedLMBase.__init__(self, _dafny, tok, tokens, tids, logits_device=torch.device(device))
            self.engine = engine

        def GenerateLogits(self, input_prefix):
            with _timed("GenerateLogits.total"):
                with _timed("GenerateLogits.prefix_text"):
                    prefix_text = self._prefix_text(input_prefix)
                    full_prompt = self.instruction_text + prefix_text

                # Prefix-cache short-circuit
                if full_prompt == self._last_full_prompt and not self._logits_dirty:
                    self._cache_hits += 1
                    return

                self._generate_count += 1
                if self._generate_count % 10 == 0:
                    print(f"    [PROGRESS] GenerateLogits call #{self._generate_count}, prefix length: {len(input_prefix)}, cache_hits={self._cache_hits}")

                with _timed("GenerateLogits.sampling_params_construct"):
                    sampling_params = SamplingParams(
                        max_tokens=1,
                        temperature=0.0,
                        logprobs=VLLM_TOPK_LOGPROBS,
                        detokenize=False,
                    )

                with _timed("GenerateLogits.engine_generate"):
                    outputs = self.engine.generate([full_prompt], sampling_params=sampling_params, use_tqdm=False)

                if not outputs or not outputs[0].outputs:
                    raise RuntimeError("vLLM returned no generation outputs for logits capture.")

                with _timed("GenerateLogits.extract_logprobs"):
                    logprob_steps = outputs[0].outputs[0].logprobs
                    if not logprob_steps:
                        raise RuntimeError("vLLM did not return next-token logprobs.")

                with _timed("GenerateLogits.finalize_from_dict"):
                    self._finalize_from_logprob_dict(logprob_steps[0])

                self._last_full_prompt = full_prompt
                self._logits_dirty = False

                if self._generate_count % _TIMINGS_PRINT_EVERY == 0:
                    _print_timings_breakdown(header=f"after {self._generate_count} GenerateLogits calls")

        def GenerateUnconstrainedChunk(self, input_prefix, maxNewTokens, openSpanToken, eosToken):
            max_new_tokens = int(maxNewTokens)
            if max_new_tokens <= 0:
                return self._build_unconstrained_chunk_result([], openSpanToken, eosToken, 0)

            prefix_text = self._prefix_text(input_prefix)
            full_prompt = self.instruction_text + prefix_text
            sampling_params = SamplingParams(
                max_tokens=max_new_tokens,
                temperature=0.0,
                detokenize=False,
            )
            outputs = self.engine.generate([full_prompt], sampling_params=sampling_params, use_tqdm=False)
            if not outputs or not outputs[0].outputs:
                raise RuntimeError("vLLM returned no generation outputs for unconstrained chunk capture.")

            token_ids = outputs[0].outputs[0].token_ids
            return self._build_unconstrained_chunk_result(token_ids, openSpanToken, eosToken, max_new_tokens)

    return VllmLM(llm, tokenizer, tokens_dafny, token_ids)


def create_runtime_lm(
    model_name: str,
    backend: str,
    device: str,
    VerifiedDecoderAgent,
    _dafny,
    token_ids=None,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    vllm_tensor_parallel_size: int | None = None,
    vllm_pipeline_parallel_size: int = 1,
    vllm_gpu_memory_utilization: float = 0.8,
    vllm_max_model_len: int = 16384,
    vllm_enforce_eager: bool = True,
):
    """Create the requested runtime LM backend."""
    if backend == "vllm":
        return create_vllm_lm(
            model_name=model_name,
            device=device,
            VerifiedDecoderAgent=VerifiedDecoderAgent,
            _dafny=_dafny,
            token_ids=token_ids,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            tensor_parallel_size=vllm_tensor_parallel_size,
            pipeline_parallel_size=vllm_pipeline_parallel_size,
            gpu_memory_utilization=vllm_gpu_memory_utilization,
            max_model_len=vllm_max_model_len,
            enforce_eager=vllm_enforce_eager,
        )

    return create_huggingface_lm(
        model_name=model_name,
        device=device,
        VerifiedDecoderAgent=VerifiedDecoderAgent,
        _dafny=_dafny,
        token_ids=token_ids,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
    )
