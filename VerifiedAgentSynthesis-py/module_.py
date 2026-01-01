import sys
from typing import Callable, Any, TypeVar, NamedTuple
from math import floor
from itertools import count

import _dafny as _dafny
import System_ as System_

# Module: module_

# NOTE:
# The Dafny `CRANE` module declares several `{:extern}` functions. The generated
# `CRANE.py` calls them as `CRANE.default__.<FnName>`. This `module_.py` file is
# the intended place to provide those Python implementations.

import os
import math
import random
from typing import Iterable, Mapping, Sequence, List, Tuple, Optional, Union, Any as TypingAny


# -----------------------------
# Sampling strategy parameters
# -----------------------------
#
# IMPORTANT CONFIG NOTE:
# The Dafny `SamplingStrategy` datatype has no parameters/fields, so any real
# sampling hyperparameters MUST be provided out-of-band via environment
# variables (or the defaults below will be used).
#
# If you want reproducible or non-default behavior, set these before running:
# - CRANE_TEMPERATURE : float  (default: 1.0)  used by Temperature
# - CRANE_TOPK        : int    (default: 50)   used by TopK
# - CRANE_TOP_P       : float  (default: 0.9)  used by Nucleus (top-p)
# - CRANE_RANDOM_SEED : string (default: unset) seeds RNG for deterministic sampling
#
# If these env vars differ across runs/machines, Temperature/TopK/Nucleus output
# can differ accordingly.

DEFAULT_TEMPERATURE: float = 1.0
DEFAULT_TOPK: int = 50
DEFAULT_TOP_P: float = 0.9


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None:
        return default
    try:
        return float(v)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None:
        return default
    try:
        return int(v)
    except ValueError:
        return default


# Avoid PEP604 (`X | Y`) so this file runs on Python <3.10 as well.
_RNG_SEED_CACHE: TypingAny = object()  # either the current env seed value, or sentinel
_RNG_INSTANCE: Optional[random.Random] = None


def _get_rng() -> random.Random:
    global _RNG_SEED_CACHE, _RNG_INSTANCE
    seed = os.getenv("CRANE_RANDOM_SEED")
    if _RNG_INSTANCE is None or seed != _RNG_SEED_CACHE:
        _RNG_SEED_CACHE = seed
        _RNG_INSTANCE = random.Random(seed)
    return _RNG_INSTANCE


def _to_list_candidates(candidates: Any) -> List[Any]:
    # Accept Dafny runtime Set/Seq or plain Python iterables.
    if candidates is None:
        return []
    elems = getattr(candidates, "Elements", None)
    if elems is not None:
        # For _dafny.Set, Elements is the set itself (iterable)
        return list(elems)
    return list(candidates)


def _logit_items_for_candidates(logits: Mapping[Any, float], candidates: Sequence[Any]) -> List[Tuple[Any, float]]:
    # Precondition in Dafny: forall t in candidates ==> t in logits
    return [(t, float(logits[t])) for t in candidates]

def _tok_key(t: Any) -> str:
    # Deterministic, stable comparison key across runs for tie-breaking.
    f = getattr(t, "__dafnystr__", None)
    if callable(f):
        try:
            return f()
        except Exception:
            pass
    return repr(t)


def _argmax(items: Sequence[Tuple[Any, float]]) -> Any:
    if not items:
        raise ValueError("argmax over empty candidate set")
    # Tie-breaking: deterministic by token string representation.
    best_t, best_s = items[0]
    for t, s in items[1:]:
        if s > best_s:
            best_t, best_s = t, s
        elif s == best_s and _tok_key(t) < _tok_key(best_t):
            best_t, best_s = t, s
    return best_t


def _softmax_probs(items: Sequence[Tuple[Any, float]], temperature: float) -> List[Tuple[Any, float]]:
    if not items:
        return []
    if temperature <= 0.0 or not math.isfinite(temperature):
        # Treat invalid temperature as greedy.
        t = _argmax(items)
        return [(tok, 1.0 if tok == t else 0.0) for tok, _ in items]

    scaled = [(t, s / temperature) for t, s in items]
    m = max(s for _, s in scaled)
    exps = [(t, math.exp(s - m)) for t, s in scaled]
    z = sum(v for _, v in exps)
    if z <= 0.0 or not math.isfinite(z):
        # Fallback to uniform
        p = 1.0 / len(items)
        return [(t, p) for t, _ in items]
    return [(t, v / z) for t, v in exps]


def _sample_from_probs(probs: Sequence[Tuple[Any, float]]) -> Any:
    if not probs:
        raise ValueError("cannot sample from empty distribution")
    r = _get_rng().random()
    acc = 0.0
    last = probs[-1][0]
    for t, p in probs:
        acc += p
        if r <= acc:
            return t
        last = t
    # In case of floating point roundoff
    return last


def SampleWithStrategy(logits: Mapping[Any, float], candidates: Any, strategy: Any) -> Any:
    """
    Implements the Dafny extern:
      function SampleWithStrategy(logits: Logits, candidates: set<Token>, strategy: SamplingStrategy): Token

    - ArgMax: greedy max-logit selection
    - Temperature: softmax(logits / T) sampling (T from env `CRANE_TEMPERATURE`, default 1.0)
    - TopK: restrict to top-K logits, then sample from softmax over that set (K from env `CRANE_TOPK`, default 50)
    - Nucleus: restrict to smallest set with cumulative prob >= top_p (p from env `CRANE_TOP_P`, default 0.9),
               then sample from renormalized probs over that set
    """
    cand_list = _to_list_candidates(candidates)
    if not cand_list:
        raise ValueError("SampleWithStrategy called with empty candidates")

    items = _logit_items_for_candidates(logits, cand_list)

    # ArgMax
    if getattr(strategy, "is_ArgMax", False):
        return _argmax(items)

    # Temperature
    if getattr(strategy, "is_Temperature", False):
        T = _env_float("CRANE_TEMPERATURE", DEFAULT_TEMPERATURE)
        probs = _softmax_probs(items, T)
        return _sample_from_probs(probs)

    # TopK
    if getattr(strategy, "is_TopK", False):
        k = _env_int("CRANE_TOPK", DEFAULT_TOPK)
        if k <= 0:
            return _argmax(items)
        # Sort by score descending, then stable token key ascending for determinism.
        items_sorted = sorted(items, key=lambda x: (-x[1], _tok_key(x[0])))
        top_items = items_sorted[: min(k, len(items_sorted))]
        probs = _softmax_probs(top_items, 1.0)
        return _sample_from_probs(probs)

    # Nucleus (top-p)
    if getattr(strategy, "is_Nucleus", False):
        top_p = _env_float("CRANE_TOP_P", DEFAULT_TOP_P)
        # Clamp to a sensible range; if invalid, fall back to greedy
        if not math.isfinite(top_p) or top_p <= 0.0:
            return _argmax(items)
        top_p = min(1.0, top_p)

        base_probs = _softmax_probs(items, 1.0)
        # Sort by prob descending, then token key ascending for determinism.
        base_sorted = sorted(base_probs, key=lambda x: (-x[1], _tok_key(x[0])))

        nucleus: List[Tuple[Any, float]] = []
        cum = 0.0
        for t, p in base_sorted:
            nucleus.append((t, p))
            cum += p
            if cum >= top_p:
                break
        # Renormalize nucleus probs
        z = sum(p for _, p in nucleus)
        if z <= 0.0 or not math.isfinite(z):
            return _argmax(items)
        nucleus = [(t, p / z) for t, p in nucleus]
        return _sample_from_probs(nucleus)

    # Unknown strategy: default to ArgMax (safe and deterministic)
    return _argmax(items)
