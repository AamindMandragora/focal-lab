# Runtime extern implementations for generated Dafny->Python code.
#
# NOTE: This file is overwritten by scripts/synthesize_csd.py.
import os
import re
import json
import time
import urllib.request
from typing import Any, Dict, Optional

# Optional Lark-based parsing for GSM grammar (used to judge constrained windows).
try:
    from lark import Lark
    from lark.exceptions import UnexpectedInput, UnexpectedEOF
except Exception:  # pragma: no cover
    Lark = None
    UnexpectedInput = Exception
    UnexpectedEOF = Exception

# Metrics (reset per-example by the evaluation harness).
METRICS: Dict[str, Any] = {
    "unconstrained_calls": 0,
    "constrained_calls": 0,
    "total_latency_s": 0.0,
    "last_success_source": None,  # 'unconstrained'|'constrained'|None
}

def reset_metrics():
    METRICS["unconstrained_calls"] = 0
    METRICS["constrained_calls"] = 0
    METRICS["total_latency_s"] = 0.0
    METRICS["last_success_source"] = None

def _http_post_json(url: str, payload: Dict[str, Any], headers: Dict[str, str]) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body)

def _vllm_generate(prompt: str, max_new_tokens: int) -> str:
    base_url = os.environ.get("QWEN_BASE_URL", "").rstrip("/")
    model = os.environ.get("QWEN_MODEL", "")
    api_key = os.environ.get("QWEN_API_KEY", "")
    if not base_url or not model:
        raise RuntimeError("Missing QWEN_BASE_URL/QWEN_MODEL env vars for runtime generation.")

    url = base_url + "/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.7,
        "max_tokens": max_new_tokens,
    }
    t0 = time.time()
    data = _http_post_json(url, payload, headers)
    METRICS["total_latency_s"] += time.time() - t0
    return data["choices"][0]["message"]["content"]

# ---- Parsing (GSM lark grammar) -----------------------------------------
_PARSER = None
def _get_parser():
    global _PARSER
    if _PARSER is not None:
        return _PARSER
    if Lark is None:
        raise RuntimeError("lark is not installed; cannot parse gsm.lark.")
    grammar_path = os.environ.get("GSM_LARK_PATH", "./grammars/gsm.lark")
    with open(grammar_path, "r", encoding="utf-8") as f:
        grammar = f.read()
    _PARSER = Lark(grammar, start="syncode", parser="lalr")
    return _PARSER

_WINDOW_RE = re.compile(r"<<(.*?)>>", flags=re.DOTALL)

def _last_complete_window(text: str) -> Optional[str]:
    ms = list(_WINDOW_RE.finditer(text))
    if not ms:
        return None
    m = ms[-1]
    return text[m.start():m.end()]

def ParseOk(g: int, s: str) -> bool:
    # CRANE-style: we consider the output parse-ok if it contains at least one
    # complete constrained window << ... >> that parses under the grammar.
    w = _last_complete_window(s)
    if w is None:
        return False
    try:
        _get_parser().parse(w)
        return True
    except Exception:
        return False

def ParseOkPrefix(g: int, s: str) -> bool:
    # Prefix-ok if either:
    # - there is a complete window that parses, OR
    # - there is an unfinished window that could still be completed.
    w = _last_complete_window(s)
    if w is not None:
        return ParseOk(g, s)
    # unfinished '<<' without matching '>>'
    return ("<<" in s) and (">>" not in s.split("<<")[-1])

# ---- Externs used by CSD.Run -------------------------------------------
def LLM__Generate(prompt: str, maxSteps: int, strategy: Any) -> str:
    METRICS["unconstrained_calls"] += 1
    # maxSteps is a token-budget in Dafny; approximate it as max_tokens for the model.
    completion = _vllm_generate(prompt, max_new_tokens=int(maxSteps))
    return prompt + completion

_CD_BACKEND = None

def _load_constraineddecoding_backend():
    """Load ConstrainedDecoding-py/ConstrainedDecoding.py without clobbering our module_."""
    global _CD_BACKEND
    if _CD_BACKEND is not None:
        return _CD_BACKEND

    repo_root = os.path.abspath(os.environ.get("FOCAL_LAB_ROOT", os.getcwd()))
    cd_dir = os.path.join(repo_root, "ConstrainedDecoding-py")
    cd_file = os.path.join(cd_dir, "ConstrainedDecoding.py")
    cd_module_file = os.path.join(cd_dir, "module_.py")
    if not os.path.exists(cd_file):
        raise RuntimeError(f"Missing {cd_file}; cannot use Option A constrained backend.")

    import importlib.util
    import sys as _sys

    # Load ConstrainedDecoding-py/module_.py as an isolated module.
    spec_mod = importlib.util.spec_from_file_location("cd_module_", cd_module_file)
    cd_module_ = importlib.util.module_from_spec(spec_mod)
    assert spec_mod and spec_mod.loader
    spec_mod.loader.exec_module(cd_module_)  # type: ignore

    # Temporarily swap sys.modules['module_'] so ConstrainedDecoding imports its own module_.
    prev_module_ = _sys.modules.get("module_")
    _sys.modules["module_"] = cd_module_
    try:
        spec = importlib.util.spec_from_file_location("cd_constrained", cd_file)
        cd = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(cd)  # type: ignore
    finally:
        if prev_module_ is not None:
            _sys.modules["module_"] = prev_module_
        else:
            _sys.modules.pop("module_", None)

    _CD_BACKEND = cd
    return cd

def ConstrainedGenerate(g: int, policy: Any, prompt: str, maxSteps: int) -> str:
    """CRANE-style constrained backend (Option A): reuse ConstrainedDecoding-py."""
    METRICS["constrained_calls"] += 1
    cd = _load_constraineddecoding_backend()
    # This decoder already alternates between unconstrained reasoning and constrained windows
    # based on << >> markers and applies grammar constraints inside the window.
    out = cd.default__.ConstrainedDecode(prompt, int(maxSteps))
    METRICS["last_success_source"] = "constrained"
    return out

def ConstrainedCompleteFromPrefix(g: int, policy: Any, prefix: str, maxSteps: int) -> str:
    METRICS["constrained_calls"] += 1
    # Option A: just run the same CRANE-style decoder starting from the prefix.
    out = ConstrainedGenerate(g, policy, prefix, maxSteps)
    METRICS["last_success_source"] = "constrained"
    return out

# Out-of-scope hooks still referenced by compiled code paths.
def SemanticOk(g: int, s: str) -> bool:
    return True

def RepairTransform(s: str) -> str:
    return s

def ConstrainedSearchGenerate(g: int, policy: Any, beamWidth: int, prompt: str, maxSteps: int) -> str:
    return ConstrainedGenerate(g, policy, prompt, maxSteps)

def BestOfNSelectPassing(g: int, n: int, strategy: Any, check: Any, prompt: str, maxSteps: int):
    # Return a CSD.SelectResult.Select(found, s)
    import CSD as CSD_mod
    for _ in range(int(n)):
        s = LLM__Generate(prompt, maxSteps, None)
        if ParseOk(g, s):
            return CSD_mod.SelectResult_Select(True, s)
    return CSD_mod.SelectResult_Select(False, prompt)

# ---- Hook wiring into generated modules --------------------------------
import Decoding as Decoding_mod
import CSD as CSD_mod

# Decoding externs used in our current model are minimal; provide safe defaults.
# These are only needed if you later implement token-level constrained decoding.
def Vocabulary():
    return set()
def Parser__ValidPrefix(prefix):  # pragma: no cover
    return True
def Parser__IsComplete(prefix):  # pragma: no cover
    return False
def Parser__AllowedNext(prefix):  # pragma: no cover
    return set()
def GetLogits(prefix):  # pragma: no cover
    return {}
def MaskLogits(prefix, logits):  # pragma: no cover
    return logits
def Accept(prefix, tok):  # pragma: no cover
    return True
def SampleWithStrategy(logits, candidates, strategy):  # pragma: no cover
    # deterministic pick
    return next(iter(candidates))

# Wire externs
Decoding_mod.default__.Vocabulary = staticmethod(Vocabulary)
Decoding_mod.default__.Parser__ValidPrefix = staticmethod(Parser__ValidPrefix)
Decoding_mod.default__.Parser__IsComplete = staticmethod(Parser__IsComplete)
Decoding_mod.default__.Parser__AllowedNext = staticmethod(Parser__AllowedNext)
Decoding_mod.default__.GetLogits = staticmethod(GetLogits)
Decoding_mod.default__.MaskLogits = staticmethod(MaskLogits)
Decoding_mod.default__.Accept = staticmethod(Accept)
Decoding_mod.default__.SampleWithStrategy = staticmethod(SampleWithStrategy)

CSD_mod.default__.LLM__Generate = staticmethod(LLM__Generate)
CSD_mod.default__.ParseOk = staticmethod(ParseOk)
CSD_mod.default__.ParseOkPrefix = staticmethod(ParseOkPrefix)
CSD_mod.default__.ConstrainedGenerate = staticmethod(ConstrainedGenerate)
CSD_mod.default__.ConstrainedCompleteFromPrefix = staticmethod(ConstrainedCompleteFromPrefix)
CSD_mod.default__.SemanticOk = staticmethod(SemanticOk)
CSD_mod.default__.RepairTransform = staticmethod(RepairTransform)
CSD_mod.default__.ConstrainedSearchGenerate = staticmethod(ConstrainedSearchGenerate)
CSD_mod.default__.BestOfNSelectPassing = staticmethod(BestOfNSelectPassing)
