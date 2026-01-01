#!/usr/bin/env python3
"""
Synthesize a Dafny `CSD.Program` using Qwen (served via vLLM OpenAI-compatible API),
inject it into `proofs/generated/GeneratedCSD.dfy`, then:
  - parse-check (resolve)
  - verify (required)
  - translate to Python (overwrite-in-place)
  - run a GSM-symbolic smoke test

On any failure, feed the diagnostic back to Qwen and retry up to --max-iters.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

RE_SNIPPET_BLOCK = re.compile(
    r"(?s)//\s*QWEN_SNIPPET_START\s*\n.*?\n\s*//\s*QWEN_SNIPPET_END"
)


def _run(cmd: List[str], cwd: str) -> Tuple[int, str, str]:
    p = subprocess.run(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return p.returncode, p.stdout, p.stderr


def _truncate(s: str, max_chars: int = 8000) -> str:
    s = s.strip()
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 200] + "\n\n... (truncated) ...\n"


def _http_post_json(url: str, payload: Dict[str, Any], headers: Dict[str, str]) -> Dict[str, Any]:
    # Standard library only (avoid adding deps).
    import urllib.request

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body)


def call_openai_compat_chat(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
    max_tokens: int = 600,
) -> str:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    data = _http_post_json(url, payload, headers)
    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        raise RuntimeError(f"Unexpected response from {url}: {_truncate(json.dumps(data, indent=2), 3000)}")


def extract_single_dafny_expr(text: str) -> str:
    """
    Accept either raw text or fenced code. Return a single-line expression.
    """
    t = text.strip()
    # Fenced blocks
    fence = re.search(r"```(?:dafny)?\s*([\s\S]*?)```", t, flags=re.IGNORECASE)
    if fence:
        t = fence.group(1).strip()

    # Remove leading/trailing commentary lines.
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    # If user pasted multiple lines, join with spaces to form a single expr.
    expr = " ".join(lines)

    # Basic hygiene: must not contain module/function definitions.
    forbidden = ["module ", "function ", "method ", "lemma ", "class ", "datatype "]
    low = expr.lower()
    if any(tok in low for tok in forbidden):
        raise ValueError("Qwen output contains forbidden definitions; expected a single expression.")
    return expr


def write_snippet(template_path: str, expr: str) -> None:
    with open(template_path, "r", encoding="utf-8") as f:
        src = f.read()

    replacement = "// QWEN_SNIPPET_START\n" + expr + "\n// QWEN_SNIPPET_END"
    if not RE_SNIPPET_BLOCK.search(src):
        raise RuntimeError("Template missing QWEN_SNIPPET_START/END markers.")
    out = RE_SNIPPET_BLOCK.sub(replacement, src, count=1)

    with open(template_path, "w", encoding="utf-8") as f:
        f.write(out)


def compiled_out_base(compiled_out_dir: str) -> str:
    d = compiled_out_dir.rstrip("/").rstrip("\\")
    if d.endswith("-py"):
        return d[: -3]
    return d


def ensure_runtime_module(compiled_py_dir: str) -> None:
    """
    Overwrite the generated module_.py with extern implementations and hook wiring.
    Assumes compilation was done *without* --python-module-name (so generated files import `module_`).
    """
    os.makedirs(compiled_py_dir, exist_ok=True)
    module_path = os.path.join(compiled_py_dir, "module_.py")
    content = textwrap.dedent(
        """
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
            \"\"\"Load ConstrainedDecoding-py/ConstrainedDecoding.py without clobbering our module_.\"\"\"
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
            \"\"\"CRANE-style constrained backend (Option A): reuse ConstrainedDecoding-py.\"\"\"
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
        """
    ).lstrip()
    with open(module_path, "w", encoding="utf-8") as f:
        f.write(content)


def load_gsm_items(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass
    items: List[Dict[str, Any]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        items.append(json.loads(line))
    return items


def gsm_prompt(question: str) -> str:
    return textwrap.dedent(
        f"""
        You are a math reasoning assistant. For the following word problem, produce step-by-step reasoning in normal text, and wrap each individual calculation or final expression in double angle brackets << >>.

        Problem:
        {question}

        Return your answer as a single expression wrapped in << >>.
        """
    ).strip() + "\n"


def smoke_test(compiled_py_dir: str, dataset_path: str, smoke_n: int, max_steps: int) -> None:
    sys.path.insert(0, compiled_py_dir)
    import importlib

    module_ = importlib.import_module("module_")
    Generated = importlib.import_module("GeneratedCSD")
    CSD_mod = importlib.import_module("CSD")

    items = load_gsm_items(dataset_path)[:smoke_n]
    program = Generated.default__.GeneratedProgram()
    for obj in items:
        q = obj.get("question", "")
        prompt = gsm_prompt(q)
        module_.reset_metrics()
        out = CSD_mod.default__.Run(program, prompt, max_steps)
        if not module_.ParseOk(0, out):
            raise RuntimeError("Smoke test produced output with no parseable <<...>> window.")


@dataclass
class Args:
    repo_root: str
    template_path: str
    out_dir: str
    compiled_out_dir: str
    max_iters: int
    smoke_n: int
    max_steps: int
    dataset_path: str
    qwen_base_url: str
    qwen_api_key: str
    qwen_model: str


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=os.getcwd())
    ap.add_argument("--template-path", default="proofs/generated/GeneratedCSD.dfy")
    ap.add_argument("--out-dir", default="proofs/generated/")
    ap.add_argument("--compiled-out-dir", default="GeneratedCSD-py/")
    ap.add_argument("--max-iters", type=int, default=10)
    ap.add_argument("--smoke-n", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--dataset-path", default="./datasets/ml-gsm-symbolic/generated_data/GSM_symbolic.jsonl")
    ap.add_argument("--qwen-base-url", default=os.environ.get("QWEN_BASE_URL", ""))
    ap.add_argument("--qwen-api-key", default=os.environ.get("QWEN_API_KEY", ""))
    ap.add_argument("--qwen-model", default=os.environ.get("QWEN_MODEL", ""))
    args_ns = ap.parse_args()

    args = Args(
        repo_root=os.path.abspath(args_ns.repo_root),
        template_path=os.path.abspath(os.path.join(args_ns.repo_root, args_ns.template_path)),
        out_dir=os.path.abspath(os.path.join(args_ns.repo_root, args_ns.out_dir)),
        compiled_out_dir=os.path.abspath(os.path.join(args_ns.repo_root, args_ns.compiled_out_dir)),
        max_iters=args_ns.max_iters,
        smoke_n=args_ns.smoke_n,
        max_steps=args_ns.max_steps,
        dataset_path=os.path.abspath(os.path.join(args_ns.repo_root, args_ns.dataset_path)),
        qwen_base_url=args_ns.qwen_base_url,
        qwen_api_key=args_ns.qwen_api_key,
        qwen_model=args_ns.qwen_model,
    )

    if not args.qwen_base_url or not args.qwen_model:
        raise SystemExit("Missing --qwen-base-url/--qwen-model (or env QWEN_BASE_URL/QWEN_MODEL).")

    dafny = os.path.join(args.repo_root, ".tools", "dafny", "dafny", "dafny")
    csd_file = os.path.join(args.repo_root, "proofs", "CSD.dfy")
    gen_file = args.template_path

    system_prompt = "You generate Dafny expressions for CSD.Program. Output ONLY the expression."
    user_prompt = textwrap.dedent(
        """
        Produce a single Dafny expression of type `CSD.Program` using only these constructors:
          - Program: TryThenElse, TryK, BestOfNThenElse, CompleteIfPrefixOkElse, ReturnParsed
          - Attempt: Unconstrained, Constrained, Repair, ConstrainedSearch
          - Policy: Base, MaskWithSynCode, MaskWithCRANE, WithRejection, Fallback, Intersect
        Forbidden:
          - defining new modules/functions/lemmas
          - Policy.Union
        Requirements:
          - The program must be grammar-consistent (use the same g at every node).
          - Ensure the chain ends in ReturnParsed(g, policy).
        Use g = 0.
        """
    ).strip()

    last_diag: Optional[str] = None
    for i in range(args.max_iters):
        messages = [{"role": "system", "content": system_prompt}]
        if last_diag:
            messages.append({"role": "user", "content": f"Previous error:\n{last_diag}"})
        messages.append({"role": "user", "content": user_prompt})

        raw = call_openai_compat_chat(
            base_url=args.qwen_base_url,
            api_key=args.qwen_api_key,
            model=args.qwen_model,
            messages=messages,
        )
        expr = extract_single_dafny_expr(raw)
        write_snippet(gen_file, expr)

        # 1) Parse-check / resolve
        code, _, err = _run([dafny, "resolve", csd_file, gen_file], cwd=args.repo_root)
        if code != 0:
            last_diag = _truncate(err)
            continue

        # 2) Verify (required)
        code, _, err = _run([dafny, "verify", csd_file, gen_file, "--verification-time-limit", "20", "--cores", "2"], cwd=args.repo_root)
        if code != 0:
            last_diag = _truncate(err)
            continue

        # 3) Translate to Python (overwrite-in-place)
        out_base = compiled_out_base(args.compiled_out_dir)
        out_dir = out_base + "-py"
        if os.path.isdir(out_dir):
            shutil.rmtree(out_dir)
        code, _, err = _run(
            [dafny, "translate", "py", csd_file, gen_file, "-o", out_base, "--include-runtime", "--no-verify"],
            cwd=args.repo_root,
        )
        if code != 0:
            last_diag = _truncate(err)
            continue

        ensure_runtime_module(out_dir)

        # 4) Smoke test
        try:
            smoke_test(out_dir, args.dataset_path, args.smoke_n, args.max_steps)
        except Exception as e:
            last_diag = _truncate(str(e))
            continue

        print(f"SUCCESS after {i+1} iteration(s).")
        print(f"Generated expr: {expr}")
        return

    raise SystemExit(f"Failed to synthesize a working program after {args.max_iters} iterations. Last error:\n{last_diag}")


if __name__ == "__main__":
    main()


