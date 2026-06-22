"""Read-only diagnostic: WHY does syntax_rate fall below 100% even for faithful CRANE?

For each result file we:
  1. extract every << ... >> span's inner content (same regex as evaluator.py),
  2. try to parse it as a COMPLETE parse of grammars/gsm.lark (start=expr), the
     exact check evaluator._check_syntax_validity uses,
  3. bucket the failures by reason so we can see the real cause.
No files are modified.
"""
import json, re, sys
from pathlib import Path
from collections import Counter

from lark import Lark
from lark.exceptions import LarkError

REPO = Path("/home/aadivyar/csd-generation")
_cands = [REPO / "synthesis" / "evaluate" / "grammars" / "gsm.lark",
          REPO / "grammars" / "gsm.lark"]
GRAMMAR_PATH = next(p for p in _cands if p.exists())
print(f"### grammar: {GRAMMAR_PATH}")
GRAMMAR = GRAMMAR_PATH.read_text()
PARSER = Lark(GRAMMAR, start="start", parser="lalr")
SPAN_RE = re.compile(r"<<\s*([^<>]+?)\s*>>")

def classify(span: str):
    try:
        PARSER.parse(span.strip())
        return "OK"
    except LarkError as e:
        if "=" in span:
            return "has '=' (canonical <<calc=result>>)"
        # incomplete trailing operator e.g. "5 +"
        if re.search(r"[-+*/%]\s*$", span.strip()):
            return "incomplete (trailing operator)"
        if any(c.isalpha() for c in span.replace("int", "")):
            return "contains words/letters"
        return f"other LarkError"

def scan_outputs(label, full_outputs):
    span_total = 0
    span_bad = 0
    reasons = Counter()
    examples_with_spans = 0
    examples_no_spans = 0
    bad_samples = []
    for out in full_outputs:
        spans = SPAN_RE.findall(out or "")
        if not spans:
            examples_no_spans += 1
            continue
        examples_with_spans += 1
        for s in spans:
            span_total += 1
            r = classify(s)
            if r != "OK":
                span_bad += 1
                reasons[r] += 1
                if len(bad_samples) < 12:
                    bad_samples.append((s, r))
    print(f"\n===== {label} =====")
    print(f"examples with >=1 span : {examples_with_spans}")
    print(f"examples with NO span  : {examples_no_spans}  (excluded from syntax_rate denominator)")
    print(f"total spans            : {span_total}")
    print(f"bad spans              : {span_bad}")
    if span_total:
        print(f"segment syntax_rate    : {(span_total - span_bad)/span_total:.1%}")
    print("failure reasons:")
    for r, c in reasons.most_common():
        print(f"   {c:4d}  {r}")
    print("sample bad spans:")
    for s, r in bad_samples:
        print(f"   [{r}]  <<{s}>>")

def load_full_outputs(path: Path):
    data = json.loads(path.read_text())
    # try common shapes
    for key in ("sample_outputs", "answers", "outputs", "examples"):
        if isinstance(data, dict) and key in data and isinstance(data[key], list):
            items = data[key]
            outs = []
            for it in items:
                if isinstance(it, dict):
                    outs.append(it.get("full_output") or it.get("output") or it.get("generation") or it.get("text") or "")
                elif isinstance(it, str):
                    outs.append(it)
            # if everything empty, report keys for debugging
            if any(outs):
                return outs, key, (items[0].keys() if items and isinstance(items[0], dict) else None)
            return outs, key, (items[0].keys() if items and isinstance(items[0], dict) else None)
    return [], None, (data.keys() if isinstance(data, dict) else None)

if __name__ == "__main__":
    for path in sys.argv[1:]:
        p = Path(path)
        if not p.exists():
            print(f"\n(missing) {path}")
            continue
        outs, key, keys = load_full_outputs(p)
        print(f"\n### file: {path}")
        print(f"### list key used: {key}; per-item keys: {list(keys) if keys else None}")
        scan_outputs(p.name, outs)
