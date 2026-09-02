#!/usr/bin/env python
"""RED-first reproduction of the doubled-'>>' exit bug in CloseConstrainedSpan.

WHAT THIS TESTS (Inputs / Outputs / Algorithm):
  Inputs:  the COMPILED VerifiedDecoderAgent.CSDHelpers from a detector build dir,
           plus hand-built Dafny token prefixes whose RENDERED text ends with '>>'
           but whose LAST TOKEN is not the exact string '>>'.
  Output:  pass/fail. Asserts the CORRECT behavior (the span already closed, so
           CloseConstrainedSpan must NOT append a second '>>').
  Algorithm:
    1. Put the build dir on sys.path; import VerifiedDecoderAgent + _dafny.
    2. tok(s)=Dafny string token, pfx(*t)=Dafny prefix seq, render(p)=plain text.
    3. Three span shapes that all END in '>>' as rendered text:
         exact    : [...,'>>']            last token == '>>'   (control)
         split    : [...,'>','>' ]        last token == '>'    (BUG trigger)
         spaced   : [...,' >>']           last token == ' >>'  (BUG trigger)
    4. Call CloseConstrainedSpan(None,None, generated, currentConstrained).
    5. CORRECT == generatedOut equals `generated` (no extra '>>'); the rendered
       result must end with exactly one '>>' (never '>>>>').

On the CURRENT compiled library the split/spaced cases FAIL (they get a 2nd '>>').
After the Dafny rendered-text-suffix fix + recompile they must all PASS.
"""
import sys
from pathlib import Path

BUILD_DIR = sys.argv[1] if len(sys.argv) > 1 else (
    "/home/aadivyar/csd-generation/outputs/generated/"
    "ralph_1p5B_gsm_relaunch_20260529_detector/"
    "ralph_1p5B_gsm_relaunch_20260529_detector_20260529_182754_657382/"
    "python/ralph_1p5B_gsm_relaunch_20260529_detector"
)
sys.path.insert(0, BUILD_DIR)

import _dafny  # noqa: E402
import VerifiedDecoderAgent as VDA  # noqa: E402


def tok(s: str):
    """One Dafny string token."""
    return _dafny.SeqWithoutIsStrInference(list(map(_dafny.CodePoint, s)))


def pfx(*toks):
    """A Dafny prefix = seq of token strings."""
    return _dafny.SeqWithoutIsStrInference([tok(t) for t in toks])


def render(prefix) -> str:
    """Flatten a prefix back to plain text for human-readable asserts."""
    out = []
    for t in prefix:
        out.append("".join(str(c) for c in t))
    return "".join(out)


CASES = {
    "exact  (last token '>>')": ["<<", "5", "+", "5", ">>"],
    "split  (last token '>')":  ["<<", "5", "+", "5", ">", ">"],
    "spaced (last token ' >>')": ["<<", "5", "+", "5", " >>"],
}

failures = []
for label, toks in CASES.items():
    h = VDA.CSDHelpers()
    h.ctor__()
    generated = pfx(*toks)          # model already emitted the closing '>>'
    current = pfx(*toks)            # the span tokens (same here)
    gout, inside, cout = h.CloseConstrainedSpan(None, None, generated, current)
    rendered = render(gout)
    appended_second = (len(gout) > len(generated))
    ok = (not appended_second) and rendered.endswith(">>") and not rendered.endswith(">>>>")
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {label}")
    print(f"          in  = {render(generated)!r}")
    print(f"          out = {rendered!r}  (appended_2nd={appended_second})")
    if not ok:
        failures.append(label)

print()
if failures:
    print(f"RED: {len(failures)} case(s) doubled the close delimiter -> {failures}")
    sys.exit(1)
print("GREEN: all span shapes closed with exactly one '>>'.")
sys.exit(0)
