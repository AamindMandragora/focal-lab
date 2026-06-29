#!/usr/bin/env python3
"""Verify the DEPLOYED focal template in focal's real environment.
Run on focal from a scratch dir that contains the deployed GeneratedCSD.dfy
(as template.dfy) and VerifiedAgentSynthesis.dfy beside it.

Cases: empty (as-is), passthrough (no-progress), CRANE (realistic).
Expect all three to verify (rc 0). Prints RC per case + overall verdict.
"""
import subprocess, sys, shutil, pathlib

HERE = pathlib.Path(__file__).resolve().parent
MARKER = "// QWEN_INSERT_STRATEGY_HERE"
DAFNY = shutil.which("dafny") or "/apps/conda/advayth2/envs/advayth2/bin/dafny"

TEMPLATE = (HERE / "template.dfy").read_text()

PASSTHROUGH = (
    "generated := generatedPrefix;\n"
    "    insideConstrainedOut := insideConstrained;\n"
    "    currentConstrainedOut := currentConstrained;\n"
    "    cost := 0;"
)
CRANE = (
    "generated := helpers.CraneGeneration(lm, parser, prompt, maxSteps, 10, eosToken);\n"
    "    cost := helpers.cost;"
)

def verify(body, tag):
    src = TEMPLATE.replace(MARKER, body)
    f = HERE / f"_focal_{tag}.dfy"
    f.write_text(src)
    p = subprocess.run([DAFNY, "verify", str(f)], capture_output=True, text=True, timeout=300)
    return p.returncode, p.stdout + p.stderr

print(f"dafny: {DAFNY}")
results = {}
for tag, body in (("empty", ""), ("passthrough", PASSTHROUGH), ("crane", CRANE)):
    rc, out = verify(body, tag)
    results[tag] = rc
    print(f"[{tag}] rc={rc} {'VERIFIED' if rc==0 else 'FAILED'}")
    if rc != 0:
        print("\n".join(out.splitlines()[:30]))

ok = all(rc == 0 for rc in results.values())
print(f"OVERALL: {'ALL VERIFY' if ok else 'MISMATCH'}  {results}")
sys.exit(0 if ok else 1)
