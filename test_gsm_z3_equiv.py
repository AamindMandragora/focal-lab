"""TDD test for the CRANE-faithful z3 GSM equivalence scorer.
Encodes the CORRECT expected behavior. Run on focal with the csd python:
  /apps/conda/aadivyar/envs/csd/bin/python test_gsm_z3_equiv.py
Exit 0 = all pass (GREEN); exit 1 = failures (RED).
"""
import sys
sys.path.insert(0, "/home/aadivyar/csd-generation")
from synthesis.evaluate.evaluator import Evaluator

ev = Evaluator.__new__(Evaluator)  # no model load needed for pure scoring
vt = {"k": "int", "x": "int", "y": "int", "d": "int", "t": "int",
      "a": "int", "b": "int", "n": "int", "frac": "float between 0 and 1"}

CASES = [
    # (label, model_expr, gold_expr, expected_bool)
    # --- int()-gold: junk model must be WRONG (was the false-positive bug) ---
    ("A1_junk_vs_int",      "1+1",                    "int((k*y)/(x*12)*100)", False),
    ("A2_junk_vs_int",      "k+x+y",                  "int((k*y)/(x*12)*100)", False),
    # --- int()-gold: exact correct model must be RIGHT (was the false-negative bug) ---
    ("B1_exact_int",        "int((k*y)/(x*12)*100)",  "int((k*y)/(x*12)*100)", True),
    # --- z3 proves an algebraically-equal-but-textually-different int() form ---
    ("B2_reordered_int",    "int(100*(k*y)/(x*12))",  "int((k*y)/(x*12)*100)", True),
    # --- floor-div golds still work (the part that wasn't broken) ---
    ("C1_floordiv_correct", "y//d*t",                 "y//d*t",                True),
    ("C2_floordiv_wrong",   "y+d+t",                  "y//d*t",                False),
    # --- plain arithmetic equivalence z3 should prove ---
    ("D1_distribute",       "a*(k+y)",                "a*k+a*y",               True),
    ("D2_not_equal",        "a*k+y",                  "a*k+a*y",               False),
    # --- CRANE rejects model answers containing round( and ** ---
    ("E1_round_rejected",   "round(k*y)",             "int(k*y)",              False),
    ("E2_pow_rejected",     "k**2",                   "k*k",                   False),
]

fails = 0
for label, model, gold, want in CASES:
    try:
        got = ev._gsm_symbolic_equivalence(model, gold, vt)
    except Exception as e:
        got = f"EXC {e!r}"
    ok = (got == want)
    if not ok:
        fails += 1
    print(f"[{'PASS' if ok else 'FAIL'}] {label}: model={model!r} gold={gold!r} -> got={got} want={want}")

print(f"\n{'GREEN — all passed' if fails == 0 else f'RED — {fails} failing'} ({len(CASES)-fails}/{len(CASES)})")
sys.exit(0 if fails == 0 else 1)
