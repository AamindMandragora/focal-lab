"""Decisive test of the int() handling in the LIVE GSM equivalence scorer.
Constructs an Evaluator and calls _gsm_symbolic_equivalence directly on:
  (A) an int()-gold vs a CLEARLY WRONG no-int model answer  -> should be False; bug predicts True
  (B) an int()-gold vs the EXACT correct int() answer        -> should be True;  bug predicts False
  (C) a non-int gold vs correct/wrong                         -> sanity (should work)
Also counts how many of the 49 seed123 eval golds contain 'int('.
Read-only."""
import sys, json, re
sys.path.insert(0, "/home/aadivyar/csd-generation")
from synthesis.evaluate.evaluator import Evaluator

# Build a minimal evaluator instance without loading models.
ev = Evaluator.__new__(Evaluator)  # bypass __init__ (no GPU/model needed for pure scoring)

vt = {"k": "int", "x": "int", "y": "int", "d": "int", "t": "int"}

def show(label, model, gold, vtypes):
    try:
        r = ev._gsm_symbolic_equivalence(model, gold, vtypes)
    except Exception as e:
        r = f"EXC {e!r}"
    print(f"{label}: model={model!r} gold={gold!r} -> {r}")

print("=== A: int()-gold vs WRONG no-int answer (expect False; bug=>True) ===")
show("A1", "1+1", "int((k*y)/(x*12)*100)", vt)
show("A2", "k+x+y", "int((k*y)/(x*12)*100)", vt)

print("=== B: int()-gold vs EXACT correct answer (expect True; bug=>False) ===")
show("B1", "int((k*y)/(x*12)*100)", "int((k*y)/(x*12)*100)", vt)

print("=== C: non-int gold sanity ===")
show("C1_correct", "y//d*t", "y//d*t", vt)
show("C2_wrong", "y+d+t", "y//d*t", vt)

print("=== Count int()-golds among 49 seed123 eval examples ===")
from synthesis.evaluate.benchmarks.gsm_symbolic.dataset import load_gsm_from_crane_folder
spec = json.load(open("environment/benchmark_splits/gsm_symbolic_crane_proportional_49x49_seed123.json"))
exs = load_gsm_from_crane_folder(crane_dir=spec["crane_dir"], indices=sorted(spec["eval_indices"]))
n_int = sum(1 for e in exs if "int(" in str(e.get("answer_parsed", "")))
n_floordiv = sum(1 for e in exs if "//" in str(e.get("answer_parsed", "")))
print(f"int()-golds: {n_int}/49   //-golds: {n_floordiv}/49")
print("int()-gold expressions:")
for e in exs:
    ap = str(e.get("answer_parsed", ""))
    if "int(" in ap:
        print("   ", ap)
