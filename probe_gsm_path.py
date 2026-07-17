"""Probe: load the seed123 GSM EVAL split exactly as the evaluator does, then
determine per example whether is_correct() takes the SYMBOLIC path
(variable_types AND answer_parsed both present) or the NUMERIC fallback.
Read-only."""
import sys, json, ast
sys.path.insert(0, "/home/aadivyar/csd-generation")
from synthesis.evaluate.benchmarks.gsm_symbolic.dataset import load_gsm_from_crane_folder

SPLIT = "environment/benchmark_splits/gsm_symbolic_crane_proportional_49x49_seed123.json"
spec = json.load(open(SPLIT))
crane_dir = spec.get("crane_dir")
eval_idx = spec.get("eval_indices")
print("crane_dir:", crane_dir)
print("n eval_indices:", len(eval_idx))

examples = load_gsm_from_crane_folder(crane_dir=crane_dir, indices=sorted(eval_idx))

sym = num = 0
no_vt = no_ap = 0
samples = []
for ex in examples:
    vt = ex.get("variable_types", {})
    if isinstance(vt, str):
        try: vt = ast.literal_eval(vt)
        except Exception: vt = {}
    ap = ex.get("answer_parsed")
    if not vt: no_vt += 1
    if not ap: no_ap += 1
    if bool(vt) and bool(ap):
        sym += 1
    else:
        num += 1
    if len(samples) < 3:
        samples.append({
            "variable_types": dict(list(vt.items())[:5]) if isinstance(vt, dict) else vt,
            "answer_parsed": ap,
            "answer_field_head": str(ex.get("answer",""))[:140],
        })

print("=== PATH COUNTS (eval N=%d) ===" % len(examples))
print("SYMBOLIC_path :", sym)
print("NUMERIC_fallback:", num)
print("missing variable_types:", no_vt, " missing answer_parsed:", no_ap)
print("=== SAMPLES ===")
for s in samples:
    print(json.dumps(s, indent=2, default=str))
