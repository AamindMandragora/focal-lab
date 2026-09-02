"""Disposable: compare predicted SQL across the diversity-probe runs.

Inputs: the 6 success_report.json files under outputs/generated/sc_diversity_probe/.
Outputs (stdout): per dev-index, the control (T=0) SQL and the 5 T=0.7 SQLs, plus
  - DISTINCT count among the 5 samples (the diversity signal)
  - whether control == stored argmax (safety check)
  - whether any of the 5 samples matches the GOLD by execution (recovery preview)
Algorithm: load each report's sample_outputs (keyed by question), align to the
probe reference (dev_index -> gold/argmax/question), tabulate.
"""
import json, sys
sys.path.insert(0, "/home/aadivyar/csd-generation")
from synthesis.evaluate.benchmarks.sql_spider.executor import prediction_matches_gold
from synthesis.evaluate.benchmarks.sql_spider.dataset import load_spider

BASE = "outputs/generated/sc_diversity_probe"
RUNS = ["sc_probe_ctrl_t00", "sc_probe_s1_t07", "sc_probe_s2_t07",
        "sc_probe_s3_t07", "sc_probe_s4_t07", "sc_probe_s5_t07"]
ref = json.load(open("/tmp/sc_probe_reference.json"))  # dev_index(str) -> {question,gold,argmax}

def load_run(name):
    import glob
    paths = glob.glob(f"{BASE}/{name}/**/results/success_report.json", recursive=True)
    if not paths:
        return None
    d = json.load(open(sorted(paths)[-1]))
    # map question -> actual SQL
    out = {}
    for e in d["sample_outputs"]:
        out[e.get("question", "").strip()] = (e.get("actual") or "").strip()
    return out

runs = {name: load_run(name) for name in RUNS}
for name, r in runs.items():
    print(f"{name}: {'MISSING' if r is None else str(len(r))+' examples'}")
print()

# db_info per dev index for execution check
dev_indices = [int(k) for k in ref.keys()]
rows = load_spider(source="auto", indices=dev_indices)
dbinfo = {di: row for di, row in zip(dev_indices, rows)}

def matches_gold(sql, row):
    try:
        return bool(prediction_matches_gold(sql, row))
    except Exception as ex:
        return f"ERR:{type(ex).__name__}"

total_distinct = 0
any_recovered = 0
for k, info in ref.items():
    q = info["question"].strip()
    gold = info["gold"]
    row = dbinfo[int(k)]
    ctrl = runs["sc_probe_ctrl_t00"].get(q, "<none>") if runs["sc_probe_ctrl_t00"] else "<none>"
    samples = [runs[n].get(q, "<none>") for n in RUNS[1:] if runs[n]]
    distinct = len(set(samples))
    total_distinct += distinct
    ctrl_eq_argmax = (ctrl.strip() == info["argmax"].strip())
    # sanity: gold vs gold should be True; ctrl(argmax) vs gold should be False (these are failures)
    gold_self = matches_gold(row.get("query", gold), row)
    ctrl_correct = matches_gold(ctrl, row)
    recovered = []
    for n, s in zip(RUNS[1:], samples):
        if s and s != "<none>" and matches_gold(s, row) is True:
            recovered.append(n)
    if recovered:
        any_recovered += 1
    print(f"dev={k}  distinct={distinct}/5  ctrl==argmax:{ctrl_eq_argmax}  gold_self:{gold_self}  ctrl_correct:{ctrl_correct}  recovered:{recovered or '-'}")
    print(f"   Q: {q[:70]}")
    print(f"   gold : {gold[:90]}")
    print(f"   ctrl : {ctrl[:90]}")
    for i, s in enumerate(samples, 1):
        print(f"   s{i}   : {s[:90]}")
    print()

print(f"=== SUMMARY ===")
print(f"avg distinct samples per example: {total_distinct/len(ref):.2f}/5")
print(f"examples where >=1 of the 5 samples recovers GOLD: {any_recovered}/{len(ref)}")
