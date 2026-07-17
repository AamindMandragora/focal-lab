"""
Official Spider execution-grader rescore of the h78 Spider-9B held-out run
(strategy h19, "aliasclean"), N=300 seed334 test split.

WHY: h78_reeval.json reports accuracy=0.74 computed by an unknown scorer. The
stored `answers[k].generated_answer` are RAW model outputs (alias-intact, with a
trailing `>>`). To settle both fairness and the true number, we extract the SQL
span the same way the pipeline does (strip a leading `SQL:`/`<<`, drop everything
from the first `>>`) and grade the RAW predictions with the SAME execution-based
`evaluate` used for the IterGen baseline. Whatever the official grader says IS the
fair number, regardless of what the 0.74-scorer did.

Predictions are aligned to sorted(test_indices) (verified: 299/300 questions match
dev.json at those indices). Gold = gold_example.txt subset at the same indices.

Run on focal:  python rescore_h78_spider9b_official.py
"""
import sys, os, json, tempfile, importlib.util
from pathlib import Path

REPO = "/home/aadivyar/csd-generation"
H78 = f"{REPO}/outputs/generated/h78_spider9b_h19_aliasclean_heldout_20260629/h78_reeval.json"
SPLIT_FILE = f"{REPO}/environment/benchmark_splits/spider_dev_proportional_300x300_seed334.json"
EVAL_DIR = Path(f"{REPO}/synthesis/evaluate/syncode/syncode/utils/sql_spider_eval")
GOLD_FILE = EVAL_DIR / "evaluation_examples" / "gold_example.txt"
TABLES = EVAL_DIR / "evaluation_examples" / "examples" / "tables.json"
DATABASES = EVAL_DIR / "databases"
DEV = "/home/aadivyar/spider_data/spider_data/dev.json"
BAR = 201  # IterGen 9B test-300 = 201/300 (67.0%); win requires strictly more


def load_evaluate():
    p = EVAL_DIR / "evaluation.py"
    sys.path.insert(0, str(p.parent))
    spec = importlib.util.spec_from_file_location("_h78_spider_eval", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.evaluate


def extract_span(raw: str) -> str:
    """Same span extraction the pipeline applies: drop a leading 'SQL:'/'<<',
    keep only text before the first '>>'. No alias touching."""
    s = raw.strip()
    if s.upper().startswith("SQL:"):
        s = s[4:].strip()
    if s.startswith("<<"):
        s = s[2:]
    if ">>" in s:
        s = s.split(">>", 1)[0]
    return s.strip()


def main():
    h = json.load(open(H78))
    ans = h["answers"]
    print(f"h78 stored: accuracy={h['accuracy']} syntax_rate={h['syntax_rate']}  answers={len(ans)}")

    split = json.load(open(SPLIT_FILE))
    ti = sorted(split["test_indices"])
    dev = json.load(open(DEV))
    gold_all = [l.rstrip("\n") for l in open(GOLD_FILE) if l.strip()]

    # Re-verify alignment before trusting it
    mismatch = sum(1 for k, idx in enumerate(ti)
                   if k < len(ans) and ans[k]["question"].strip() != dev[idx]["question"].strip())
    print(f"alignment check vs sorted(test_indices): mismatch={mismatch}/{len(ti)}")

    preds = [extract_span(a["generated_answer"]) for a in ans]
    gold_subset = [gold_all[i] for i in ti]
    assert len(preds) == len(gold_subset) == 300, (len(preds), len(gold_subset))
    print("sample extracted pred[0]:", repr(preds[0]))
    print("sample gold[0]:", repr(gold_subset[0]))

    with tempfile.TemporaryDirectory() as td:
        pf = os.path.join(td, "predict.txt")
        gf = os.path.join(td, "gold.txt")
        open(pf, "w").write("\n".join(p.replace("\n", " ") for p in preds) + "\n")
        open(gf, "w").write("\n".join(gold_subset) + "\n")
        samples = [{"task_id": i, "completion": preds[i]} for i in range(len(preds))]
        evaluate = load_evaluate()
        scores, err = evaluate(pf, gf, str(DATABASES), etype="all",
                               table=str(TABLES), result_jsonl=samples)

    exec_acc = scores["all"]["exec"]
    n = scores["all"].get("count", len(preds))
    correct = round(exec_acc * n)
    print("\n===== OFFICIAL SPIDER EXECUTION GRADER (h78 raw preds) =====")
    print(f"exec accuracy: {exec_acc:.4f}  ({correct}/{n})")
    print(f"IterGen-9B bar: {BAR}/300 (67.0%)  ->  win needs > {BAR}")
    print("WIN" if correct > BAR else f"NOT A WIN vs bar ({correct} <= {BAR})")
    print("by difficulty:", {k: (round(v.get('exec',0),3), v.get('count')) for k,v in scores.items() if k!='all'})
    print("error types:", dict(err))


if __name__ == "__main__":
    main()
