"""
Official Spider grader scoring for our CSD pipeline's Spider-7B clean-win
held-out eval (test split, seed334, N=300).

Reads sample_outputs from the pipeline's success_report.json, extracts the
predicted SQL (the 'actual' field — the extracted span content), aligns it
against the subset gold file for the test split, and runs the same
execution-based spider_evaluate used for the IterGen/CARS baselines.

Run from /home/aadivyar/csd-generation:
    python rescore_our_spider7b_heldout_20260612.py

Or override result dir:
    RESULT_DIR=<path> python rescore_our_spider7b_heldout_20260612.py
"""

import sys
import os
import json
import tempfile
import importlib.util
from pathlib import Path

# ── Config ───────────────────────────────────────────────────────────────────
SPLIT_FILE = os.environ.get(
    "SPLIT_FILE",
    "/home/aadivyar/csd-generation/environment/benchmark_splits"
    "/spider_dev_proportional_300x300_seed334.json",
)
RESULT_DIR = os.environ.get("RESULT_DIR", "")  # set via env or auto-detected below

EVAL_DIR = (
    Path("/home/aadivyar/csd-generation/synthesis/evaluate/syncode/syncode"
         "/utils/sql_spider_eval")
)
GOLD_FILE = EVAL_DIR / "evaluation_examples" / "gold_example.txt"
TABLES    = EVAL_DIR / "evaluation_examples" / "examples" / "tables.json"
DATABASES = EVAL_DIR / "databases"


def _load_spider_evaluate():
    eval_path = EVAL_DIR / "evaluation.py"
    if not eval_path.exists():
        raise FileNotFoundError(f"Spider evaluator not found: {eval_path}")
    eval_dir = str(eval_path.parent)
    if eval_dir not in sys.path:
        sys.path.insert(0, eval_dir)
    module_name = "_vas_sql_spider_eval"
    module = sys.modules.get(module_name)
    if module is None:
        spec = importlib.util.spec_from_file_location(module_name, eval_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return module.evaluate


def find_success_report(result_dir: str) -> Path:
    """Walk result_dir to find the success_report.json."""
    for root, _dirs, files in os.walk(result_dir):
        if "success_report.json" in files:
            return Path(root) / "success_report.json"
    raise FileNotFoundError(f"No success_report.json under {result_dir}")


def auto_detect_result_dir() -> str:
    """Find the most recently modified spider7b_cleanwin_heldout output dir."""
    base = Path("/home/aadivyar/csd-generation/outputs/generated")
    candidates = sorted(
        base.glob("spider7b_cleanwin_heldout_20260612*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            "No spider7b_cleanwin_heldout_20260612* dir found under "
            f"{base}"
        )
    return str(candidates[0])


def main():
    result_dir = RESULT_DIR or auto_detect_result_dir()
    print(f"result_dir: {result_dir}")

    report_path = find_success_report(result_dir)
    print(f"success_report: {report_path}")

    with open(report_path) as f:
        report = json.load(f)

    sample_outputs = report.get("sample_outputs", [])
    if not sample_outputs:
        # Sometimes sample_outputs lives directly in evaluation_result
        sample_outputs = report.get("evaluation_result", {}).get("sample_outputs", [])
    if not sample_outputs:
        raise ValueError("No sample_outputs found in success_report.json")

    print(f"sample_outputs count: {len(sample_outputs)}")

    # Report pipeline scorer numbers
    ev = report.get("evaluation_result", {})
    print(f"\nPipeline scorer: accuracy={ev.get('accuracy'):.4f} "
          f"({ev.get('num_correct')}/{ev.get('num_examples')})  "
          f"syntax_rate={ev.get('syntax_rate'):.4f}")

    # Load seed334 test indices
    with open(SPLIT_FILE) as f:
        split = json.load(f)
    test_indices = sorted(split["test_indices"])
    print(f"\ntest_indices: {len(test_indices)} entries, "
          f"range {min(test_indices)}..{max(test_indices)}")

    # Load full gold file
    with open(GOLD_FILE) as f:
        all_gold_lines = [l.rstrip("\n") for l in f if l.strip()]
    print(f"Full gold file: {len(all_gold_lines)} lines")

    # Build subset gold (one line per test_index, in sorted order)
    subset_gold_lines = [all_gold_lines[i] for i in test_indices if i < len(all_gold_lines)]
    print(f"Subset gold: {len(subset_gold_lines)} lines")

    if len(sample_outputs) != len(subset_gold_lines):
        print(f"WARNING: sample_outputs({len(sample_outputs)}) != "
              f"subset_gold({len(subset_gold_lines)})")

    # Extract predictions — 'actual' is the raw SQL span content extracted by our pipeline
    predictions = [s.get("actual", "") or "" for s in sample_outputs]
    print(f"Predictions extracted: {len(predictions)}")
    print(f"Empty predictions: {sum(1 for p in predictions if not p.strip())}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        predict_file = os.path.join(tmp_dir, "predict.txt")
        gold_subset_file = os.path.join(tmp_dir, "gold_subset.txt")

        with open(predict_file, "w") as f:
            for p in predictions:
                f.write(p.strip() + "\n")

        with open(gold_subset_file, "w") as f:
            for g in subset_gold_lines:
                f.write(g + "\n")

        samples = [
            {"task_id": i, "completion": predictions[i]}
            for i in range(len(predictions))
        ]

        evaluate = _load_spider_evaluate()
        scores, error_types = evaluate(
            predict_file, gold_subset_file, str(DATABASES),
            etype="all", table=str(TABLES), result_jsonl=samples,
        )

    print("\n===== OFFICIAL SPIDER GRADER RESULT =====")
    exec_acc = scores["all"]["exec"]
    count_all = scores["all"].get("count", len(predictions))
    correct = round(exec_acc * count_all)
    print(f"Execution accuracy (all): {exec_acc:.4f}  ({correct}/{count_all})")
    print(f"Win threshold: 198/300 = 0.6600")
    if correct >= 198:
        print(f"RESULT: CLEAN WIN  ({correct}/300 >= 198)")
    else:
        print(f"RESULT: DID NOT REACH WIN THRESHOLD  ({correct}/300 < 198)")
    print("\nBy difficulty:")
    for lvl in scores:
        if lvl != "all":
            lvl_exec = scores[lvl].get("exec", 0)
            lvl_count = scores[lvl].get("count", 0)
            print(f"  {lvl}: {lvl_exec:.3f}  (count={lvl_count})")
    print("Error types:", dict(error_types))


if __name__ == "__main__":
    main()
