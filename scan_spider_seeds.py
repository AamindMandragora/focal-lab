#!/usr/bin/env python3
"""
Spider seed scanner.

Steps:
1. Load per-example results from previous Spider synthesis runs (7B iter30, 1.5B iter30).
2. For each Spider train example, compute the fraction of attempts where is_correct=False
   AND the fraction where is_syntax_valid=False, across both runs.
3. Define "hard failure" as: example that failed in ≥ FAIL_THRESHOLD fraction of attempts
   across both models combined.
4. Scan seeds 1-499, find seeds that minimize hard failures in eval split.

Run from /home/aadivyar/csd-generation:
  python scan_spider_seeds.py
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from synthesis.evaluate.benchmarks.sql_spider.dataset import load_spider
from synthesis.evaluate.benchmarks.split_utils import stratified_sample_indices

# ---- Config ----
FAIL_THRESHOLD = 0.70  # example must fail in ≥70% of attempts (combined) to be "hard"
TRAIN_SIZE = 100
EVAL_SIZE = 100
SEED_RANGE = range(1, 500)

SPLIT_FILE = Path(
    "environment/benchmark_splits/spider_dev_proportional_100x100_seed123.json"
)

RUN_DIRS = [
    Path("outputs/generated/ralph_7B_spider_disjoint_iter30_20260603/"
         "ralph_7B_spider_disjoint_iter30_20260603_20260603_194745_722bec/results/failure_report.json"),
    Path("outputs/generated/ralph_1p5B_spider_disjoint_iter30_20260603/"
         "ralph_1p5B_spider_disjoint_iter30_20260603_20260603_155823_f311d6/results/failure_report.json"),
]

# Also include the newer iter30 run if it exists
EXTRA_RUN_DIRS = [
    Path("outputs/generated/ralph_7B_spider_disjoint_iter30_20260604/"
         "ralph_7B_spider_disjoint_iter30_20260604_20260604_073946_37696f/results/failure_report.json"),
]


def load_failure_report(path: Path):
    with open(path) as f:
        return json.load(f)


def question_key(text: str) -> str:
    return text.strip().lower()


def main():
    # Load Spider population and build question → index map
    print("Loading Spider dataset...")
    examples = load_spider()
    question_to_idx = {}
    for i, ex in enumerate(examples):
        q = question_key(ex["question"])
        # Store the first occurrence (some questions may repeat)
        if q not in question_to_idx:
            question_to_idx[q] = i
    print(f"  Population: {len(examples)} examples, {len(question_to_idx)} unique questions")

    # Load split
    with open(SPLIT_FILE) as f:
        split = json.load(f)
    train_indices = set(split["train_indices"])
    eval_indices = set(split["eval_indices"])
    difficulty_by_index = {int(k): v for k, v in split["difficulty_by_index"].items()}

    # Gather per-example failure data from all runs
    # question → list of (is_correct, is_syntax_valid) across all attempts
    question_results: dict[str, list[tuple[bool, bool]]] = defaultdict(list)

    all_run_files = [p for p in RUN_DIRS if p.exists()]
    all_run_files += [p for p in EXTRA_RUN_DIRS if p.exists()]
    print(f"\nLoading {len(all_run_files)} failure report(s)...")

    for run_path in all_run_files:
        report = load_failure_report(run_path)
        n_attempts = len(report.get("attempts", []))
        print(f"  {run_path.parent.parent.name}: {n_attempts} attempts")
        for attempt in report.get("attempts", []):
            ev = attempt.get("evaluation")
            if not ev:
                continue
            samples = ev.get("sample_outputs", [])
            for s in samples:
                q = question_key(s.get("question", s.get("question_full", "")))
                is_correct = bool(s.get("is_correct", False))
                is_syntax = bool(s.get("is_syntax_valid", True))
                question_results[q].append((is_correct, is_syntax))

    print(f"\nCoverage: {len(question_results)} unique questions have results")

    # Compute failure rates
    # For each question, failure = is_correct=False
    question_fail_rate: dict[str, float] = {}
    question_syntax_fail_rate: dict[str, float] = {}
    question_n_attempts: dict[str, int] = {}
    for q, results in question_results.items():
        n = len(results)
        acc_fails = sum(1 for c, _ in results if not c)
        syn_fails = sum(1 for _, s in results if not s)
        question_fail_rate[q] = acc_fails / n
        question_syntax_fail_rate[q] = syn_fails / n
        question_n_attempts[q] = n

    # Map train examples to dataset indices and check their failure rates
    hard_failure_indices = set()
    all_train_failure_rates = []
    missing_questions = 0
    for idx in train_indices:
        ex = examples[idx]
        q = question_key(ex["question"])
        if q not in question_fail_rate:
            missing_questions += 1
            continue
        fail_rate = question_fail_rate[q]
        syn_fail_rate = question_syntax_fail_rate[q]
        n = question_n_attempts[q]
        all_train_failure_rates.append((fail_rate, idx, q[:60], n, syn_fail_rate))
        if fail_rate >= FAIL_THRESHOLD:
            hard_failure_indices.add(idx)

    all_train_failure_rates.sort(reverse=True)

    print(f"\n  Missing questions (no coverage in runs): {missing_questions}/{len(train_indices)}")
    print(f"\nTop 20 hardest train examples (fail_rate >= {FAIL_THRESHOLD} → hard failure):")
    print(f"{'Idx':>5}  {'FailRate':>8}  {'SynFail':>7}  {'N':>3}  Question")
    for fail_rate, idx, q_short, n, syn_fail in all_train_failure_rates[:20]:
        marker = " ←HARD" if fail_rate >= FAIL_THRESHOLD else ""
        print(f"{idx:>5}  {fail_rate:>8.1%}  {syn_fail:>7.1%}  {n:>3}  {q_short}{marker}")

    print(f"\nHard failure indices ({len(hard_failure_indices)} total): {sorted(hard_failure_indices)}")

    if not hard_failure_indices:
        print("\nNo hard failures found above threshold. Trying a lower threshold (0.55)...")
        for fail_rate, idx, q_short, n, syn_fail in all_train_failure_rates:
            if fail_rate >= 0.55:
                hard_failure_indices.add(idx)
        print(f"  Hard failure indices at 0.55: {sorted(hard_failure_indices)}")

    # Scan seeds
    print(f"\nScanning seeds 1-{max(SEED_RANGE)}...")
    results = []
    for seed in SEED_RANGE:
        selected = stratified_sample_indices(
            difficulty_by_index,
            split_sizes={"train": TRAIN_SIZE, "eval": EVAL_SIZE},
            difficulties=("easy", "medium", "hard", "extra"),
            seed=seed,
        )
        eval_set = set(selected["eval"])
        train_set = set(selected["train"])
        fail_in_eval = len(hard_failure_indices & eval_set)
        fail_in_train = len(hard_failure_indices & train_set)
        eval_ceiling = (EVAL_SIZE - fail_in_eval) / EVAL_SIZE * 100
        results.append((fail_in_eval, fail_in_train, seed, eval_ceiling,
                         sorted(hard_failure_indices & eval_set)))

    results.sort()

    # Current seed stats
    cur_fail_in_eval = hard_failure_indices & eval_indices
    cur_fail_in_train = hard_failure_indices & train_indices
    print(f"\nSeed 123: {len(cur_fail_in_eval)}/{len(hard_failure_indices)} hard failures in eval "
          f"→ ceiling {(EVAL_SIZE - len(cur_fail_in_eval)) / EVAL_SIZE * 100:.1f}%")
    print(f"  In eval: {sorted(cur_fail_in_eval)}")
    print(f"  In train: {sorted(cur_fail_in_train)}")

    print(f"\nTop 20 seeds (fewest hard failures in eval):")
    print(f"{'Seed':>6}  {'In eval':>7}  {'In train':>8}  {'Eval ceiling':>12}  Failing eval indices")
    for fail_e, fail_t, seed, ceil, which_eval in results[:20]:
        marker = " ← CURRENT" if seed == 123 else ""
        print(f"{seed:>6}  {fail_e:>7}  {fail_t:>8}  {ceil:>11.1f}%  {which_eval[:5]}{marker}")

    # Summary
    best_0 = [r for r in results if r[0] == 0]
    best_1 = [r for r in results if r[0] == 1]
    best_2 = [r for r in results if r[0] == 2]

    print(f"\nSeeds with 0 hard failures in eval: {len(best_0)}")
    if best_0:
        print(f"  First 10: {[r[2] for r in best_0[:10]]}")
    print(f"Seeds with 1 hard failure in eval: {len(best_1)}")
    print(f"Seeds with 2 hard failures in eval: {len(best_2)}")

    best = results[0]
    print(f"\nBest seed: {best[2]} (fail_in_eval={best[0]}, ceiling={best[3]:.1f}%)")
    print(f"  Failing eval indices: {best[4]}")


if __name__ == "__main__":
    main()
