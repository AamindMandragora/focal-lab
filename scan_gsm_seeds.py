#!/usr/bin/env python3
"""
Scan candidate seeds for the GSM 49x49 split.

For each seed 1-250, generate a 49-train / 49-eval split from the same
population, count how many grammar-incompatible examples land in each half,
and report the most balanced seeds.

A grammar-incompatible example is one whose answer_parsed contains:
  =      (assignment notation the grammar doesn't support)
  round( (function the grammar doesn't support)
  { or } (brace notation explicitly rejected by eval_logic.py)

Run from /home/aadivyar/csd-generation:
  python scan_gsm_seeds.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from synthesis.evaluate.benchmarks.split_utils import stratified_sample_indices
from synthesis.evaluate.benchmarks.gsm_symbolic.dataset import load_gsm_from_crane_folder
from synthesis.project_defaults import default_gsm_source_dir

GSM_DIFFICULTIES = ("easy", "medium", "hard")
TRAIN_SIZE = 49
EVAL_SIZE = 49
SEED_RANGE = range(1, 251)

SPLIT_FILE = Path(
    "environment/benchmark_splits/gsm_symbolic_crane_proportional_49x49_seed123.json"
)


def is_grammar_incompatible(example: dict) -> bool:
    answer_parsed = str(example.get("answer_parsed") or "")
    return "=" in answer_parsed or "round(" in answer_parsed or "{" in answer_parsed or "}" in answer_parsed


def main():
    # Read the existing split to get difficulty_by_index for the population
    with open(SPLIT_FILE) as f:
        existing = json.load(f)

    difficulty_by_index = {int(k): v for k, v in existing["difficulty_by_index"].items()}
    total_examples = existing.get("total_examples", len(difficulty_by_index))
    all_pop_indices = sorted(difficulty_by_index.keys())

    print(f"Population: {len(all_pop_indices)} examples  (total_examples field={total_examples})")
    print(f"Split size: {TRAIN_SIZE} train + {EVAL_SIZE} eval")

    # Check whether the existing split covers the FULL dataset or a subset
    crane_dir = Path(default_gsm_source_dir())
    all_crane_files = sorted(crane_dir.glob("*.json"))
    print(f"CRANE folder has {len(all_crane_files)} files total")

    # If the population in the split covers all files, use difficulty_by_index directly.
    # Otherwise load the full dataset and assign difficulties.
    if len(all_crane_files) == len(all_pop_indices):
        print("Population == full CRANE folder → using existing difficulty_by_index directly")
        full_difficulty_by_index = difficulty_by_index
    else:
        print(f"Population ({len(all_pop_indices)}) < CRANE folder ({len(all_crane_files)}) "
              f"→ loading full dataset to assign difficulties")
        # Load all examples and assign difficulty from crane source config (main=easy,p1=medium,p2=hard)
        all_examples = load_gsm_from_crane_folder(crane_dir)
        full_difficulty_by_index = {}
        for idx, ex in enumerate(all_examples):
            src = str(ex.get("crane_source_file", ""))
            if "main" in src:
                diff = "easy"
            elif "p1" in src:
                diff = "medium"
            else:
                diff = "hard"
            full_difficulty_by_index[idx] = diff

    # Load examples for all population indices to identify grammar-incompatible ones
    pop_indices = sorted(full_difficulty_by_index.keys())
    print(f"\nLoading {len(pop_indices)} examples to check grammar compatibility...")
    examples = load_gsm_from_crane_folder(crane_dir, indices=pop_indices)

    incompatible_set: set[int] = set()
    print("\nGrammar-incompatible examples in population:")
    for i, ex in zip(pop_indices, examples):
        if is_grammar_incompatible(ex):
            incompatible_set.add(i)
            answer = str(ex.get("answer_parsed") or "")[:100]
            print(f"  idx={i:3d} ({full_difficulty_by_index.get(i,'?'):6s}): {answer!r}")

    print(f"\nTotal incompatible in population: {len(incompatible_set)}/{len(pop_indices)}")

    # Show current seed 123 distribution
    cur_train_incompat = sum(1 for i in existing["train_indices"] if i in incompatible_set)
    cur_eval_incompat = sum(1 for i in existing["eval_indices"] if i in incompatible_set)
    print(f"\nCurrent seed 123: train_incompat={cur_train_incompat}, eval_incompat={cur_eval_incompat}")
    print(f"  Train eval ceiling: {(TRAIN_SIZE - cur_train_incompat)/TRAIN_SIZE*100:.1f}%")
    print(f"  Eval ceiling:       {(EVAL_SIZE - cur_eval_incompat)/EVAL_SIZE*100:.1f}%")

    # Scan seeds
    print(f"\nScanning seeds 1–{max(SEED_RANGE)}...")
    results = []
    for seed in SEED_RANGE:
        selected = stratified_sample_indices(
            full_difficulty_by_index,
            split_sizes={"train": TRAIN_SIZE, "eval": EVAL_SIZE},
            difficulties=GSM_DIFFICULTIES,
            seed=seed,
        )
        t_inc = sum(1 for i in selected["train"] if i in incompatible_set)
        e_inc = sum(1 for i in selected["eval"] if i in incompatible_set)
        imbalance = abs(t_inc - e_inc)
        eval_ceiling = (EVAL_SIZE - e_inc) / EVAL_SIZE * 100
        results.append((imbalance, e_inc, t_inc, seed, eval_ceiling))

    results.sort()  # sort by imbalance asc, then eval_incompat asc

    print("\nTop 30 most balanced seeds:")
    print(f"{'Seed':>6}  {'Train incompat':>14}  {'Eval incompat':>13}  {'|diff|':>7}  {'Eval syntax ceiling':>20}")
    for imbalance, e, t, seed, ceil in results[:30]:
        marker = " ← CURRENT" if seed == 123 else ""
        print(f"{seed:>6}  {t:>14}  {e:>13}  {imbalance:>7}  {ceil:>19.1f}%{marker}")

    # Find the seed with fewest eval incompatible (best ceiling), breaking ties by balance
    best_by_ceil = min(results, key=lambda r: (r[1], r[0]))
    print(f"\nBest seed by eval ceiling: seed={best_by_ceil[3]}  "
          f"eval_incompat={best_by_ceil[1]}  train_incompat={best_by_ceil[2]}  "
          f"eval_ceiling={best_by_ceil[4]:.1f}%")
    print(f"Best balanced seed: seed={results[0][3]}  "
          f"eval_incompat={results[0][1]}  train_incompat={results[0][2]}  "
          f"eval_ceiling={results[0][4]:.1f}%")


if __name__ == "__main__":
    main()
