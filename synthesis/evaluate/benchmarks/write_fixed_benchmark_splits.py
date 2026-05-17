"""Write tracked fixed train/eval manifests for GSM-Symbolic and Spider."""

from __future__ import annotations

import argparse
from pathlib import Path

from synthesis.evaluate.benchmarks.gsm_symbolic.dataset import write_gsm_proportional_train_eval_split
from synthesis.evaluate.benchmarks.sql_spider.dataset import write_spider_proportional_train_test_split
from synthesis.project_defaults import default_gsm_source_dir

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "environment" / "benchmark_splits"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for split manifest JSON files",
    )
    parser.add_argument("--gsm-train-size", type=int, default=0)
    parser.add_argument("--gsm-eval-size", type=int, default=100)
    parser.add_argument("--spider-train-size", type=int, default=50)
    parser.add_argument("--spider-test-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    gsm_path = output_dir / "gsm_symbolic_crane_proportional.json"
    gsm_split = write_gsm_proportional_train_eval_split(
        gsm_path,
        crane_dir=default_gsm_source_dir(),
        train_size=args.gsm_train_size,
        eval_size=args.gsm_eval_size,
        seed=args.seed,
    )
    print(f"Wrote {gsm_path} (eval={gsm_split['eval_size']}, train={gsm_split['train_size']})")
    print(f"  population={gsm_split.get('population_composition')}")
    print(f"  eval={gsm_split.get('eval_composition')}")

    spider_path = output_dir / "spider_dev_proportional.json"
    spider_split = write_spider_proportional_train_test_split(
        spider_path,
        train_size=args.spider_train_size,
        test_size=args.spider_test_size,
        seed=args.seed,
    )
    print(f"Wrote {spider_path} (test={spider_split['test_size']}, train={spider_split['train_size']})")
    print(f"  population={spider_split.get('population_composition')}")
    print(f"  test={spider_split.get('test_composition')}")


if __name__ == "__main__":
    main()
