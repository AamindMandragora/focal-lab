#!/usr/bin/env python
"""Fast-iteration synthesis loop on the 50-example Spider train subset.

The production entry point (synthesis/run_synthesis.py) is pinned to the
canonical 300x300 seed334 Spider split and takes no split flags. This wrapper
exists so quick experiments can still run on the proportional-hardness
50-example train subset (~6x faster per iteration) without that knob living
in the production script. It is a dev/test tool, not part of the recorded
pipeline — results from it are for iteration speed only, never for
results_matrix.md rows.

Usage: identical to run_synthesis, e.g.
    python scripts/fast_loop_spider50.py --task ... --dataset spider ...
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import synthesis.run_constants as run_constants

_FAST_SPLIT = run_constants.SPLIT_FILE_BY_DATASET["spider"].with_name(
    "spider_dev_proportional_50train_seed334.json"
)

# run_synthesis reads this mapping at call time; swapping the entry here is
# the only difference from a production run.
run_constants.SPLIT_FILE_BY_DATASET["spider"] = _FAST_SPLIT

from synthesis.run_synthesis import main  # noqa: E402


if __name__ == "__main__":
    if "--dataset" in sys.argv and "spider" not in sys.argv:
        sys.exit("fast_loop_spider50 only supports --dataset spider")
    main()
