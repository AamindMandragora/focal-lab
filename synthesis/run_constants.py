"""Hard-coded run settings for synthesis (2026-07-17 simplification).

These used to be ~50 CLI flags on run_synthesis.py. Per the user's code
review, anything with exactly one correct value is a constant here, not a
flag. If a value needs to change, change it HERE — do not resurrect a flag.

The surviving flags on run_synthesis.py are only the things that genuinely
vary per run: dataset, models, bars, iteration count, sample size, the two
vLLM sizing knobs (GPU memory share and context length, which change per
GPU-sharing situation), and the reasoning budget.
"""

from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Evaluation runtime
# ---------------------------------------------------------------------------
EVAL_BACKEND = "vllm"  # every recorded run uses vllm; huggingface/openai removed
TEMPERATURE = 0.7
VLLM_ENFORCE_EAGER = True
EVAL_EARLY_STOP_ON_ANSWER = True  # CRANE-style answer stopping; adopted 2026-07-17
VLLM_GPU_MEMORY_UTILIZATION = 0.8  # GPU memory fraction reserved by vLLM
VLLM_MAX_MODEL_LEN = 16384  # max context length passed to vLLM (must fit prompt + output)

# Threshold-impossible early stops may only fire once at least this many
# examples have been evaluated in the iteration (settled value; was a flag).
MIN_EXAMPLES_BEFORE_THRESHOLD_STOP = 15

# Whether a dataset's outputs must contain a visible << >> span is NOT set
# here. It is a fact about each benchmark's own output surface -- GSM
# strategies live inside << >> spans, while Spider and SMILES generate whole
# outputs and cannot emit that span at all -- so each benchmark answers for
# itself via emits_visible_delimiters(), and callers go through
# registry.resolve_require_delimiters(). Do not add a lookup table back here:
# it would override the benchmark and silently make --require-delimiters dead.
# Guarded by tests/test_delimiter_single_source_of_truth.py.

# ---------------------------------------------------------------------------
# Splits — one canonical split file per dataset (SMILES has none).
# The Spider 50-example fast-iteration set lives in its own script
# (scripts/fast_loop/), not here.
# ---------------------------------------------------------------------------
SPLITS_DIR = _REPO_ROOT / "environment" / "benchmark_splits"
SPLIT_FILE_BY_DATASET = {
    "gsm_symbolic": SPLITS_DIR / "gsm_symbolic_crane_proportional_49x49_seed123.json",
    "spider": SPLITS_DIR / "spider_dev_proportional_300x300_seed334.json",
    "smiles": None,
}
# Synthesis always scores itself on the train side; held-out re-evals go
# through reevaluate_compiled_csd.py, never run_synthesis.
SYNTHESIS_SPLIT_NAME = "train"

# ---------------------------------------------------------------------------
# Output layout
# ---------------------------------------------------------------------------
OUTPUT_DIR = _REPO_ROOT / "outputs" / "generated"
GSM_SOURCE_DIR = _REPO_ROOT / "legacy" / "CRANE" / "src" / "gsm_symbolic"
GRAMMARS_DIR = None  # use the built-in synthesis/evaluate/grammars

# ---------------------------------------------------------------------------
# Author (strategy-generation) model
# ---------------------------------------------------------------------------
# Extended thinking is ALWAYS on for hosted Claude authors — non-thinking
# authoring is not a supported mode. Effort/display are fixed; only the
# budget is a flag (--synthesizer-reasoning-budget), because it is
# provider-agnostic and genuinely tunable.
ANTHROPIC_EFFORT = "high"
ANTHROPIC_THINKING_DISPLAY = "summarized"
SYNTHESIZER_REASONING_BUDGET_DEFAULT = 4096  # tokens; min 1024, must be < max output tokens

# Claude Code Max author process (single active caller: launch_queue_cell.py).
# Account identity (config dir / expected account) stays env-based because it
# is a billing-safety check, not a tuning knob.
CLAUDE_EXECUTABLE = os.environ.get("CSD_CLAUDE_EXECUTABLE", "claude")
CLAUDE_IDLE_TIMEOUT_SECONDS = 900.0
CLAUDE_EMERGENCY_TIMEOUT_SECONDS = 7200.0
CLAUDE_MAX_RETRIES = 2
CLAUDE_RETRY_DELAY_SECONDS = 30.0
CLAUDE_TELEMETRY_DIR = _REPO_ROOT / ".context" / "claude_stream_telemetry"
CLAUDE_AUTHOR_LOCK_FILE = _REPO_ROOT / ".context" / "claude_author.lock"
