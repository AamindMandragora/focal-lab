"""Split-provenance guard and metadata for train/eval comparisons.

Why this module exists (incident 2026-07-17): synthesis runs default to the
TRAIN side of a split file, while baseline runs (run_legacy_fixed_strategy)
and re-evals default to the EVAL/TEST side. Three recorded cells (GSM
Qwen3.5-4B/9B, Spider Qwen3.5-4B) were launched with an acceptance bar
(--min-accuracy) that had been measured on the eval side, then scored on the
train side — an apples-to-oranges comparison that made metaDecode look far
under bars it was actually near. Nothing recorded which split a run used, so
the mismatch was only recoverable from wrapper logs.

This module fixes both halves at the root:

1. `check_bar_split_provenance(...)` — called at launch by run_synthesis.
   When a run uses a split file and a nonzero accuracy bar, the caller must
   declare which split the bar was measured on (--bar-split-name), and it
   must match the split being evaluated. A mismatch always refuses to start —
   there is no override flag; cross-split comparison is exactly the bug this
   guard exists to prevent.

2. `split_provenance_metadata(...)` — the split file/name (and declared bar
   split) that every synthesis report and eval output JSON now embeds, so any
   future number is traceable to its split without log forensics.

Pure logic, no heavy imports — unit-testable off-GPU.
"""

from __future__ import annotations

from typing import Any


class BarSplitProvenanceError(ValueError):
    """Raised when an accuracy bar's split provenance is missing or mismatched."""


# One dataset, one vocabulary, zero aliases. These match the index keys the
# split files actually store: GSM manifests have train_indices/eval_indices,
# Spider manifests have train_indices/test_indices. The old "eval" alias for
# Spider (removed 2026-07-17) meant one concept had three names and four
# translation sites, which is how the bar/eval split mixup stayed invisible.
SPLIT_NAMES_BY_DATASET = {
    "gsm_symbolic": ("train", "eval"),
    "spider": ("train", "test"),
}


def validate_split_name(dataset: str, split_name: str | None, flag: str) -> None:
    """Reject split names outside the dataset's vocabulary (no aliases)."""
    valid = SPLIT_NAMES_BY_DATASET.get(dataset)
    if valid is None or split_name is None:
        return
    if split_name not in valid:
        hint = (
            " Spider's held-out side is named 'test' (the 'eval' alias was removed 2026-07-17)."
            if dataset == "spider" and split_name == "eval"
            else ""
        )
        raise BarSplitProvenanceError(
            f"{flag} '{split_name}' is not a valid split name for {dataset}; "
            f"expected one of {list(valid)}.{hint}"
        )


def check_bar_split_provenance(
    dataset: str,
    split_file: str | None,
    split_name: str | None,
    min_accuracy: float,
    bar_split_name: str | None,
) -> None:
    """Refuse to launch when the accuracy bar's split doesn't match the eval split.

    The guard only applies when a train/eval split file is actually in use
    (no split file = no ambiguity) and the run has a real accuracy bar.
    SMILES has no split-file mechanism, so it is exempt.
    """
    if dataset == "smiles":
        return
    if split_file is None:
        return
    if min_accuracy is None or min_accuracy <= 0:
        return

    validate_split_name(dataset, split_name, "split name")
    validate_split_name(dataset, bar_split_name, "--bar-split-name")
    eval_side = split_name
    bar_side = bar_split_name

    if bar_side is None:
        valid = "|".join(SPLIT_NAMES_BY_DATASET.get(dataset, ("train", "eval")))
        raise BarSplitProvenanceError(
            f"--min-accuracy {min_accuracy} is used as an acceptance bar on the "
            f"'{eval_side}' side of {split_file}, but --bar-split-name was not "
            "given. Declare which split the bar was measured on "
            f"(--bar-split-name {valid}). Refusing to launch: comparing "
            "a bar from one split side against scores from the other produced "
            "wrong win/loss verdicts on 2026-07-17 (GSM Qwen3.5-4B/9B)."
        )
    if bar_side != eval_side:
        raise BarSplitProvenanceError(
            f"Bar split mismatch: --bar-split-name '{bar_side}' but this run "
            f"evaluates the '{eval_side}' side of {split_file}. Either re-measure "
            f"the bar on '{eval_side}' or evaluate on '{bar_side}'. There is no "
            "override: cross-split bars produced wrong verdicts on 2026-07-17."
        )


def split_provenance_metadata(
    evaluator: Any,
    bar_split_name: str | None = None,
) -> dict:
    """Split provenance block for run_configuration / output JSONs.

    Reads the split attributes the evaluator already carries; safe to call
    with any evaluator (missing attributes become None).
    """

    def _str_or_none(value: Any) -> str | None:
        return str(value) if value is not None else None

    return {
        "gsm_split_file": _str_or_none(getattr(evaluator, "gsm_split_file", None)),
        "gsm_split_name": getattr(evaluator, "gsm_split_name", None),
        "spider_split_file": _str_or_none(getattr(evaluator, "spider_split_file", None)),
        "spider_split_name": getattr(evaluator, "spider_split_name", None),
        "bar_split_name": bar_split_name,
    }
