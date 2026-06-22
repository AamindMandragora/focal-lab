"""Metric computation helpers — no heavy imports (no torch/vllm/transformers)."""


def choose_denominator_basis(early_stop_reason, planned_num_examples, evaluated_count) -> int:
    """Return planned_num_examples when any early stop occurred, else evaluated_count.

    On a runtime/timeout or accuracy early-stop, only a fraction of examples
    ran. Using evaluated_count as the denominator inflates accuracy, so we
    always use planned_num_examples in those cases.
    """
    return planned_num_examples if early_stop_reason else evaluated_count
