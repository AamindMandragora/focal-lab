"""Rolling few-shot prompt for the SMILES generative eval (Fix A, 2026-06-10).

Each successive eval example's prompt grows to include the molecules accepted from
earlier examples — the SAME diversity mechanism the baseline adapters
(CARS/GCD/IterGen/CRANE/unconstrained) use in run_legacy_fixed_strategy.py. This gives
the CSD (compiled-strategy) eval path parity with the baselines on SMILES diversity.
Kept byte-identical to that path; see tests/test_smiles_rolling_suffix.py.
"""

MAX_SUFFIX_CHARS = 45000


def cap_suffix(suffix: str) -> str:
    """Drop the oldest appended molecules if suffix exceeds ``MAX_SUFFIX_CHARS``."""
    if len(suffix) <= MAX_SUFFIX_CHARS:
        return suffix
    lines = suffix.split("\n")
    while len("\n".join(lines)) > MAX_SUFFIX_CHARS and len(lines) > 1:
        lines.pop(0)
    return "\n".join(lines)


def apply_suffix(example: dict, suffix_by_class: dict) -> None:
    """Prepend the accumulated rolling suffix onto this example's prompt, in place."""
    cls = str(example.get("class_name", ""))
    example["prompt"] = example["prompt"].rstrip() + suffix_by_class.get(cls, "")


def update_suffix(example: dict, suffix_by_class: dict, actual: str, syntax_valid: bool) -> None:
    """After scoring, append this molecule to the class's rolling suffix if it is a
    syntactically-valid, non-empty molecule (the baseline's exact gate)."""
    if syntax_valid and actual:
        cls = str(example.get("class_name", ""))
        suffix_by_class[cls] = cap_suffix(
            suffix_by_class.get(cls, "") + f" {actual}\nMolecule:"
        )
