"""Parity test (Fix A, 2026-06-10): the SMILES rolling-prompt helper used by the CSD
eval path must be byte-identical in behavior to the baseline rolling prompt in
run_legacy_fixed_strategy.py, so metaDecode is evaluated under the SAME diversity
protocol as the CARS/GCD/IterGen/CRANE/unconstrained baselines."""
from synthesis.evaluate.benchmarks.smiles import rolling_prompt
from synthesis.evaluate.run_legacy_fixed_strategy import _cap_suffix as baseline_cap
import synthesis.evaluate.run_legacy_fixed_strategy as legacy


def test_max_suffix_chars_matches_baseline():
    assert rolling_prompt.MAX_SUFFIX_CHARS == legacy._MAX_SUFFIX_CHARS


def test_cap_suffix_short_matches_baseline():
    s = " CCO\nMolecule: OCCO\nMolecule:"
    assert rolling_prompt.cap_suffix(s) == baseline_cap(s)


def test_cap_suffix_over_cap_matches_baseline():
    one = " " + ("C" * 50) + "\nMolecule:"
    s = one * 2000
    assert len(s) > rolling_prompt.MAX_SUFFIX_CHARS
    assert rolling_prompt.cap_suffix(s) == baseline_cap(s)
    assert len(rolling_prompt.cap_suffix(s)) <= rolling_prompt.MAX_SUFFIX_CHARS


def _baseline_three_part(examples, outputs):
    """Re-implementation of the baseline's exact apply/update over a sequence."""
    suffix = {}
    seen_prompts = []
    for ex, (actual, syntax_valid) in zip(examples, outputs):
        cls = str(ex.get("class_name", ""))
        applied = ex["prompt"].rstrip() + suffix.get(cls, "")
        seen_prompts.append(applied)
        if syntax_valid and actual:
            suffix[cls] = baseline_cap(suffix.get(cls, "") + f" {actual}\nMolecule:")
    return seen_prompts, suffix


def test_apply_update_sequence_matches_baseline():
    base = [{"class_name": "chain_extenders", "prompt": "P\nMolecule: "} for _ in range(4)]
    outputs = [("OCCO", True), ("", True), ("OC", True), ("CCO", False)]
    exp_prompts, exp_suffix = _baseline_three_part([dict(e) for e in base], outputs)
    suffix = {}
    got_prompts = []
    for e, (actual, sv) in zip([dict(x) for x in base], outputs):
        rolling_prompt.apply_suffix(e, suffix)
        got_prompts.append(e["prompt"])
        rolling_prompt.update_suffix(e, suffix, actual, sv)
    assert got_prompts == exp_prompts
    assert suffix == exp_suffix
