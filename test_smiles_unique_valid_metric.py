"""TDD for the SMILES metric switch: accuracy must become the unique-valid RATE
(deduplicated RDKit-valid + in-class + non-exemplar molecules / N), NOT the gameable
membership rate. A collapsed strategy (one molecule x N) must score ~1/N, not 1.0.

Run on focal:  python test_smiles_unique_valid_metric.py
"""
from synthesis.evaluate.benchmarks.smiles.metrics import smiles_trial_metrics
from synthesis.evaluate.benchmarks.smiles import eval_logic


def _sample(smiles, *, valid=True, member=True, exemplar=False):
    uvc = bool(smiles and valid and member and not exemplar)
    return {
        "smiles_eval": {
            "smiles": smiles,
            "rdkit_valid": valid,
            "class_membership": member,
            "is_prompt_exemplar": exemplar,
            "unique_valid_candidate": uvc,
        }
    }


def test_collapse_scores_near_zero():
    # The exact pathology: emit "OC" 100 times. All valid + in-class, but 1 distinct.
    collapse = [_sample("OC") for _ in range(100)]
    trial = smiles_trial_metrics(collapse)
    assert trial["unique_valid_count"] == 1, trial["unique_valid_count"]
    acc = eval_logic.override_accuracy(
        {"smiles_paper_trial": trial}, num_examples=100
    )
    assert abs(acc - 0.01) < 1e-9, f"collapse should score ~0.01, got {acc}"


def test_diverse_set_scores_high():
    # 80 distinct valid in-class molecules among 100 samples -> unique-valid rate 0.80.
    mols = [f"OC{'C'*i}" for i in range(80)]  # 80 distinct strings
    diverse = [_sample(m) for m in mols] + [_sample("OC") for _ in range(20)]
    trial = smiles_trial_metrics(diverse)
    # 80 distinct from the first group + "OC" already counted in group0 -> 80 unique
    acc = eval_logic.override_accuracy(
        {"smiles_paper_trial": trial}, num_examples=100
    )
    assert acc > 0.5, f"diverse set should score high, got {acc}"
    assert acc > eval_logic.override_accuracy(
        {"smiles_paper_trial": smiles_trial_metrics([_sample('OC')]*100)}, num_examples=100
    )


def test_invalids_penalized_by_N_denominator():
    # 30 distinct valid + 70 invalid -> unique-valid rate 0.30 (denominator is N, not valid-count).
    good = [_sample(f"OC{'C'*i}") for i in range(30)]
    bad = [_sample("X", valid=False, member=False) for _ in range(70)]
    trial = smiles_trial_metrics(good + bad)
    acc = eval_logic.override_accuracy({"smiles_paper_trial": trial}, num_examples=100)
    assert abs(acc - 0.30) < 1e-9, f"expected 0.30, got {acc}"


if __name__ == "__main__":
    import traceback
    passed = failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
                passed += 1
            except Exception as e:
                print(f"FAIL {name}: {type(e).__name__}: {e}")
                traceback.print_exc()
                failed += 1
    print(f"\n{passed} passed, {failed} failed")
    raise SystemExit(1 if failed else 0)
