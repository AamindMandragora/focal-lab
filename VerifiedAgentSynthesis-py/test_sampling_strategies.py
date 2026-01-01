import os

import _dafny as _dafny
import CRANE as CRANE
import module_ as module_


def _tok(s: str) -> _dafny.Seq:
    # Dafny strings compile to Seq[CodePoint]
    return _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, s))


def test_argmax_is_greedy():
    a, b, c = _tok("a"), _tok("b"), _tok("c")
    logits = _dafny.Map({a: 0.1, b: 3.0, c: 2.0})
    candidates = _dafny.Set([a, b, c])
    out = module_.SampleWithStrategy(logits, candidates, CRANE.SamplingStrategy_ArgMax())
    assert out == b


def test_topk_never_selects_outside_topk():
    os.environ["CRANE_RANDOM_SEED"] = "0"
    os.environ["CRANE_TOPK"] = "1"

    a, b, c = _tok("a"), _tok("b"), _tok("c")
    logits = _dafny.Map({a: 0.1, b: 3.0, c: 2.0})
    candidates = _dafny.Set([a, b, c])
    out = module_.SampleWithStrategy(logits, candidates, CRANE.SamplingStrategy_TopK())
    assert out == b  # only top-1 is allowed


def test_nucleus_restricts_to_top_p_mass():
    os.environ["CRANE_RANDOM_SEED"] = "1"
    os.environ["CRANE_TOP_P"] = "0.5"

    # Make one token dominate so nucleus should contain only that token for top_p=0.5
    a, b = _tok("a"), _tok("b")
    logits = _dafny.Map({a: 10.0, b: -10.0})
    candidates = _dafny.Set([a, b])
    out = module_.SampleWithStrategy(logits, candidates, CRANE.SamplingStrategy_Nucleus())
    assert out == a


def test_temperature_low_is_almost_greedy():
    os.environ["CRANE_RANDOM_SEED"] = "2"
    os.environ["CRANE_TEMPERATURE"] = "0.0001"

    a, b = _tok("a"), _tok("b")
    logits = _dafny.Map({a: 0.0, b: 1.0})
    candidates = _dafny.Set([a, b])
    out = module_.SampleWithStrategy(logits, candidates, CRANE.SamplingStrategy_Temperature())
    assert out == b


if __name__ == "__main__":
    # Minimal runner so this can be executed without pytest.
    test_argmax_is_greedy()
    test_topk_never_selects_outside_topk()
    test_nucleus_restricts_to_top_p_mass()
    test_temperature_low_is_almost_greedy()
    print("ok")


