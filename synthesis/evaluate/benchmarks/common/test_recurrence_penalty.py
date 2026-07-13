import math
import sys
from unittest.mock import MagicMock


for _heavy in ("torch", "transformers", "vllm"):
    if _heavy not in sys.modules:
        try:
            __import__(_heavy)
        except Exception:
            sys.modules[_heavy] = MagicMock()

from synthesis.evaluate.benchmarks.common.model_utils import _TensorizedLMBase  # noqa: E402


class _Scalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value

    def __add__(self, other):
        return _Scalar(self.value + other)

    def __radd__(self, other):
        return _Scalar(other + self.value)


class _Vector:
    def __init__(self, values):
        self.values = list(values)

    def numel(self):
        return len(self.values)

    def __getitem__(self, index):
        return _Scalar(self.values[index])

    def __setitem__(self, index, value):
        self.values[index] = value.value if isinstance(value, _Scalar) else value


def _lm():
    lm = object.__new__(_TensorizedLMBase)
    lm.instruction_text = "EXAMPLE-1 "
    lm._tried_token_penalties = {}
    lm._penalty_instruction_key = None
    lm._recurrence_penalty = 0.3
    lm._recurrence_flat = False
    lm._logits_dirty = False
    lm._token_str_to_indices = {"A": [0], "B": [1], "C": [2]}
    lm._token_ids_tensor = _Vector([2, 0, 1])
    lm._logits_tensor = _Vector([-0.1, -0.5, -2.0])
    lm._full_logits = _Vector([-0.5, -2.0, -0.1])
    return lm


def test_empty_penalty_map_is_a_noop():
    lm = _lm()
    before = list(lm._logits_tensor.values), list(lm._full_logits.values)
    lm._apply_recurrence_penalty("EXAMPLE-1 x")
    assert (lm._logits_tensor.values, lm._full_logits.values) == before


def test_penalty_changes_subset_and_matching_full_vocabulary_logit():
    lm = _lm()
    lm.PenalizeTriedTokenAt(["x"], "A")
    lm._apply_recurrence_penalty("EXAMPLE-1 x")
    delta = math.log(0.3)
    assert lm._logits_tensor.values[0] == -0.1 + delta
    assert lm._full_logits.values[2] == -0.1 + delta


def test_penalty_is_cumulative_by_default():
    lm = _lm()
    lm.PenalizeTriedTokenAt(["x"], "A")
    lm.PenalizeTriedTokenAt(["x"], "A")
    lm._apply_recurrence_penalty("EXAMPLE-1 x")
    assert lm._logits_tensor.values[0] == -0.1 + 2 * math.log(0.3)


def test_penalty_does_not_cross_prefixes():
    lm = _lm()
    lm.PenalizeTriedTokenAt(["x"], "A")
    lm._apply_recurrence_penalty("EXAMPLE-1 y")
    assert lm._logits_tensor.values == [-0.1, -0.5, -2.0]


def test_new_instruction_clears_previous_example_penalties():
    lm = _lm()
    lm.PenalizeTriedTokenAt(["x"], "A")
    lm.instruction_text = "EXAMPLE-2 "
    lm.PenalizeTriedTokenAt(["x"], "B")
    assert list(lm._tried_token_penalties) == ["EXAMPLE-2 x"]


def test_flat_mode_applies_one_penalty_after_repeated_failures():
    lm = _lm()
    lm._recurrence_flat = True
    lm.PenalizeTriedTokenAt(["x"], "A")
    lm.PenalizeTriedTokenAt(["x"], "A")
    lm._apply_recurrence_penalty("EXAMPLE-1 x")
    assert lm._logits_tensor.values[0] == -0.1 + math.log(0.3)
