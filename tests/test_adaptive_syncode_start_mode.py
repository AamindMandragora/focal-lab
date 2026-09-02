from pathlib import Path
import sys


SYNCODE_ROOT = Path(__file__).resolve().parents[1] / "synthesis" / "evaluate" / "syncode"
sys.path.insert(0, str(SYNCODE_ROOT))

from syncode.language_model import (  # noqa: E402
    ConstrainedMode,
    UnconstrainedMode,
    _adaptive_initial_state,
)


class _GrammarDecoder:
    def __init__(self):
        self.reset_offsets = []

    def reset_adaptive(self, offset):
        self.reset_offsets.append(offset)


def test_adaptive_decoder_can_start_constrained_at_the_first_generated_token():
    decoder = _GrammarDecoder()

    last_end, state, start_from = _adaptive_initial_state(
        True,
        [[10, 11, 12]],
        decoder,
    )

    assert last_end == 2
    assert state is ConstrainedMode
    assert start_from == 3
    assert decoder.reset_offsets == [3]


def test_adaptive_decoder_keeps_delimited_benchmarks_unconstrained_initially():
    decoder = _GrammarDecoder()

    _last_end, state, start_from = _adaptive_initial_state(
        False,
        [[10, 11, 12]],
        decoder,
    )

    assert state is UnconstrainedMode
    assert start_from is None
    assert decoder.reset_offsets == []
