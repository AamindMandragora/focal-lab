"""END-OF-TURN may only be chosen when the query is actually finished.

Why this file exists
--------------------
A Spider run scored 0% accuracy and 0% syntax on all 300 examples. The cause,
established by a controlled comparison in
``saved-results/2026-07-24-constrained-path-root-cause.md`` (attempt 8 scored
35.0%/88.7% with 0/300 empty outputs; attempt 9 scored 0.0%/0.0% with 300/300
empty outputs), is that the END-OF-TURN token is treated as legal at every
parser state.

The grammar mask already marks END-OF-TURN invalid at position 0. But
``MaskValidNextAndEos`` then force-set it back to allowed on every step:

    eos_indices = self._token_indices_for_token(eosToken)
    if eos_indices:
        subset_mask[eos_indices] = True      # <- unconditional

So the model could pick "stop" as its very first token. Every generated
strategy breaks out of its loop on END-OF-TURN *before* appending anything, so
one such pick throws the whole answer away: zero tokens, empty output, and both
scores zero (syntax is computed inside answer extraction, so an empty answer
forces both).

The rule these tests pin down
-----------------------------
END-OF-TURN is allowed only when stopping is actually legal:

    prefix is a complete query      -> allowed   (normal finish)
    prefix incomplete, moves exist  -> BLOCKED   (must keep writing)
    prefix incomplete, no moves     -> allowed   (real dead end; stop, do not crash)

The third case matters: before this change the unconditional force-allow meant
the "no valid tokens" guard could never fire. Gating without handling dead ends
would turn a silent wrong answer into a crash.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from synthesis.evaluate.benchmarks.common.model_utils import _TensorizedLMBase

# Token layout for the stand-in model: four ordinary tokens plus END-OF-TURN.
VOCAB = ["SELECT", "count", "FROM", "singer", "<|im_end|>"]
EOS_INDEX = 4
NON_EOS = [0, 1, 2, 3]


class _FakeParser:
    """Only the two questions the masking code asks of a parser."""

    def __init__(self, *, complete: bool, valid_indices):
        self._complete = complete
        self._valid_indices = list(valid_indices)

    def IsCompletePrefix(self, prefix):
        return self._complete

    # Kept for implementations that ask in the negative instead.
    def IsValidPrefix(self, prefix):
        return True


class _Recorder:
    def update_tensors(self, *args, **kwargs):
        pass


class _StandInLM:
    """Enough of _TensorizedLMBase's state to run the real masking method.

    The parser mask deliberately reports END-OF-TURN as NOT allowed, which is
    what the real grammar mask does at position 0. Any test failure therefore
    means the code under test put it back.
    """

    def __init__(self, parser_valid_indices):
        n = len(VOCAB)
        self._token_ids = list(range(n))
        self._token_ids_tensor = torch.arange(n)
        self._logits_tensor = torch.zeros(n, dtype=torch.float32)
        self._full_logits = None
        self._logits_dirty = False
        self.Logits = _Recorder()
        self._parser_valid_indices = list(parser_valid_indices)

    def _parser_full_mask(self, parser, prefix):
        mask = torch.zeros(len(self._token_ids), dtype=torch.bool)
        for i in self._parser_valid_indices:
            mask[i] = True
        return mask

    def _token_indices_for_token(self, token):
        return [EOS_INDEX]

    def is_blocked(self, index):
        """A token is blocked when its logit was driven to the -1e9 floor."""
        return bool(self._logits_tensor[index].item() <= -1e8)


def _run_mask(lm, parser, prefix):
    """Call the real method with the stand-in as ``self``."""
    _TensorizedLMBase.MaskValidNextAndEos(lm, parser, prefix, "<|im_end|>")


# --------------------------------------------------------------------------
# The regression that produced 0% / 0%
# --------------------------------------------------------------------------

def test_eos_is_blocked_at_the_very_first_token():
    """The exact 300/300-empty-output bug: stopping before writing anything."""
    lm = _StandInLM(parser_valid_indices=NON_EOS)
    parser = _FakeParser(complete=False, valid_indices=NON_EOS)

    _run_mask(lm, parser, [])

    assert lm.is_blocked(EOS_INDEX), (
        "END-OF-TURN was left selectable on an empty prefix. The model can "
        "then choose 'stop' as its first token; every generated strategy "
        "breaks before appending, so the output is empty and both accuracy "
        "and syntax score 0. This is the attempt-9 failure."
    )
    assert not lm.is_blocked(0), "a real SQL token must still be selectable"


def test_eos_is_blocked_while_the_query_is_unfinished():
    """Mid-query: 'SELECT count' is not a query yet, so stopping is illegal."""
    lm = _StandInLM(parser_valid_indices=NON_EOS)
    parser = _FakeParser(complete=False, valid_indices=NON_EOS)

    _run_mask(lm, parser, ["SELECT", "count"])

    assert lm.is_blocked(EOS_INDEX), (
        "END-OF-TURN was selectable on an incomplete query, which lets the "
        "model emit a truncated statement that cannot parse."
    )


# --------------------------------------------------------------------------
# The behaviour that must be preserved
# --------------------------------------------------------------------------

def test_eos_is_allowed_once_the_query_is_complete():
    """Gating must not make it impossible to ever stop."""
    lm = _StandInLM(parser_valid_indices=NON_EOS)
    parser = _FakeParser(complete=True, valid_indices=NON_EOS)

    _run_mask(lm, parser, ["SELECT", "count", "FROM", "singer"])

    assert not lm.is_blocked(EOS_INDEX), (
        "END-OF-TURN was blocked on a complete query, so generation can never "
        "terminate normally and every example burns to max_steps."
    )


def test_eos_is_allowed_at_a_genuine_dead_end():
    """No legal continuation and not complete: stop cleanly, do not crash.

    Before gating, the 'no valid next tokens' guard was unreachable because EOS
    was always force-added. Gating makes it reachable, so this must not raise.
    """
    lm = _StandInLM(parser_valid_indices=[])
    parser = _FakeParser(complete=False, valid_indices=[])

    _run_mask(lm, parser, ["SELECT", "count"])

    assert not lm.is_blocked(EOS_INDEX), (
        "At a dead end with no legal continuation, END-OF-TURN must stay "
        "selectable so the example stops instead of raising or spinning to "
        "max_steps."
    )


def test_boost_path_uses_the_same_rule():
    """The soft/boost path must not reopen the hole the hard path closed."""
    lm = _StandInLM(parser_valid_indices=NON_EOS)
    parser = _FakeParser(complete=False, valid_indices=NON_EOS)

    _TensorizedLMBase.BoostValidNextAndEos(lm, parser, [], 10.0, "<|im_end|>")

    eos_logit = lm._logits_tensor[EOS_INDEX].item()
    best_real = max(lm._logits_tensor[i].item() for i in NON_EOS)
    assert eos_logit < best_real, (
        "BoostValidNextAndEos boosted END-OF-TURN on an empty prefix, so the "
        "soft-constrained helpers can still stop before writing anything."
    )


# --------------------------------------------------------------------------
# The formal contract, so a future strategy cannot reintroduce this
# --------------------------------------------------------------------------

# Every helper that can hand a strategy an END-OF-TURN token. Listed by name
# rather than matched by pattern: the postconditions are phrased several
# different ways, and a pattern that catches only some of them would let a
# "fix" pass while leaving the rest of the hole open.
EOS_ADMITTING_HELPERS = [
    "ConstrainedStep",
    "GroupBoostedConstrainedStep",
    "AdaptiveConstrainedStep",
    "AdaptiveConstrainedStepWithPenalties",
    "PenalizedConstrainedStep",
    "BoostedConstrainedStep",
    "SafeBoostedConstrainedStep",
    "SafePenalizedConstrainedStep",
    "SoftConstrainedStep",
    "SafeSoftConstrainedStep",
    "ConfidenceGatedStep",
    "TopValidCandidates",
    "RepetitionPenaltyStep",
    "SafeRepetitionPenaltyStep",
    "TemperatureConstrainedStep",
    "SafeTemperatureConstrainedStep",
]


def _ensures_block(source: str, method_name: str) -> str:
    """The contract lines of one method: from its declaration to its body."""
    lines = source.splitlines()
    start = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(("method " + method_name, "static method " + method_name)):
            start = i
            break
    assert start is not None, f"{method_name} not found in the library"

    collected = []
    for line in lines[start:]:
        stripped = line.strip()
        if stripped == "{":  # body begins; contract is over
            break
        collected.append(stripped)
    return "\n".join(collected)


def test_dafny_contract_no_longer_exempts_eos_unconditionally():
    """The library must not promise that END-OF-TURN is legal at any state.

    These helpers carried a postcondition of the form

        ensures (next == eosToken) || (parser.ValidNextToken(prefix, next))

    which reads as 'either it is END-OF-TURN, or it is parser-valid' -- placing
    no requirement at all on END-OF-TURN. A strategy that picks an
    EOS-admitting helper is then formally verified while producing nothing,
    which is exactly what attempt 9 did on all 300 examples.
    """
    library = REPO_ROOT / "synthesis" / "verify" / "library" / "VerifiedAgentSynthesis.dfy"
    source = library.read_text(encoding="utf-8")

    offenders = []
    for name in EOS_ADMITTING_HELPERS:
        contract = _ensures_block(source, name)
        if "eosToken" not in contract:
            continue  # this helper cannot return END-OF-TURN at all
        if "IsCompletePrefix" not in contract:
            offenders.append(name)

    assert offenders == [], (
        f"{len(offenders)} of {len(EOS_ADMITTING_HELPERS)} helpers still let a "
        f"strategy stop at any parser state: {offenders}. Each must state when "
        "stopping is legal, i.e. add "
        "'ensures next == eosToken ==> parser.IsCompletePrefix(constrainedPrefix)' "
        "(or the dead-end equivalent) to its contract."
    )
