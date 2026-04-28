"""
Tests for the updated VerifiedAgentSynthesis library functions.

Covers: new LM methods, CSDHelpers suffix-based design, natural delimiter
helpers, and end-to-end strategy smoke test.
"""

import sys
from pathlib import Path

import pytest

from verification.transpiler.transpiler import transpile_contract_library

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "generation" / "csd"))

from VerifiedAgentSynthesis import (
    CSDHelpers,
    LM,
    LeftDelimiter,
    Parser,
    Prefix,
    RightDelimiter,
    SpacedLeftDelimiter,
    SpacedRightDelimiter,
    Token,
)


# ---------------------------------------------------------------------------
# Minimal concrete LM for testing
# ---------------------------------------------------------------------------

class SimpleLM(LM):
    def __init__(self, tokens: list[str]) -> None:
        self.Tokens = tokens
        self.Ids = list(range(len(tokens)))
        self.Logits = [0.0] * len(tokens)

    def GenerateLogits(self, input: Prefix) -> None:
        # Give each token a distinct logit based on its index
        for i in range(len(self.Tokens)):
            self.Logits[i] = float(i)

    def ChooseNextToken(self) -> Token:
        best_i = -1
        best_l = -2e9
        for i in range(len(self.Tokens)):
            if self.Logits[i] > best_l and self.Logits[i] != -1e9:
                best_l = self.Logits[i]
                best_i = i
        if best_i == -1:
            raise ValueError("All tokens masked")
        return self.Tokens[best_i]


# ---------------------------------------------------------------------------
# Minimal concrete Parser for testing
# ---------------------------------------------------------------------------

GRAMMAR_TOKENS = {"a", "b", "c"}

class SimpleParser(Parser):
    """Accepts sequences of 'a', 'b', 'c'; complete after exactly 3 tokens."""

    def IsValidPrefix(self, prefix: Prefix) -> bool:
        return len(prefix) <= 3 and all(t in GRAMMAR_TOKENS for t in prefix)

    def IsCompletePrefix(self, prefix: Prefix) -> bool:
        return len(prefix) == 3 and all(t in GRAMMAR_TOKENS for t in prefix)

    def ValidNextTokens(self, prefix: Prefix) -> Prefix:
        if len(prefix) >= 3:
            return []
        return list(GRAMMAR_TOKENS)


class ExtendableCompleteParser(Parser):
    """Accepts 'a'/'b' prefixes up to 3 tokens; complete after the first token."""

    def IsValidPrefix(self, prefix: Prefix) -> bool:
        return len(prefix) <= 3 and all(t in {"a", "b"} for t in prefix)

    def IsCompletePrefix(self, prefix: Prefix) -> bool:
        return len(prefix) >= 1 and self.IsValidPrefix(prefix)

    def ValidNextTokens(self, prefix: Prefix) -> Prefix:
        if len(prefix) >= 3:
            return []
        return ["a", "b"]


# ---------------------------------------------------------------------------
# LM: BiasToken / BiasTokens
# ---------------------------------------------------------------------------

def test_bias_token_adds_delta():
    lm = SimpleLM(["x", "y", "z"])
    lm.Logits = [1.0, 2.0, 3.0]
    lm.BiasToken("x", 5.0)
    assert lm.Logits[0] == 6.0
    assert lm.Logits[1] == 2.0  # unchanged


def test_bias_token_clamps_high():
    lm = SimpleLM(["x"])
    lm.Logits = [9e8]
    lm.BiasToken("x", 9e8)
    assert lm.Logits[0] == 1e9


def test_bias_token_clamps_low():
    lm = SimpleLM(["x"])
    lm.Logits = [-9e8]
    lm.BiasToken("x", -9e8)
    assert lm.Logits[0] == -1e9


def test_bias_tokens_applies_to_all():
    lm = SimpleLM(["x", "y", "z"])
    lm.Logits = [1.0, 2.0, 3.0]
    lm.BiasTokens(["x", "z"], 10.0)
    assert lm.Logits[0] == 11.0
    assert lm.Logits[1] == 2.0   # unchanged
    assert lm.Logits[2] == 13.0


# ---------------------------------------------------------------------------
# LM: ScaleToken / ScaleTokens
# ---------------------------------------------------------------------------

def test_scale_token_multiplies():
    lm = SimpleLM(["x", "y"])
    lm.Logits = [4.0, 2.0]
    lm.ScaleToken("x", 3.0)
    assert lm.Logits[0] == 12.0
    assert lm.Logits[1] == 2.0


def test_scale_tokens_multiplies_all():
    lm = SimpleLM(["x", "y"])
    lm.Logits = [2.0, 3.0]
    lm.ScaleTokens(["x", "y"], 2.0)
    assert lm.Logits[0] == 4.0
    assert lm.Logits[1] == 6.0


# ---------------------------------------------------------------------------
# LM: ClampLogits
# ---------------------------------------------------------------------------

def test_clamp_logits():
    lm = SimpleLM(["x", "y", "z"])
    lm.Logits = [-500.0, 0.0, 500.0]
    lm.ClampLogits(-100.0, 100.0)
    assert lm.Logits[0] == -100.0
    assert lm.Logits[1] == 0.0
    assert lm.Logits[2] == 100.0


# ---------------------------------------------------------------------------
# LM: TopKFilter
# ---------------------------------------------------------------------------

def test_top_k_filter_keeps_highest():
    lm = SimpleLM(["a", "b", "c", "d"])
    lm.Logits = [1.0, 4.0, 3.0, 2.0]
    lm.TopKFilter(2)
    assert lm.Logits[1] != -1e9  # "b" highest
    assert lm.Logits[2] != -1e9  # "c" second
    assert lm.Logits[0] == -1e9  # "a" masked
    assert lm.Logits[3] == -1e9  # "d" masked


# ---------------------------------------------------------------------------
# Parser: ValidContinuationCount
# ---------------------------------------------------------------------------

def test_valid_continuation_count():
    p = SimpleParser()
    assert p.ValidContinuationCount([]) == 3
    assert p.ValidContinuationCount(["a", "b", "c"]) == 0


# ---------------------------------------------------------------------------
# CSDHelpers: LongestValidSuffix
# ---------------------------------------------------------------------------

def make_helpers(tokens=None):
    if tokens is None:
        tokens = ["a", "b", "c", "<<", ">>", "x"]
    lm = SimpleLM(tokens)
    parser = SimpleParser()
    return CSDHelpers(lm, parser)


def test_longest_valid_suffix_empty_prefix():
    h = make_helpers()
    assert h.LongestValidSuffix([]) == []


def test_longest_valid_suffix_fully_valid():
    h = make_helpers()
    assert h.LongestValidSuffix(["a", "b"]) == ["a", "b"]


def test_longest_valid_suffix_strips_invalid_front():
    h = make_helpers()
    # "x" is not in GRAMMAR_TOKENS, so ["x", "a", "b"] prefix:
    # ["x","a","b"] — not valid (x not in grammar)
    # ["a","b"] — valid!
    result = h.LongestValidSuffix(["x", "a", "b"])
    assert result == ["a", "b"]


def test_longest_valid_suffix_all_invalid():
    h = make_helpers()
    result = h.LongestValidSuffix(["x", "x", "x"])
    assert result == []


def test_longest_valid_suffix_after_delimiter():
    h = make_helpers()
    # After emitting "<<", the suffix should reset to [] because "<<" is not a grammar token
    result = h.LongestValidSuffix(["a", "<<"])
    assert result == []


def test_longest_valid_suffix_tracking_through_constrained():
    h = make_helpers()
    # Simulates: prefix after << then constrained tokens
    result = h.LongestValidSuffix(["some", "<<", "a", "b"])
    assert result == ["a", "b"]


# ---------------------------------------------------------------------------
# CSDHelpers: UnconstrainedStep / ConstrainedStep
# ---------------------------------------------------------------------------

def test_unconstrained_step_returns_token_and_decrements():
    lm = SimpleLM(["a", "b", "c", "<<", ">>"])
    lm.Logits = [1.0, 2.0, 3.0, 0.0, 0.0]
    parser = SimpleParser()
    h = CSDHelpers(lm, parser)
    tok, steps = h.UnconstrainedStep([], [], 10)
    assert tok in lm.Tokens
    assert steps == 9


def test_constrained_step_produces_grammar_valid_token():
    lm = SimpleLM(["a", "b", "c", "<<", ">>", "x"])
    parser = SimpleParser()
    h = CSDHelpers(lm, parser)
    # generated = ["<<"] → LongestValidSuffix = [] → grammar start
    tok, steps = h.ConstrainedStep([], ["<<"], 10)
    assert tok in GRAMMAR_TOKENS
    assert steps == 9


# ---------------------------------------------------------------------------
# CSDHelpers: ForcedTokenStep / ergonomic wrappers
# ---------------------------------------------------------------------------

def test_forced_token_step_returns_exact_token():
    lm = SimpleLM(["a", "<<", ">>"])
    parser = SimpleParser()
    h = CSDHelpers(lm, parser)
    tok, steps = h.ForcedTokenStep([], [], "<<", 5)
    assert tok == "<<"
    assert steps == 4


def test_can_constrain_matches_suffix_completion():
    h = make_helpers()
    assert h.CanConstrain(["<<"])
    assert not h.CanConstrain(["<<", "a", "b", "c"])


def test_append_left_delimiter_appends_exact_token():
    lm = SimpleLM(["a", "<<", ">>"])
    parser = SimpleParser()
    h = CSDHelpers(lm, parser)
    generated, steps = h.AppendLeftDelimiter([], 5)
    assert generated == ["<<"]
    assert steps == 4


def test_append_constrained_step_appends_grammar_valid_token():
    lm = SimpleLM(["a", "b", "c", "<<", ">>", "x"])
    parser = SimpleParser()
    h = CSDHelpers(lm, parser)
    generated, steps = h.AppendConstrainedStep([], ["<<"], 10)
    assert generated[:1] == ["<<"]
    assert generated[-1] in GRAMMAR_TOKENS
    assert steps == 9


def test_unconstrained_step_masks_delimiter_variants_when_alternatives_exist():
    lm = SimpleLM(["safe", LeftDelimiter, RightDelimiter, SpacedLeftDelimiter, SpacedRightDelimiter])
    parser = SimpleParser()
    helpers = CSDHelpers(lm, parser)

    token, steps = helpers.UnconstrainedStep([], [], 10)

    assert token == "safe"
    assert steps == 9
    assert lm.IsMasked(LeftDelimiter)
    assert lm.IsMasked(RightDelimiter)
    assert lm.IsMasked(SpacedLeftDelimiter)
    assert lm.IsMasked(SpacedRightDelimiter)


def test_constrained_or_right_delimiter_allows_spaced_right_after_completion():
    lm = SimpleLM(["a", "b", "c", RightDelimiter, SpacedRightDelimiter])
    parser = SimpleParser()
    helpers = CSDHelpers(lm, parser)

    token, steps = helpers.ConstrainedOrRightDelimiterStep([], ["a", "b", "c"], 10)

    assert token == SpacedRightDelimiter
    assert steps == 9
    assert not lm.IsMasked(RightDelimiter)
    assert not lm.IsMasked(SpacedRightDelimiter)


def test_transpiled_helpers_declare_any_referenced_spaced_delimiter_constants():
    source = Path("generation/csd/VerifiedAgentSynthesis.py").read_text()

    result = transpile_contract_library(source, module_name_hint="VerifiedAgentSynthesis")

    assert result.is_ok()
    for constant_name in ("SpacedLeftDelimiter", "SpacedRightDelimiter"):
        if constant_name in result.value:
            assert f"const {constant_name}" in result.value


def test_transpiled_step_helpers_use_stable_return_names():
    source = Path("generation/csd/VerifiedAgentSynthesis.py").read_text()

    result = transpile_contract_library(source, module_name_hint="VerifiedAgentSynthesis")

    assert result.is_ok()
    assert "stepsLeft'" not in result.value
    assert "method UnconstrainedStep" in result.value
    assert "returns (nextToken: Token, remainingSteps: nat)" in result.value
    assert "ensures nextToken in lm.Tokens" in result.value


def test_helper_parser_wrappers_route_through_longest_valid_suffix():
    lm = SimpleLM(["a", "b", "c", "<<", ">>"])
    parser = SimpleParser()
    h = CSDHelpers(lm, parser)

    generated = ["free", "<<", "a", "b", "c"]

    assert h.IsComplete(generated)
    assert h.ValidContinuationCount(generated) == 0
    assert h.ParserDistanceToComplete(generated) == 0
    assert h.MinStepsToComplete(generated) == 0


def test_append_natural_left_delimiter_step_updates_prefix():
    lm = SimpleLM(["safe", RightDelimiter, SpacedRightDelimiter, "x", SpacedLeftDelimiter])
    parser = SimpleParser()
    h = CSDHelpers(lm, parser)

    generated, steps = h.AppendUnconstrainedNudgeLeftDelimiterStep([], [], 10)

    assert generated == [SpacedLeftDelimiter]
    assert h.EndsWithLeftDelimiter(generated)
    assert steps == 9


def test_append_right_delimiter_appends_exact_token():
    lm = SimpleLM(["a", "<<", ">>"])
    parser = SimpleParser()
    h = CSDHelpers(lm, parser)
    generated, steps = h.AppendRightDelimiter(["a"], 3)
    assert generated == ["a", ">>"]
    assert steps == 2


# ---------------------------------------------------------------------------
# CSDHelpers: end-to-end with delimiters
# ---------------------------------------------------------------------------

def test_full_strategy_produces_delimited_output():
    """Smoke test: simulate a strategy that emits << constrained >> output."""
    tokens = list(GRAMMAR_TOKENS) + [LeftDelimiter, RightDelimiter, "free"]
    lm = SimpleLM(tokens)
    parser = SimpleParser()
    h = CSDHelpers(lm, parser)

    generated: Prefix = []
    stepsLeft = 20

    # Free-form phase
    tok, stepsLeft = h.UnconstrainedStep([], generated, stepsLeft)
    generated = generated + [tok]

    # Emit <<
    tok, stepsLeft = h.ForcedTokenStep([], generated, LeftDelimiter, stepsLeft)
    generated = generated + [tok]

    # Constrained phase
    while stepsLeft > 1 and not parser.IsCompletePrefix(h.LongestValidSuffix(generated)):
        tok, stepsLeft = h.ConstrainedStep([], generated, stepsLeft)
        generated = generated + [tok]

    # Emit >>
    tok, stepsLeft = h.ForcedTokenStep([], generated, RightDelimiter, stepsLeft)
    generated = generated + [tok]

    output = "".join(generated)
    assert "<<" in output
    assert ">>" in output
    suffix = h.LongestValidSuffix(generated[:-1])  # suffix before >>
    assert parser.IsCompletePrefix(suffix)


def test_full_strategy_with_append_helpers_produces_delimited_output():
    tokens = list(GRAMMAR_TOKENS) + [LeftDelimiter, RightDelimiter, "free"]
    lm = SimpleLM(tokens)
    parser = SimpleParser()
    h = CSDHelpers(lm, parser)

    generated: Prefix = []
    stepsLeft = 20

    generated, stepsLeft = h.AppendUnconstrainedStep([], generated, stepsLeft)
    generated, stepsLeft = h.AppendLeftDelimiter(generated, stepsLeft)

    while stepsLeft > 1 and h.CanConstrain(generated):
        generated, stepsLeft = h.AppendConstrainedStep([], generated, stepsLeft)

    generated, stepsLeft = h.AppendRightDelimiter(generated, stepsLeft)

    output = "".join(generated)
    assert "<<" in output
    assert ">>" in output
    suffix = h.LongestValidSuffix(generated[:-1])
    assert parser.IsCompletePrefix(suffix)
