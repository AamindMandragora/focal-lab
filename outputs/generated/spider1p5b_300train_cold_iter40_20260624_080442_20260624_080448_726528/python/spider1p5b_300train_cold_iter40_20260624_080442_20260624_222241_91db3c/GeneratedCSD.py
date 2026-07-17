import sys
from typing import Callable, Any, TypeVar, NamedTuple
from math import floor
from itertools import count

import module_ as module_
import _dafny as _dafny
import System_ as System_
import VerifiedDecoderAgent as VerifiedDecoderAgent

# Module: GeneratedCSD

class default__:
    def  __init__(self):
        pass

    @staticmethod
    def MyCSDStrategy(lm, parser, prompt, generatedPrefix, insideConstrained, currentConstrained, maxSteps, stepTokenBudget, validTokenGroups, eosToken):
        generated: _dafny.Seq = _dafny.Seq({})
        insideConstrainedOut: bool = False
        currentConstrainedOut: _dafny.Seq = _dafny.Seq({})
        cost: int = int(0)
        out0_: _dafny.Seq
        out1_: bool
        out2_: _dafny.Seq
        out3_: int
        out0_, out1_, out2_, out3_ = default__.AuthorBody(lm, parser, prompt, generatedPrefix, insideConstrained, currentConstrained, maxSteps, stepTokenBudget, validTokenGroups, eosToken)
        generated = out0_
        insideConstrainedOut = out1_
        currentConstrainedOut = out2_
        cost = out3_
        if ((maxSteps) > (0)) and ((cost) <= (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

    @staticmethod
    def AuthorBody(lm, parser, prompt, generatedPrefix, insideConstrained, currentConstrained, maxSteps, stepTokenBudget, validTokenGroups, eosToken):
        generated: _dafny.Seq = _dafny.Seq({})
        insideConstrainedOut: bool = False
        currentConstrainedOut: _dafny.Seq = _dafny.Seq({})
        cost: int = int(0)
        d_0_helpers_: VerifiedDecoderAgent.CSDHelpers
        nw0_ = VerifiedDecoderAgent.CSDHelpers()
        nw0_.ctor__()
        d_0_helpers_ = nw0_
        generated = generatedPrefix
        insideConstrainedOut = insideConstrained
        currentConstrainedOut = currentConstrained
        cost = 0
        generated = generatedPrefix
        insideConstrainedOut = insideConstrained
        currentConstrainedOut = currentConstrained
        cost = 0
        d_1_steps_: int
        d_1_steps_ = 0
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate a complete and correct SQL query. Key rules: (1) COUNT rows: always use COUNT(*), never COUNT(column). (2) 'X that did both A and B' or 'visited both ... and ...': use INTERSECT of two complete SELECT queries with proper JOINs. (3) 'most/highest/largest' with grouping: GROUP BY col ORDER BY COUNT(*) DESC LIMIT 1. (4) Filtering by existence: use WHERE col IN (SELECT col FROM table WHERE condition). (5) Never use placeholder values like 'VALUE' or 'CODE' in WHERE clauses. (6) Always complete the full query with all required JOINs. Output only the SQL.")))
        d_2_freeLimit_: int
        d_2_freeLimit_ = _dafny.euclidian_division((maxSteps) * (5), 6)
        if ((d_2_freeLimit_) < (2)) and ((maxSteps) >= (2)):
            d_2_freeLimit_ = 2
        if ((d_2_freeLimit_) > ((maxSteps) - (2))) and ((maxSteps) >= (2)):
            d_2_freeLimit_ = (maxSteps) - (2)
        with _dafny.label("0"):
            while ((d_1_steps_) < (d_2_freeLimit_)) and (not(insideConstrainedOut)):
                with _dafny.c_label("0"):
                    d_3_next_: _dafny.Seq
                    out0_: _dafny.Seq
                    out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                    d_3_next_ = out0_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if (d_3_next_) == (eosToken):
                        raise _dafny.Break("0")
                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_3_next_]))
                    d_4_genStr_: _dafny.Seq
                    d_4_genStr_ = VerifiedDecoderAgent.CSDHelpers.PrefixToString(generated)
                    d_5_selectCountUpper_: int
                    d_5_selectCountUpper_ = VerifiedDecoderAgent.CSDHelpers.CountSubstring(d_4_genStr_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SELECT")))
                    d_6_selectCountLower_: int
                    d_6_selectCountLower_ = VerifiedDecoderAgent.CSDHelpers.CountSubstring(d_4_genStr_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "select")))
                    if ((d_5_selectCountUpper_) >= (5)) or ((d_6_selectCountLower_) >= (5)):
                        raise _dafny.Break("0")
                    pass
            pass
        if not(insideConstrainedOut):
            out1_: _dafny.Seq
            out2_: bool
            out3_: _dafny.Seq
            out1_, out2_, out3_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
            generated = out1_
            insideConstrainedOut = out2_
            currentConstrainedOut = out3_
        if (d_1_steps_) < (maxSteps):
            d_7_rem_: int
            d_7_rem_ = (maxSteps) - (d_1_steps_)
            d_8_fillBudget_: int
            d_8_fillBudget_ = _dafny.euclidian_division(d_7_rem_, 4)
            if (d_8_fillBudget_) >= (1):
                d_9_stable_: _dafny.Seq
                d_9_stable_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                d_10_filled_: _dafny.Seq
                out4_: _dafny.Seq
                out4_ = (d_0_helpers_).RegenerateUnitOnGroundingFailure(lm, parser, (prompt) + (d_9_stable_), currentConstrainedOut, eosToken, d_8_fillBudget_, 3, d_8_fillBudget_)
                d_10_filled_ = out4_
                generated = (d_9_stable_) + (d_10_filled_)
                currentConstrainedOut = d_10_filled_
                d_1_steps_ = (d_1_steps_) + (d_8_fillBudget_)
        if (d_1_steps_) < (maxSteps):
            d_11_closeBudget_: int
            d_11_closeBudget_ = (maxSteps) - (d_1_steps_)
            out5_: _dafny.Seq
            out6_: bool
            out7_: _dafny.Seq
            out5_, out6_, out7_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_11_closeBudget_)
            generated = out5_
            insideConstrainedOut = out6_
            currentConstrainedOut = out7_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

