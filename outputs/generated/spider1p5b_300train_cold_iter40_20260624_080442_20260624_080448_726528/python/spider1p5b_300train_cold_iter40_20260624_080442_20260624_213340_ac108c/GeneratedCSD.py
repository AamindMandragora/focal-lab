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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Write one SQL query answering the question. CRITICAL PATTERNS: (1) For 'both X and Y' or 'satisfies two separate conditions across different records': use INTERSECT of two SELECT...JOIN...WHERE queries. (2) For 'which has the most/maximum' or 'model with most versions' or 'highest count': use SELECT col FROM table GROUP BY col ORDER BY COUNT(*) DESC LIMIT 1. NEVER use nested WHERE col IN (SELECT...) for finding maximums. (3) For 'at least N records': GROUP BY col HAVING COUNT(*) >= N. (4) For questions asking for names/values WITH an aggregate (e.g. 'first names and their total X'): SELECT name, AGG(col) FROM table JOIN ... GROUP BY name. Do NOT add WHERE clauses filtering by year, date, or any condition NOT explicitly stated in the question. (5) Output only the SQL query.")))
        d_2_freeLimit_: int
        d_2_freeLimit_ = _dafny.euclidian_division((maxSteps) * (4), 5)
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
                    d_4_glen_: int
                    d_4_glen_ = len(generated)
                    if (d_4_glen_) >= ((len(generatedPrefix)) + (12)):
                        if (_dafny.SeqWithoutIsStrInference((generated)[(d_4_glen_) - (6):d_4_glen_:])) == (_dafny.SeqWithoutIsStrInference((generated)[(d_4_glen_) - (12):(d_4_glen_) - (6):])):
                            raise _dafny.Break("0")
                    if (d_4_glen_) >= ((len(generatedPrefix)) + (10)):
                        if (_dafny.SeqWithoutIsStrInference((generated)[(d_4_glen_) - (5):d_4_glen_:])) == (_dafny.SeqWithoutIsStrInference((generated)[(d_4_glen_) - (10):(d_4_glen_) - (5):])):
                            raise _dafny.Break("0")
                    if (d_4_glen_) >= ((len(generatedPrefix)) + (8)):
                        if (_dafny.SeqWithoutIsStrInference((generated)[(d_4_glen_) - (4):d_4_glen_:])) == (_dafny.SeqWithoutIsStrInference((generated)[(d_4_glen_) - (8):(d_4_glen_) - (4):])):
                            raise _dafny.Break("0")
                    if (d_4_glen_) >= ((len(generatedPrefix)) + (6)):
                        if (_dafny.SeqWithoutIsStrInference((generated)[(d_4_glen_) - (3):d_4_glen_:])) == (_dafny.SeqWithoutIsStrInference((generated)[(d_4_glen_) - (6):(d_4_glen_) - (3):])):
                            raise _dafny.Break("0")
                    d_5_genStr_: _dafny.Seq
                    d_5_genStr_ = VerifiedDecoderAgent.CSDHelpers.PrefixToString(generated)
                    d_6_selectCountUpper_: int
                    d_6_selectCountUpper_ = VerifiedDecoderAgent.CSDHelpers.CountSubstring(d_5_genStr_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SELECT")))
                    d_7_selectCountLower_: int
                    d_7_selectCountLower_ = VerifiedDecoderAgent.CSDHelpers.CountSubstring(d_5_genStr_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "select")))
                    if ((d_6_selectCountUpper_) >= (5)) or ((d_7_selectCountLower_) >= (5)):
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
            d_8_rem_: int
            d_8_rem_ = (maxSteps) - (d_1_steps_)
            d_9_fillBudget_: int
            d_9_fillBudget_ = _dafny.euclidian_division(d_8_rem_, 3)
            if (d_9_fillBudget_) >= (1):
                d_10_stable_: _dafny.Seq
                d_10_stable_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                d_11_filled_: _dafny.Seq
                out4_: _dafny.Seq
                out4_ = (d_0_helpers_).RegenerateUnitOnGroundingFailure(lm, parser, (prompt) + (d_10_stable_), currentConstrainedOut, eosToken, d_9_fillBudget_, 3, d_9_fillBudget_)
                d_11_filled_ = out4_
                generated = (d_10_stable_) + (d_11_filled_)
                currentConstrainedOut = d_11_filled_
                d_1_steps_ = (d_1_steps_) + (d_9_fillBudget_)
        if (d_1_steps_) < (maxSteps):
            d_12_closeBudget_: int
            d_12_closeBudget_ = (maxSteps) - (d_1_steps_)
            out5_: _dafny.Seq
            out6_: bool
            out7_: _dafny.Seq
            out5_, out6_, out7_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_12_closeBudget_)
            generated = out5_
            insideConstrainedOut = out6_
            currentConstrainedOut = out7_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

