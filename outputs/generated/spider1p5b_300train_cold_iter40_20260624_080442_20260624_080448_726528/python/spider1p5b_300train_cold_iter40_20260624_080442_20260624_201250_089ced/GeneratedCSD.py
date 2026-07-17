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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Write exactly one SQL query. STRICT RULES: (1) For 'in ascending/descending order': use ORDER BY col ASC or ORDER BY col DESC. NEVER use WHERE col IN (SELECT col FROM ...) for ordering. (2) For 'both condition A and condition B' where a record must satisfy both at different times: use INTERSECT of two full SELECT...JOIN...WHERE queries. (3) For 'which has the most': SELECT col FROM t GROUP BY col ORDER BY COUNT(*) DESC LIMIT 1. (4) For 'how many per group': SELECT col, COUNT(*) FROM t GROUP BY col. (5) For 'at least N': HAVING COUNT(*) >= N. (6) Use the simplest possible SQL: avoid subqueries when JOIN or GROUP BY works. (7) Do not add WHERE conditions that reference tables not in FROM. Output the SQL query only.")))
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
                    d_8_inSelectUpper_: int
                    d_8_inSelectUpper_ = VerifiedDecoderAgent.CSDHelpers.CountSubstring(d_5_genStr_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " IN (SELECT")))
                    d_9_inSelectLower_: int
                    d_9_inSelectLower_ = VerifiedDecoderAgent.CSDHelpers.CountSubstring(d_5_genStr_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " in (select")))
                    if ((d_8_inSelectUpper_) >= (2)) or ((d_9_inSelectLower_) >= (2)):
                        raise _dafny.Break("0")
                    d_10_whereCountUpper_: int
                    d_10_whereCountUpper_ = VerifiedDecoderAgent.CSDHelpers.CountSubstring(d_5_genStr_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " WHERE ")))
                    d_11_whereCountLower_: int
                    d_11_whereCountLower_ = VerifiedDecoderAgent.CSDHelpers.CountSubstring(d_5_genStr_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " where ")))
                    if ((d_10_whereCountUpper_) >= (3)) or ((d_11_whereCountLower_) >= (3)):
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
            d_12_rem_: int
            d_12_rem_ = (maxSteps) - (d_1_steps_)
            d_13_fillBudget_: int
            d_13_fillBudget_ = _dafny.euclidian_division(d_12_rem_, 2)
            if (d_13_fillBudget_) >= (1):
                d_14_stable_: _dafny.Seq
                d_14_stable_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                d_15_filled_: _dafny.Seq
                out4_: _dafny.Seq
                out4_ = (d_0_helpers_).RegenerateUnitOnGroundingFailure(lm, parser, (prompt) + (d_14_stable_), currentConstrainedOut, eosToken, d_13_fillBudget_, 3, d_13_fillBudget_)
                d_15_filled_ = out4_
                generated = (d_14_stable_) + (d_15_filled_)
                currentConstrainedOut = d_15_filled_
                d_1_steps_ = (d_1_steps_) + (d_13_fillBudget_)
        if (d_1_steps_) < (maxSteps):
            d_16_closeBudget_: int
            d_16_closeBudget_ = (maxSteps) - (d_1_steps_)
            out5_: _dafny.Seq
            out6_: bool
            out7_: _dafny.Seq
            out5_, out6_, out7_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_16_closeBudget_)
            generated = out5_
            insideConstrainedOut = out6_
            currentConstrainedOut = out7_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

