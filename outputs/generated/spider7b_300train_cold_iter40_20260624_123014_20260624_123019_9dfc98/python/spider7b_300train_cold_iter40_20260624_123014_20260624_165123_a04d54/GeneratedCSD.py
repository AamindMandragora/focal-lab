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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate complete and accurate SQL for Spider benchmark. CRITICAL RULES: (1) Always use explicit JOIN ... ON syntax when combining multiple tables. (2) When counting groups, always include GROUP BY and HAVING clauses. (3) For 'both X and Y' queries use INTERSECT between two SELECT statements. (4) For 'not in' comparisons use NOT IN with a subquery. (5) For 'maximum/minimum/largest/smallest' use ORDER BY col DESC/ASC LIMIT 1 within a subquery. (6) NEVER use table aliases or AS keyword for table/column aliases. (7) Generate the full complete SQL including all necessary JOINs, WHERE, GROUP BY, HAVING, and subquery clauses.")))
        if not(insideConstrainedOut):
            out0_: _dafny.Seq
            out1_: bool
            out2_: _dafny.Seq
            out0_, out1_, out2_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
            generated = out0_
            insideConstrainedOut = out1_
            currentConstrainedOut = out2_
        if (d_1_steps_) < (maxSteps):
            d_2_rem_: int
            d_2_rem_ = (maxSteps) - (d_1_steps_)
            d_3_fillBudget1_: int
            d_3_fillBudget1_ = _dafny.euclidian_division(d_2_rem_, 5)
            if (d_3_fillBudget1_) >= (1):
                d_4_stable_: _dafny.Seq
                d_4_stable_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                d_5_filled_: _dafny.Seq
                out3_: _dafny.Seq
                out3_ = (d_0_helpers_).RegenerateUnitOnGroundingFailure(lm, parser, (prompt) + (d_4_stable_), currentConstrainedOut, eosToken, d_3_fillBudget1_, 3, 5)
                d_5_filled_ = out3_
                generated = (d_4_stable_) + (d_5_filled_)
                currentConstrainedOut = d_5_filled_
                d_1_steps_ = (d_1_steps_) + (d_3_fillBudget1_)
        if (d_1_steps_) < (maxSteps):
            d_6_rem2_: int
            d_6_rem2_ = (maxSteps) - (d_1_steps_)
            d_7_fillBudget2_: int
            d_7_fillBudget2_ = _dafny.euclidian_division((d_6_rem2_) * (3), 5)
            d_8_loopSteps_: int
            d_8_loopSteps_ = 0
            with _dafny.label("2_0"):
                while (d_8_loopSteps_) < (d_7_fillBudget2_):
                    with _dafny.c_label("2_0"):
                        d_9_isComplete_: bool
                        d_9_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_9_isComplete_:
                            raise _dafny.Break("2_0")
                        d_10_stable2_: _dafny.Seq
                        d_10_stable2_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_11_constrainedPrompt2_: _dafny.Seq
                        d_11_constrainedPrompt2_ = (prompt) + (d_10_stable2_)
                        d_12_penaltyTokens_: _dafny.Seq
                        d_12_penaltyTokens_ = _dafny.SeqWithoutIsStrInference([eosToken, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "AS")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " AS")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " AS ")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "as")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " as")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " as "))])
                        d_13_nextAdapt_: _dafny.Seq
                        out4_: _dafny.Seq
                        out4_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_11_constrainedPrompt2_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_12_penaltyTokens_, _dafny.BigRational('4e0'), 12, eosToken)
                        d_13_nextAdapt_ = out4_
                        d_8_loopSteps_ = (d_8_loopSteps_) + (1)
                        if (d_13_nextAdapt_) == (eosToken):
                            raise _dafny.Break("2_0")
                        d_14_isCompleteAfter_: bool
                        d_14_isCompleteAfter_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_14_isCompleteAfter_:
                            raise _dafny.Break("2_0")
                        d_15_ng_: _dafny.Seq
                        d_16_ni_: bool
                        d_17_nc_: _dafny.Seq
                        out5_: _dafny.Seq
                        out6_: bool
                        out7_: _dafny.Seq
                        out5_, out6_, out7_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_13_nextAdapt_)
                        d_15_ng_ = out5_
                        d_16_ni_ = out6_
                        d_17_nc_ = out7_
                        generated = d_15_ng_
                        insideConstrainedOut = d_16_ni_
                        currentConstrainedOut = d_17_nc_
                        pass
                pass
            d_1_steps_ = (d_1_steps_) + (d_7_fillBudget2_)
        if (d_1_steps_) < (maxSteps):
            d_18_closeBudget_: int
            d_18_closeBudget_ = (maxSteps) - (d_1_steps_)
            out8_: _dafny.Seq
            out9_: bool
            out10_: _dafny.Seq
            out8_, out9_, out10_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_18_closeBudget_)
            generated = out8_
            insideConstrainedOut = out9_
            currentConstrainedOut = out10_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

