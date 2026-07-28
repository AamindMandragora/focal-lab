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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate exactly one minimal SQL query. Use only the columns and tables from the schema. Do not add columns not asked for. Do not repeat conditions.")))
        if not(insideConstrainedOut):
            out0_: _dafny.Seq
            out1_: bool
            out2_: _dafny.Seq
            out0_, out1_, out2_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
            generated = out0_
            insideConstrainedOut = out1_
            currentConstrainedOut = out2_
        d_2_closingReserve_: int
        d_2_closingReserve_ = _dafny.euclidian_division(maxSteps, 5)
        if ((d_2_closingReserve_) < (5)) and ((maxSteps) >= (5)):
            d_2_closingReserve_ = 5
        if (d_2_closingReserve_) > (40):
            d_2_closingReserve_ = 40
        d_3_genBudget_: int
        if (maxSteps) > (d_2_closingReserve_):
            d_3_genBudget_ = (maxSteps) - (d_2_closingReserve_)
        elif True:
            d_3_genBudget_ = maxSteps
        with _dafny.label("0"):
            while ((d_1_steps_) < (d_3_genBudget_)) and (insideConstrainedOut):
                with _dafny.c_label("0"):
                    if (parser).IsCompletePrefix(currentConstrainedOut):
                        d_4_cg_: _dafny.Seq
                        d_5_ci_: bool
                        d_6_cc_: _dafny.Seq
                        d_7_closed_: bool
                        out3_: _dafny.Seq
                        out4_: bool
                        out5_: _dafny.Seq
                        out6_: bool
                        out3_, out4_, out5_, out6_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                        d_4_cg_ = out3_
                        d_5_ci_ = out4_
                        d_6_cc_ = out5_
                        d_7_closed_ = out6_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if d_7_closed_:
                            generated = d_4_cg_
                            insideConstrainedOut = d_5_ci_
                            currentConstrainedOut = d_6_cc_
                        elif True:
                            d_8_constrainedPrompt_: _dafny.Seq
                            d_8_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_9_next_: _dafny.Seq
                            out7_: _dafny.Seq
                            out7_ = (d_0_helpers_).RepetitionPenaltyStep(lm, parser, d_8_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('3e0'), eosToken)
                            d_9_next_ = out7_
                            if (d_9_next_) == (eosToken):
                                raise _dafny.Break("0")
                            d_10_ag_: _dafny.Seq
                            d_11_ai_: bool
                            d_12_ac_: _dafny.Seq
                            out8_: _dafny.Seq
                            out9_: bool
                            out10_: _dafny.Seq
                            out8_, out9_, out10_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_9_next_)
                            d_10_ag_ = out8_
                            d_11_ai_ = out9_
                            d_12_ac_ = out10_
                            generated = d_10_ag_
                            insideConstrainedOut = d_11_ai_
                            currentConstrainedOut = d_12_ac_
                    elif True:
                        d_13_constrainedPrompt_: _dafny.Seq
                        d_13_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_14_next_: _dafny.Seq
                        out11_: _dafny.Seq
                        out11_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_13_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                        d_14_next_ = out11_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_14_next_) == (eosToken):
                            raise _dafny.Break("0")
                        d_15_ag_: _dafny.Seq
                        d_16_ai_: bool
                        d_17_ac_: _dafny.Seq
                        out12_: _dafny.Seq
                        out13_: bool
                        out14_: _dafny.Seq
                        out12_, out13_, out14_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_14_next_)
                        d_15_ag_ = out12_
                        d_16_ai_ = out13_
                        d_17_ac_ = out14_
                        generated = d_15_ag_
                        insideConstrainedOut = d_16_ai_
                        currentConstrainedOut = d_17_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_18_closeBudget_: int
            d_18_closeBudget_ = (maxSteps) - (d_1_steps_)
            out15_: _dafny.Seq
            out16_: bool
            out17_: _dafny.Seq
            out15_, out16_, out17_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_18_closeBudget_)
            generated = out15_
            insideConstrainedOut = out16_
            currentConstrainedOut = out17_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

