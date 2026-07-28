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
        d_1_guidance_: _dafny.Seq
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate exactly one valid SQL query. Output format: SQL: <<YOUR QUERY>>. Use only tables and columns from the schema. No explanation, no markdown, no extra text."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_narrowThreshold_: int
        d_3_narrowThreshold_ = 12
        d_4_closeBudgetReserve_: int
        d_4_closeBudgetReserve_ = 250
        d_5_maxConstrainedTokens_: int
        d_5_maxConstrainedTokens_ = 200
        d_6_constrainedTokenCount_: int
        d_6_constrainedTokenCount_ = 0
        with _dafny.label("0"):
            while (d_2_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_7_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_7_next_ = out0_
                        d_2_steps_ = (d_2_steps_) + (1)
                        if (d_7_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_7_next_]))
                            if (d_7_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                d_6_constrainedTokenCount_ = 0
                    elif True:
                        d_8_remainingBudget_: int
                        d_8_remainingBudget_ = (maxSteps) - (d_2_steps_)
                        if ((d_8_remainingBudget_) <= (d_4_closeBudgetReserve_)) or ((d_6_constrainedTokenCount_) >= (d_5_maxConstrainedTokens_)):
                            d_9_closeBudget_: int
                            d_9_closeBudget_ = d_8_remainingBudget_
                            if (d_9_closeBudget_) > (0):
                                d_10_cg_: _dafny.Seq
                                d_11_ci_: bool
                                d_12_cc_: _dafny.Seq
                                out1_: _dafny.Seq
                                out2_: bool
                                out3_: _dafny.Seq
                                out1_, out2_, out3_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_9_closeBudget_)
                                d_10_cg_ = out1_
                                d_11_ci_ = out2_
                                d_12_cc_ = out3_
                                generated = d_10_cg_
                                insideConstrainedOut = d_11_ci_
                                currentConstrainedOut = d_12_cc_
                                d_2_steps_ = (d_2_steps_) + (d_9_closeBudget_)
                            elif True:
                                d_2_steps_ = maxSteps
                            raise _dafny.Break("0")
                        elif True:
                            d_13_cg_: _dafny.Seq
                            d_14_ci_: bool
                            d_15_cc_: _dafny.Seq
                            d_16_closed_: bool
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out7_: bool
                            out4_, out5_, out6_, out7_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                            d_13_cg_ = out4_
                            d_14_ci_ = out5_
                            d_15_cc_ = out6_
                            d_16_closed_ = out7_
                            d_2_steps_ = (d_2_steps_) + (1)
                            if d_16_closed_:
                                generated = d_13_cg_
                                insideConstrainedOut = d_14_ci_
                                currentConstrainedOut = d_15_cc_
                                raise _dafny.Break("0")
                            elif True:
                                d_17_constrainedPrompt_: _dafny.Seq
                                d_17_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_18_next_: _dafny.Seq
                                out8_: _dafny.Seq
                                out8_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_17_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_3_narrowThreshold_, eosToken)
                                d_18_next_ = out8_
                                if (d_18_next_) == (eosToken):
                                    d_19_remaining_: int
                                    d_19_remaining_ = (maxSteps) - (d_2_steps_)
                                    if (d_19_remaining_) > (0):
                                        d_20_cg2_: _dafny.Seq
                                        d_21_ci2_: bool
                                        d_22_cc2_: _dafny.Seq
                                        out9_: _dafny.Seq
                                        out10_: bool
                                        out11_: _dafny.Seq
                                        out9_, out10_, out11_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_19_remaining_)
                                        d_20_cg2_ = out9_
                                        d_21_ci2_ = out10_
                                        d_22_cc2_ = out11_
                                        generated = d_20_cg2_
                                        insideConstrainedOut = d_21_ci2_
                                        currentConstrainedOut = d_22_cc2_
                                        d_2_steps_ = (d_2_steps_) + (d_19_remaining_)
                                    raise _dafny.Break("0")
                                elif True:
                                    d_23_ag_: _dafny.Seq
                                    d_24_ai_: bool
                                    d_25_ac_: _dafny.Seq
                                    out12_: _dafny.Seq
                                    out13_: bool
                                    out14_: _dafny.Seq
                                    out12_, out13_, out14_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_18_next_)
                                    d_23_ag_ = out12_
                                    d_24_ai_ = out13_
                                    d_25_ac_ = out14_
                                    generated = d_23_ag_
                                    insideConstrainedOut = d_24_ai_
                                    currentConstrainedOut = d_25_ac_
                                    d_6_constrainedTokenCount_ = (d_6_constrainedTokenCount_) + (1)
                    pass
            pass
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

