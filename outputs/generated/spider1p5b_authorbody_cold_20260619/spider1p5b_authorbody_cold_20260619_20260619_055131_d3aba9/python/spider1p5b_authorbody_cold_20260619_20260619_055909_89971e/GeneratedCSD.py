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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Output exactly: SQL: <<YOUR SQL QUERY HERE>> with no other text. The SQL query must be inside << and >>. Example: SQL: <<SELECT col FROM tbl WHERE cond>>. Do not add semicolons inside the span. Use only schema tables and columns."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_narrowThreshold_: int
        d_3_narrowThreshold_ = 20
        d_4_unconstrainedStepsTaken_: int
        d_4_unconstrainedStepsTaken_ = 0
        d_5_maxUnconstrainedSteps_: int
        d_5_maxUnconstrainedSteps_ = 8
        with _dafny.label("0"):
            while (d_2_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if ((d_4_unconstrainedStepsTaken_) >= (d_5_maxUnconstrainedSteps_)) and (((d_2_steps_) + (1)) <= (maxSteps)):
                            d_6_og_: _dafny.Seq
                            d_7_oi_: bool
                            d_8_oc_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_6_og_ = out0_
                            d_7_oi_ = out1_
                            d_8_oc_ = out2_
                            generated = d_6_og_
                            insideConstrainedOut = d_7_oi_
                            currentConstrainedOut = d_8_oc_
                            d_2_steps_ = (d_2_steps_) + (1)
                            d_4_unconstrainedStepsTaken_ = 0
                        elif True:
                            d_9_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_9_next_ = out3_
                            d_2_steps_ = (d_2_steps_) + (1)
                            d_4_unconstrainedStepsTaken_ = (d_4_unconstrainedStepsTaken_) + (1)
                            if (d_9_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_9_next_]))
                                if (d_9_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    insideConstrainedOut = True
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                    d_4_unconstrainedStepsTaken_ = 0
                    elif True:
                        d_10_remainingBudget_: int
                        d_10_remainingBudget_ = (maxSteps) - (d_2_steps_)
                        if (d_10_remainingBudget_) <= (5):
                            d_11_cg_: _dafny.Seq
                            d_12_ci_: bool
                            d_13_cc_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_10_remainingBudget_)
                            d_11_cg_ = out4_
                            d_12_ci_ = out5_
                            d_13_cc_ = out6_
                            generated = d_11_cg_
                            insideConstrainedOut = d_12_ci_
                            currentConstrainedOut = d_13_cc_
                            d_2_steps_ = maxSteps
                        elif True:
                            d_14_cg_: _dafny.Seq
                            d_15_ci_: bool
                            d_16_cc_: _dafny.Seq
                            d_17_closed_: bool
                            out7_: _dafny.Seq
                            out8_: bool
                            out9_: _dafny.Seq
                            out10_: bool
                            out7_, out8_, out9_, out10_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                            d_14_cg_ = out7_
                            d_15_ci_ = out8_
                            d_16_cc_ = out9_
                            d_17_closed_ = out10_
                            d_2_steps_ = (d_2_steps_) + (1)
                            if d_17_closed_:
                                generated = d_14_cg_
                                insideConstrainedOut = d_15_ci_
                                currentConstrainedOut = d_16_cc_
                                raise _dafny.Break("0")
                            elif True:
                                d_18_constrainedPrompt_: _dafny.Seq
                                d_18_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_19_next_: _dafny.Seq
                                out11_: _dafny.Seq
                                out11_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_18_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('5e0'), d_3_narrowThreshold_, eosToken)
                                d_19_next_ = out11_
                                if (d_19_next_) == (eosToken):
                                    d_20_closeBudget_: int
                                    d_20_closeBudget_ = (maxSteps) - (d_2_steps_)
                                    if (d_20_closeBudget_) > (0):
                                        d_21_cg2_: _dafny.Seq
                                        d_22_ci2_: bool
                                        d_23_cc2_: _dafny.Seq
                                        out12_: _dafny.Seq
                                        out13_: bool
                                        out14_: _dafny.Seq
                                        out12_, out13_, out14_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_20_closeBudget_)
                                        d_21_cg2_ = out12_
                                        d_22_ci2_ = out13_
                                        d_23_cc2_ = out14_
                                        generated = d_21_cg2_
                                        insideConstrainedOut = d_22_ci2_
                                        currentConstrainedOut = d_23_cc2_
                                        d_2_steps_ = maxSteps
                                    raise _dafny.Break("0")
                                elif True:
                                    d_24_ag_: _dafny.Seq
                                    d_25_ai_: bool
                                    d_26_ac_: _dafny.Seq
                                    out15_: _dafny.Seq
                                    out16_: bool
                                    out17_: _dafny.Seq
                                    out15_, out16_, out17_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_19_next_)
                                    d_24_ag_ = out15_
                                    d_25_ai_ = out16_
                                    d_26_ac_ = out17_
                                    generated = d_24_ag_
                                    insideConstrainedOut = d_25_ai_
                                    currentConstrainedOut = d_26_ac_
                    pass
            pass
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

