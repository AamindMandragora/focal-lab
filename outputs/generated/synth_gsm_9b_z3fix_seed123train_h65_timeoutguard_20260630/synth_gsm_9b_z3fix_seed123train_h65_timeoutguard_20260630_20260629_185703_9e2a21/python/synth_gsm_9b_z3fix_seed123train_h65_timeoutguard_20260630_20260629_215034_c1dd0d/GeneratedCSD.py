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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math problem step by step using the given variable names. Write your final answer ONLY as <<expression>> where expression is a single Python-style arithmetic expression using ONLY the variable names from the problem (no curly braces), numbers, and operators: +, -, *, /, //, %, int(), **. Use int() to wrap the final result if it must be an integer. IMPORTANT: Write ONLY ONE final <<expression>> at the very end. Do not use {varname} syntax - write varname directly. Examples: <<n1 + n2>>, <<int(a * b / c)>>, <<n0 * (r + 1) ** d>>.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_spanSteps_: int
        d_2_spanSteps_ = 0
        d_3_spanBudget_: int
        d_3_spanBudget_ = 60
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_4_remainingBudget_: int
                        d_4_remainingBudget_ = (maxSteps) - (d_1_steps_)
                        if ((d_4_remainingBudget_) <= (80)) and ((d_4_remainingBudget_) > (3)):
                            d_5_og_: _dafny.Seq
                            d_6_oi_: bool
                            d_7_oc_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_5_og_ = out0_
                            d_6_oi_ = out1_
                            d_7_oc_ = out2_
                            generated = d_5_og_
                            insideConstrainedOut = d_6_oi_
                            currentConstrainedOut = d_7_oc_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_2_spanSteps_ = 0
                        elif True:
                            d_8_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_8_next_ = out3_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_8_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_8_next_]))
                                if (d_8_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    insideConstrainedOut = True
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                    d_2_spanSteps_ = 0
                    elif (d_2_spanSteps_) >= (d_3_spanBudget_):
                        d_9_remainingSteps_: int
                        d_9_remainingSteps_ = (maxSteps) - (d_1_steps_)
                        if (d_9_remainingSteps_) == (0):
                            raise _dafny.Break("0")
                        d_10_closeBudget2_: int
                        if (d_9_remainingSteps_) < (20):
                            d_10_closeBudget2_ = d_9_remainingSteps_
                        elif True:
                            d_10_closeBudget2_ = 20
                        d_11_cg2_: _dafny.Seq
                        d_12_ci2_: bool
                        d_13_cc2_: _dafny.Seq
                        out4_: _dafny.Seq
                        out5_: bool
                        out6_: _dafny.Seq
                        out4_, out5_, out6_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_10_closeBudget2_)
                        d_11_cg2_ = out4_
                        d_12_ci2_ = out5_
                        d_13_cc2_ = out6_
                        generated = d_11_cg2_
                        insideConstrainedOut = d_12_ci2_
                        currentConstrainedOut = d_13_cc2_
                        d_1_steps_ = (d_1_steps_) + (d_10_closeBudget2_)
                        d_2_spanSteps_ = 0
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
                        if d_17_closed_:
                            generated = d_14_cg_
                            insideConstrainedOut = d_15_ci_
                            currentConstrainedOut = d_16_cc_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_2_spanSteps_ = 0
                        elif True:
                            d_18_constrainedPrompt_: _dafny.Seq
                            d_18_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_19_nextSoft_: _dafny.Seq
                            d_20_softOk_: bool
                            out11_: _dafny.Seq
                            out12_: bool
                            out11_, out12_ = (d_0_helpers_).SafeSoftConstrainedStep(lm, parser, d_18_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('6e0'), eosToken)
                            d_19_nextSoft_ = out11_
                            d_20_softOk_ = out12_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_2_spanSteps_ = (d_2_spanSteps_) + (1)
                            if (d_19_nextSoft_) == (eosToken):
                                d_21_remainingSteps_: int
                                d_21_remainingSteps_ = (maxSteps) - (d_1_steps_)
                                if (d_21_remainingSteps_) == (0):
                                    raise _dafny.Break("0")
                                d_22_closeBudget3_: int
                                if (d_21_remainingSteps_) < (15):
                                    d_22_closeBudget3_ = d_21_remainingSteps_
                                elif True:
                                    d_22_closeBudget3_ = 15
                                d_23_cg3_: _dafny.Seq
                                d_24_ci3_: bool
                                d_25_cc3_: _dafny.Seq
                                out13_: _dafny.Seq
                                out14_: bool
                                out15_: _dafny.Seq
                                out13_, out14_, out15_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_22_closeBudget3_)
                                d_23_cg3_ = out13_
                                d_24_ci3_ = out14_
                                d_25_cc3_ = out15_
                                generated = d_23_cg3_
                                insideConstrainedOut = d_24_ci3_
                                currentConstrainedOut = d_25_cc3_
                                d_1_steps_ = (d_1_steps_) + (d_22_closeBudget3_)
                                raise _dafny.Break("0")
                            elif d_20_softOk_:
                                d_26_ag_: _dafny.Seq
                                d_27_ai_: bool
                                d_28_ac_: _dafny.Seq
                                out16_: _dafny.Seq
                                out17_: bool
                                out18_: _dafny.Seq
                                out16_, out17_, out18_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_19_nextSoft_)
                                d_26_ag_ = out16_
                                d_27_ai_ = out17_
                                d_28_ac_ = out18_
                                generated = d_26_ag_
                                insideConstrainedOut = d_27_ai_
                                currentConstrainedOut = d_28_ac_
                            elif True:
                                d_29_remainingSteps_: int
                                d_29_remainingSteps_ = (maxSteps) - (d_1_steps_)
                                if (d_29_remainingSteps_) == (0):
                                    raise _dafny.Break("0")
                                d_30_closeBudget4_: int
                                if (d_29_remainingSteps_) < (15):
                                    d_30_closeBudget4_ = d_29_remainingSteps_
                                elif True:
                                    d_30_closeBudget4_ = 15
                                d_31_cg4_: _dafny.Seq
                                d_32_ci4_: bool
                                d_33_cc4_: _dafny.Seq
                                out19_: _dafny.Seq
                                out20_: bool
                                out21_: _dafny.Seq
                                out19_, out20_, out21_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_30_closeBudget4_)
                                d_31_cg4_ = out19_
                                d_32_ci4_ = out20_
                                d_33_cc4_ = out21_
                                generated = d_31_cg4_
                                insideConstrainedOut = d_32_ci4_
                                currentConstrainedOut = d_33_cc4_
                                d_1_steps_ = (d_1_steps_) + (d_30_closeBudget4_)
                                d_2_spanSteps_ = 0
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

