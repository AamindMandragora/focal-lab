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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve this math problem step by step. Reason through the problem. At the end of your reasoning, write your final symbolic answer ONCE inside << >> using variable names (no curly braces), numbers, and operators +, -, *, /, //, %, int(). Write exactly one << >> at the very end. Stop after closing >>."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_prefixBudget_: int
        if (maxSteps) > (200):
            d_3_prefixBudget_ = (maxSteps) - (60)
        elif True:
            if (maxSteps) > (80):
                d_3_prefixBudget_ = (maxSteps) - (30)
            elif True:
                d_3_prefixBudget_ = _dafny.euclidian_division(maxSteps, 2)
        with _dafny.label("0"):
            while (d_2_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if ((d_2_steps_) >= (d_3_prefixBudget_)) and (((maxSteps) - (d_2_steps_)) > (3)):
                            d_4_og_: _dafny.Seq
                            d_5_oi_: bool
                            d_6_oc_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_4_og_ = out0_
                            d_5_oi_ = out1_
                            d_6_oc_ = out2_
                            generated = d_4_og_
                            insideConstrainedOut = d_5_oi_
                            currentConstrainedOut = d_6_oc_
                            d_2_steps_ = (d_2_steps_) + (1)
                        elif True:
                            d_7_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_7_next_ = out3_
                            d_2_steps_ = (d_2_steps_) + (1)
                            if (d_7_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif (d_7_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))):
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_7_next_]))
                                raise _dafny.Break("0")
                            elif (d_7_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_7_next_]))
                                d_8_eg_: _dafny.Seq
                                d_9_ei_: bool
                                d_10_ec_: _dafny.Seq
                                out4_: _dafny.Seq
                                out5_: bool
                                out6_: _dafny.Seq
                                out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_8_eg_ = out4_
                                d_9_ei_ = out5_
                                d_10_ec_ = out6_
                                generated = d_8_eg_
                                insideConstrainedOut = d_9_ei_
                                currentConstrainedOut = d_10_ec_
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_7_next_]))
                    elif True:
                        d_11_remainingBudget_: int
                        d_11_remainingBudget_ = (maxSteps) - (d_2_steps_)
                        if ((d_11_remainingBudget_) <= (4)) and ((d_11_remainingBudget_) > (0)):
                            d_12_sg_: _dafny.Seq
                            d_13_si_: bool
                            d_14_sc_: _dafny.Seq
                            out7_: _dafny.Seq
                            out8_: bool
                            out9_: _dafny.Seq
                            out7_, out8_, out9_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_11_remainingBudget_)
                            d_12_sg_ = out7_
                            d_13_si_ = out8_
                            d_14_sc_ = out9_
                            generated = d_12_sg_
                            insideConstrainedOut = d_13_si_
                            currentConstrainedOut = d_14_sc_
                            d_2_steps_ = (d_2_steps_) + (d_11_remainingBudget_)
                        elif True:
                            d_15_cg_: _dafny.Seq
                            d_16_ci_: bool
                            d_17_cc_: _dafny.Seq
                            d_18_closed_: bool
                            out10_: _dafny.Seq
                            out11_: bool
                            out12_: _dafny.Seq
                            out13_: bool
                            out10_, out11_, out12_, out13_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                            d_15_cg_ = out10_
                            d_16_ci_ = out11_
                            d_17_cc_ = out12_
                            d_18_closed_ = out13_
                            d_2_steps_ = (d_2_steps_) + (1)
                            if d_18_closed_:
                                generated = d_15_cg_
                                insideConstrainedOut = d_16_ci_
                                currentConstrainedOut = d_17_cc_
                                raise _dafny.Break("0")
                            elif True:
                                d_19_constrainedPrompt_: _dafny.Seq
                                d_19_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_20_next_: _dafny.Seq
                                out14_: _dafny.Seq
                                out14_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_19_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('3e0'), eosToken)
                                d_20_next_ = out14_
                                if (d_20_next_) == (eosToken):
                                    d_21_rem_: int
                                    d_21_rem_ = (maxSteps) - (d_2_steps_)
                                    if (d_21_rem_) > (0):
                                        d_22_sg2_: _dafny.Seq
                                        d_23_si2_: bool
                                        d_24_sc2_: _dafny.Seq
                                        out15_: _dafny.Seq
                                        out16_: bool
                                        out17_: _dafny.Seq
                                        out15_, out16_, out17_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_21_rem_)
                                        d_22_sg2_ = out15_
                                        d_23_si2_ = out16_
                                        d_24_sc2_ = out17_
                                        generated = d_22_sg2_
                                        insideConstrainedOut = d_23_si2_
                                        currentConstrainedOut = d_24_sc2_
                                        d_2_steps_ = (d_2_steps_) + (d_21_rem_)
                                    raise _dafny.Break("0")
                                elif True:
                                    d_25_ag_: _dafny.Seq
                                    d_26_ai_: bool
                                    d_27_ac_: _dafny.Seq
                                    out18_: _dafny.Seq
                                    out19_: bool
                                    out20_: _dafny.Seq
                                    out18_, out19_, out20_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_20_next_)
                                    d_25_ag_ = out18_
                                    d_26_ai_ = out19_
                                    d_27_ac_ = out20_
                                    generated = d_25_ag_
                                    insideConstrainedOut = d_26_ai_
                                    currentConstrainedOut = d_27_ac_
                    pass
            pass
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

