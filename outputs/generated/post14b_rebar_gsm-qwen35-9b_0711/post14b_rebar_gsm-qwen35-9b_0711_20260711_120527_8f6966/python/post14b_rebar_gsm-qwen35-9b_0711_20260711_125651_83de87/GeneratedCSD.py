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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve this math problem step by step using only text and arithmetic. At the very end, place the single final numeric expression inside << >>. Do not use << >> for intermediate steps. Write the complete formula once as the final answer, e.g. <<int(n * p / 100)>>."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_narrowThreshold_: int
        d_3_narrowThreshold_ = 16
        d_4_chunkSize_: int
        d_4_chunkSize_ = 32
        d_5_minSpanTokens_: int
        d_5_minSpanTokens_ = 3
        with _dafny.label("0"):
            while (d_2_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_6_remainingSteps_: int
                        d_6_remainingSteps_ = (maxSteps) - (d_2_steps_)
                        d_7_actualChunk_: int
                        if (d_6_remainingSteps_) < (d_4_chunkSize_):
                            d_7_actualChunk_ = d_6_remainingSteps_
                        elif True:
                            d_7_actualChunk_ = d_4_chunkSize_
                        if (d_7_actualChunk_) == (0):
                            raise _dafny.Break("0")
                        d_8_generatedOut_: _dafny.Seq
                        d_9_stoppedOnOpenSpan_: bool
                        d_10_stoppedOnEos_: bool
                        d_11_stepsUsed_: int
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: bool
                        out3_: int
                        out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_7_actualChunk_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_8_generatedOut_ = out0_
                        d_9_stoppedOnOpenSpan_ = out1_
                        d_10_stoppedOnEos_ = out2_
                        d_11_stepsUsed_ = out3_
                        d_2_steps_ = (d_2_steps_) + (d_11_stepsUsed_)
                        generated = d_8_generatedOut_
                        if d_10_stoppedOnEos_:
                            raise _dafny.Break("0")
                        elif d_9_stoppedOnOpenSpan_:
                            d_12_g2_: _dafny.Seq
                            d_13_i2_: bool
                            d_14_c2_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                            d_12_g2_ = out4_
                            d_13_i2_ = out5_
                            d_14_c2_ = out6_
                            generated = d_12_g2_
                            insideConstrainedOut = d_13_i2_
                            currentConstrainedOut = d_14_c2_
                    elif True:
                        d_15_remainingBudget_: int
                        d_15_remainingBudget_ = (maxSteps) - (d_2_steps_)
                        if ((d_15_remainingBudget_) <= (25)) and ((d_15_remainingBudget_) > (0)):
                            d_16_sg_: _dafny.Seq
                            d_17_si_: bool
                            d_18_sc_: _dafny.Seq
                            out7_: _dafny.Seq
                            out8_: bool
                            out9_: _dafny.Seq
                            out7_, out8_, out9_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_15_remainingBudget_)
                            d_16_sg_ = out7_
                            d_17_si_ = out8_
                            d_18_sc_ = out9_
                            generated = d_16_sg_
                            insideConstrainedOut = d_17_si_
                            currentConstrainedOut = d_18_sc_
                            d_2_steps_ = (d_2_steps_) + (d_15_remainingBudget_)
                        elif True:
                            d_19_spanLen_: int
                            d_19_spanLen_ = len(currentConstrainedOut)
                            if ((d_19_spanLen_) >= (d_5_minSpanTokens_)) and ((parser).IsCompletePrefix(currentConstrainedOut)):
                                d_20_cg_: _dafny.Seq
                                d_21_ci_: bool
                                d_22_cc_: _dafny.Seq
                                d_23_closed_: bool
                                out10_: _dafny.Seq
                                out11_: bool
                                out12_: _dafny.Seq
                                out13_: bool
                                out10_, out11_, out12_, out13_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                                d_20_cg_ = out10_
                                d_21_ci_ = out11_
                                d_22_cc_ = out12_
                                d_23_closed_ = out13_
                                d_2_steps_ = (d_2_steps_) + (1)
                                if d_23_closed_:
                                    generated = d_20_cg_
                                    insideConstrainedOut = d_21_ci_
                                    currentConstrainedOut = d_22_cc_
                                elif True:
                                    d_24_constrainedPrompt_: _dafny.Seq
                                    d_24_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                    d_25_next_: _dafny.Seq
                                    out14_: _dafny.Seq
                                    out14_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_24_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_3_narrowThreshold_, eosToken)
                                    d_25_next_ = out14_
                                    if (d_25_next_) == (eosToken):
                                        d_26_remainingBudget2_: int
                                        d_26_remainingBudget2_ = (maxSteps) - (d_2_steps_)
                                        if (d_26_remainingBudget2_) > (0):
                                            d_27_sg2_: _dafny.Seq
                                            d_28_si2_: bool
                                            d_29_sc2_: _dafny.Seq
                                            out15_: _dafny.Seq
                                            out16_: bool
                                            out17_: _dafny.Seq
                                            out15_, out16_, out17_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_26_remainingBudget2_)
                                            d_27_sg2_ = out15_
                                            d_28_si2_ = out16_
                                            d_29_sc2_ = out17_
                                            generated = d_27_sg2_
                                            insideConstrainedOut = d_28_si2_
                                            currentConstrainedOut = d_29_sc2_
                                            d_2_steps_ = (d_2_steps_) + (d_26_remainingBudget2_)
                                        raise _dafny.Break("0")
                                    elif True:
                                        d_30_ag_: _dafny.Seq
                                        d_31_ai_: bool
                                        d_32_ac_: _dafny.Seq
                                        out18_: _dafny.Seq
                                        out19_: bool
                                        out20_: _dafny.Seq
                                        out18_, out19_, out20_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_25_next_)
                                        d_30_ag_ = out18_
                                        d_31_ai_ = out19_
                                        d_32_ac_ = out20_
                                        generated = d_30_ag_
                                        insideConstrainedOut = d_31_ai_
                                        currentConstrainedOut = d_32_ac_
                            elif True:
                                d_33_constrainedPrompt_: _dafny.Seq
                                d_33_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_34_next_: _dafny.Seq
                                out21_: _dafny.Seq
                                out21_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_33_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_3_narrowThreshold_, eosToken)
                                d_34_next_ = out21_
                                d_2_steps_ = (d_2_steps_) + (1)
                                if (d_34_next_) == (eosToken):
                                    d_35_remainingBudget2_: int
                                    d_35_remainingBudget2_ = (maxSteps) - (d_2_steps_)
                                    if (d_35_remainingBudget2_) > (0):
                                        d_36_sg2_: _dafny.Seq
                                        d_37_si2_: bool
                                        d_38_sc2_: _dafny.Seq
                                        out22_: _dafny.Seq
                                        out23_: bool
                                        out24_: _dafny.Seq
                                        out22_, out23_, out24_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_35_remainingBudget2_)
                                        d_36_sg2_ = out22_
                                        d_37_si2_ = out23_
                                        d_38_sc2_ = out24_
                                        generated = d_36_sg2_
                                        insideConstrainedOut = d_37_si2_
                                        currentConstrainedOut = d_38_sc2_
                                        d_2_steps_ = (d_2_steps_) + (d_35_remainingBudget2_)
                                    raise _dafny.Break("0")
                                elif True:
                                    d_39_ag_: _dafny.Seq
                                    d_40_ai_: bool
                                    d_41_ac_: _dafny.Seq
                                    out25_: _dafny.Seq
                                    out26_: bool
                                    out27_: _dafny.Seq
                                    out25_, out26_, out27_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_34_next_)
                                    d_39_ag_ = out25_
                                    d_40_ai_ = out26_
                                    d_41_ac_ = out27_
                                    generated = d_39_ag_
                                    insideConstrainedOut = d_40_ai_
                                    currentConstrainedOut = d_41_ac_
                    pass
            pass
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

