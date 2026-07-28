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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. For EACH intermediate calculation and the final numeric answer, write exactly one expression inside << >>. Inside << >> use ONLY numbers and operators + - * / // % ( ) int. NO variables like {n}, NO ** operator, NO curly braces. Final answer must be a single number expression like <<42>> or <<3*4+2>>."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        with _dafny.label("0"):
            while (d_2_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_3_chunkBudget_: int
                        d_3_chunkBudget_ = (maxSteps) - (d_2_steps_)
                        if (d_3_chunkBudget_) > (40):
                            d_3_chunkBudget_ = 40
                        d_4_generatedOut_: _dafny.Seq
                        d_5_stoppedOnOpenSpan_: bool
                        d_6_stoppedOnEos_: bool
                        d_7_stepsUsed_: int
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: bool
                        out3_: int
                        out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_3_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_4_generatedOut_ = out0_
                        d_5_stoppedOnOpenSpan_ = out1_
                        d_6_stoppedOnEos_ = out2_
                        d_7_stepsUsed_ = out3_
                        d_2_steps_ = (d_2_steps_) + (d_7_stepsUsed_)
                        generated = d_4_generatedOut_
                        if d_6_stoppedOnEos_:
                            raise _dafny.Break("0")
                        if d_5_stoppedOnOpenSpan_:
                            d_8_g2_: _dafny.Seq
                            d_9_i2_: bool
                            d_10_c2_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                            d_8_g2_ = out4_
                            d_9_i2_ = out5_
                            d_10_c2_ = out6_
                            generated = d_8_g2_
                            insideConstrainedOut = d_9_i2_
                            currentConstrainedOut = d_10_c2_
                    elif True:
                        d_11_cg_: _dafny.Seq
                        d_12_ci_: bool
                        d_13_cc_: _dafny.Seq
                        d_14_closed_: bool
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: _dafny.Seq
                        out10_: bool
                        out7_, out8_, out9_, out10_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                        d_11_cg_ = out7_
                        d_12_ci_ = out8_
                        d_13_cc_ = out9_
                        d_14_closed_ = out10_
                        d_2_steps_ = (d_2_steps_) + (1)
                        if d_14_closed_:
                            generated = d_11_cg_
                            insideConstrainedOut = d_12_ci_
                            currentConstrainedOut = d_13_cc_
                        elif True:
                            d_15_remainingBudget_: int
                            d_15_remainingBudget_ = (maxSteps) - (d_2_steps_)
                            if ((d_15_remainingBudget_) <= (5)) and ((d_15_remainingBudget_) > (0)):
                                d_16_sg_: _dafny.Seq
                                d_17_si_: bool
                                d_18_sc_: _dafny.Seq
                                out11_: _dafny.Seq
                                out12_: bool
                                out13_: _dafny.Seq
                                out11_, out12_, out13_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_15_remainingBudget_)
                                d_16_sg_ = out11_
                                d_17_si_ = out12_
                                d_18_sc_ = out13_
                                generated = d_16_sg_
                                insideConstrainedOut = d_17_si_
                                currentConstrainedOut = d_18_sc_
                                d_2_steps_ = (d_2_steps_) + (d_15_remainingBudget_)
                            elif True:
                                d_19_constrainedPrompt_: _dafny.Seq
                                d_19_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_20_next_: _dafny.Seq
                                out14_: _dafny.Seq
                                out14_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_19_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_20_next_ = out14_
                                if (d_20_next_) == (eosToken):
                                    if ((maxSteps) - (d_2_steps_)) > (0):
                                        d_21_sg2_: _dafny.Seq
                                        d_22_si2_: bool
                                        d_23_sc2_: _dafny.Seq
                                        out15_: _dafny.Seq
                                        out16_: bool
                                        out17_: _dafny.Seq
                                        out15_, out16_, out17_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, 1)
                                        d_21_sg2_ = out15_
                                        d_22_si2_ = out16_
                                        d_23_sc2_ = out17_
                                        generated = d_21_sg2_
                                        insideConstrainedOut = d_22_si2_
                                        currentConstrainedOut = d_23_sc2_
                                        d_2_steps_ = (d_2_steps_) + (1)
                                    raise _dafny.Break("0")
                                elif True:
                                    d_24_ag_: _dafny.Seq
                                    d_25_ai_: bool
                                    d_26_ac_: _dafny.Seq
                                    out18_: _dafny.Seq
                                    out19_: bool
                                    out20_: _dafny.Seq
                                    out18_, out19_, out20_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_20_next_)
                                    d_24_ag_ = out18_
                                    d_25_ai_ = out19_
                                    d_26_ac_ = out20_
                                    generated = d_24_ag_
                                    insideConstrainedOut = d_25_ai_
                                    currentConstrainedOut = d_26_ac_
                    pass
            pass
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

