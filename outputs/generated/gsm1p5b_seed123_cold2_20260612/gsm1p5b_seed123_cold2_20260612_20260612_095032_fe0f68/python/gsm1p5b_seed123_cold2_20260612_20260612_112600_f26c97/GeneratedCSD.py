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
        d_0_helpers_: VerifiedDecoderAgent.CSDHelpers
        nw0_ = VerifiedDecoderAgent.CSDHelpers()
        nw0_.ctor__()
        d_0_helpers_ = nw0_
        generated = generatedPrefix
        insideConstrainedOut = insideConstrained
        currentConstrainedOut = currentConstrained
        cost = 0
        d_1_guidance_: _dafny.Seq
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. For each calculation and the final answer, write it inside << >> like <<3*4=12>>. Keep expressions simple and complete."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_closeReserve_: int
        d_3_closeReserve_ = 20
        with _dafny.label("0"):
            while (d_2_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    d_4_remaining_: int
                    d_4_remaining_ = (maxSteps) - (d_2_steps_)
                    if not(insideConstrainedOut):
                        d_5_chunkMax_: int
                        d_5_chunkMax_ = 80
                        if (d_4_remaining_) < (d_5_chunkMax_):
                            d_5_chunkMax_ = d_4_remaining_
                        if (d_5_chunkMax_) == (0):
                            raise _dafny.Break("0")
                        d_6_generatedOut_: _dafny.Seq
                        d_7_stoppedOnOpenSpan_: bool
                        d_8_stoppedOnEos_: bool
                        d_9_stepsUsed_: int
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: bool
                        out3_: int
                        out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_5_chunkMax_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_6_generatedOut_ = out0_
                        d_7_stoppedOnOpenSpan_ = out1_
                        d_8_stoppedOnEos_ = out2_
                        d_9_stepsUsed_ = out3_
                        d_2_steps_ = (d_2_steps_) + (d_9_stepsUsed_)
                        generated = d_6_generatedOut_
                        if d_7_stoppedOnOpenSpan_:
                            d_10_cg_: _dafny.Seq
                            d_11_ci_: bool
                            d_12_cc_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                            d_10_cg_ = out4_
                            d_11_ci_ = out5_
                            d_12_cc_ = out6_
                            generated = d_10_cg_
                            insideConstrainedOut = d_11_ci_
                            currentConstrainedOut = d_12_cc_
                        elif d_8_stoppedOnEos_:
                            raise _dafny.Break("0")
                    elif True:
                        d_13_remaining2_: int
                        d_13_remaining2_ = (maxSteps) - (d_2_steps_)
                        if (d_13_remaining2_) <= (d_3_closeReserve_):
                            d_14_rg_: _dafny.Seq
                            d_15_rc_: _dafny.Seq
                            out7_: _dafny.Seq
                            out8_: _dafny.Seq
                            out7_, out8_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                            d_14_rg_ = out7_
                            d_15_rc_ = out8_
                            generated = d_14_rg_
                            currentConstrainedOut = d_15_rc_
                            if (parser).IsCompletePrefix(currentConstrainedOut):
                                d_16_cg_: _dafny.Seq
                                d_17_ci_: bool
                                d_18_cc_: _dafny.Seq
                                out9_: _dafny.Seq
                                out10_: bool
                                out11_: _dafny.Seq
                                out9_, out10_, out11_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_16_cg_ = out9_
                                d_17_ci_ = out10_
                                d_18_cc_ = out11_
                                d_2_steps_ = (d_2_steps_) + (1)
                                generated = d_16_cg_
                                insideConstrainedOut = d_17_ci_
                                currentConstrainedOut = d_18_cc_
                            elif True:
                                d_2_steps_ = (d_2_steps_) + (1)
                                insideConstrainedOut = False
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                raise _dafny.Break("0")
                        elif (parser).IsCompletePrefix(currentConstrainedOut):
                            d_19_cg_: _dafny.Seq
                            d_20_ci_: bool
                            d_21_cc_: _dafny.Seq
                            out12_: _dafny.Seq
                            out13_: bool
                            out14_: _dafny.Seq
                            out12_, out13_, out14_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_19_cg_ = out12_
                            d_20_ci_ = out13_
                            d_21_cc_ = out14_
                            d_2_steps_ = (d_2_steps_) + (1)
                            generated = d_19_cg_
                            insideConstrainedOut = d_20_ci_
                            currentConstrainedOut = d_21_cc_
                        elif True:
                            d_22_constrainedPrompt_: _dafny.Seq
                            d_22_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_23_next_: _dafny.Seq
                            out15_: _dafny.Seq
                            out15_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_22_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_23_next_ = out15_
                            d_2_steps_ = (d_2_steps_) + (1)
                            if (d_23_next_) == (eosToken):
                                d_24_rg_: _dafny.Seq
                                d_25_rc_: _dafny.Seq
                                out16_: _dafny.Seq
                                out17_: _dafny.Seq
                                out16_, out17_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                                d_24_rg_ = out16_
                                d_25_rc_ = out17_
                                generated = d_24_rg_
                                currentConstrainedOut = d_25_rc_
                                if (parser).IsCompletePrefix(currentConstrainedOut):
                                    if (d_2_steps_) < (maxSteps):
                                        d_26_cg_: _dafny.Seq
                                        d_27_ci_: bool
                                        d_28_cc_: _dafny.Seq
                                        out18_: _dafny.Seq
                                        out19_: bool
                                        out20_: _dafny.Seq
                                        out18_, out19_, out20_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                        d_26_cg_ = out18_
                                        d_27_ci_ = out19_
                                        d_28_cc_ = out20_
                                        d_2_steps_ = (d_2_steps_) + (1)
                                        generated = d_26_cg_
                                        insideConstrainedOut = d_27_ci_
                                        currentConstrainedOut = d_28_cc_
                                    elif True:
                                        raise _dafny.Break("0")
                                elif True:
                                    insideConstrainedOut = False
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                    raise _dafny.Break("0")
                            elif True:
                                d_29_ag_: _dafny.Seq
                                d_30_ai_: bool
                                d_31_ac_: _dafny.Seq
                                out21_: _dafny.Seq
                                out22_: bool
                                out23_: _dafny.Seq
                                out21_, out22_, out23_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_23_next_)
                                d_29_ag_ = out21_
                                d_30_ai_ = out22_
                                d_31_ac_ = out23_
                                generated = d_29_ag_
                                insideConstrainedOut = d_30_ai_
                                currentConstrainedOut = d_31_ac_
                                if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_2_steps_) < (maxSteps)):
                                    d_32_cg_: _dafny.Seq
                                    d_33_ci_: bool
                                    d_34_cc_: _dafny.Seq
                                    out24_: _dafny.Seq
                                    out25_: bool
                                    out26_: _dafny.Seq
                                    out24_, out25_, out26_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_32_cg_ = out24_
                                    d_33_ci_ = out25_
                                    d_34_cc_ = out26_
                                    d_2_steps_ = (d_2_steps_) + (1)
                                    generated = d_32_cg_
                                    insideConstrainedOut = d_33_ci_
                                    currentConstrainedOut = d_34_cc_
                    pass
            pass
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

