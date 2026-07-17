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
        d_1_SPAN__RESERVE_: int
        d_1_SPAN__RESERVE_ = 300
        d_2_steps_: int
        d_2_steps_ = 0
        (d_0_helpers_).SafePenalizeTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]), _dafny.BigRational('25e0'))
        if ((maxSteps) > ((d_1_SPAN__RESERVE_) + (2))) and (not(insideConstrainedOut)):
            d_3_reasoningBudget_: int
            d_3_reasoningBudget_ = ((maxSteps) - (d_1_SPAN__RESERVE_)) - (2)
            d_4_generatedOut_: _dafny.Seq
            d_5_stoppedOnOpenSpan_: bool
            d_6_stoppedOnEos_: bool
            d_7_stepsUsed_: int
            out0_: _dafny.Seq
            out1_: bool
            out2_: bool
            out3_: int
            out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_3_reasoningBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
            d_4_generatedOut_ = out0_
            d_5_stoppedOnOpenSpan_ = out1_
            d_6_stoppedOnEos_ = out2_
            d_7_stepsUsed_ = out3_
            d_2_steps_ = (d_2_steps_) + (d_7_stepsUsed_)
            generated = d_4_generatedOut_
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
        if ((d_2_steps_) < (maxSteps)) and (not(insideConstrainedOut)):
            d_11_g2_: _dafny.Seq
            d_12_i2_: bool
            d_13_c2_: _dafny.Seq
            out7_: _dafny.Seq
            out8_: bool
            out9_: _dafny.Seq
            out7_, out8_, out9_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_11_g2_ = out7_
            d_12_i2_ = out8_
            d_13_c2_ = out9_
            generated = d_11_g2_
            insideConstrainedOut = d_12_i2_
            currentConstrainedOut = d_13_c2_
            d_2_steps_ = (d_2_steps_) + (1)
        with _dafny.label("0"):
            while ((d_2_steps_) < (maxSteps)) and (insideConstrainedOut):
                with _dafny.c_label("0"):
                    if (parser).IsCompletePrefix(currentConstrainedOut):
                        d_14_cg_: _dafny.Seq
                        d_15_ci_: bool
                        d_16_cc_: _dafny.Seq
                        out10_: _dafny.Seq
                        out11_: bool
                        out12_: _dafny.Seq
                        out10_, out11_, out12_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_14_cg_ = out10_
                        d_15_ci_ = out11_
                        d_16_cc_ = out12_
                        generated = d_14_cg_
                        insideConstrainedOut = d_15_ci_
                        currentConstrainedOut = d_16_cc_
                        d_2_steps_ = (d_2_steps_) + (1)
                        raise _dafny.Break("0")
                    elif True:
                        d_17_constrainedPrompt_: _dafny.Seq
                        d_17_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_18_next_: _dafny.Seq
                        out13_: _dafny.Seq
                        out13_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_17_constrainedPrompt_, currentConstrainedOut, eosToken)
                        d_18_next_ = out13_
                        d_2_steps_ = (d_2_steps_) + (1)
                        if (d_18_next_) == (eosToken):
                            d_19_rg_: _dafny.Seq
                            d_20_rc_: _dafny.Seq
                            out14_: _dafny.Seq
                            out15_: _dafny.Seq
                            out14_, out15_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                            d_19_rg_ = out14_
                            d_20_rc_ = out15_
                            generated = d_19_rg_
                            currentConstrainedOut = d_20_rc_
                            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_2_steps_) < (maxSteps)):
                                d_21_cg2_: _dafny.Seq
                                d_22_ci2_: bool
                                d_23_cc2_: _dafny.Seq
                                out16_: _dafny.Seq
                                out17_: bool
                                out18_: _dafny.Seq
                                out16_, out17_, out18_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_21_cg2_ = out16_
                                d_22_ci2_ = out17_
                                d_23_cc2_ = out18_
                                generated = d_21_cg2_
                                insideConstrainedOut = d_22_ci2_
                                currentConstrainedOut = d_23_cc2_
                                d_2_steps_ = (d_2_steps_) + (1)
                            elif True:
                                insideConstrainedOut = False
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                            raise _dafny.Break("0")
                        elif True:
                            d_24_ag_: _dafny.Seq
                            d_25_ai_: bool
                            d_26_ac_: _dafny.Seq
                            out19_: _dafny.Seq
                            out20_: bool
                            out21_: _dafny.Seq
                            out19_, out20_, out21_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_18_next_)
                            d_24_ag_ = out19_
                            d_25_ai_ = out20_
                            d_26_ac_ = out21_
                            generated = d_24_ag_
                            insideConstrainedOut = d_25_ai_
                            currentConstrainedOut = d_26_ac_
                            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_2_steps_) < (maxSteps)):
                                d_27_cg3_: _dafny.Seq
                                d_28_ci3_: bool
                                d_29_cc3_: _dafny.Seq
                                out22_: _dafny.Seq
                                out23_: bool
                                out24_: _dafny.Seq
                                out22_, out23_, out24_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_27_cg3_ = out22_
                                d_28_ci3_ = out23_
                                d_29_cc3_ = out24_
                                generated = d_27_cg3_
                                insideConstrainedOut = d_28_ci3_
                                currentConstrainedOut = d_29_cc3_
                                d_2_steps_ = (d_2_steps_) + (1)
                                raise _dafny.Break("0")
                    pass
            pass
        if insideConstrainedOut:
            d_30_rg_: _dafny.Seq
            d_31_rc_: _dafny.Seq
            out25_: _dafny.Seq
            out26_: _dafny.Seq
            out25_, out26_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
            d_30_rg_ = out25_
            d_31_rc_ = out26_
            generated = d_30_rg_
            currentConstrainedOut = d_31_rc_
            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_2_steps_) < (maxSteps)):
                d_32_cg_: _dafny.Seq
                d_33_ci_: bool
                d_34_cc_: _dafny.Seq
                out27_: _dafny.Seq
                out28_: bool
                out29_: _dafny.Seq
                out27_, out28_, out29_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_32_cg_ = out27_
                d_33_ci_ = out28_
                d_34_cc_ = out29_
                generated = d_32_cg_
                insideConstrainedOut = d_33_ci_
                currentConstrainedOut = d_34_cc_
                d_2_steps_ = (d_2_steps_) + (1)
            elif True:
                insideConstrainedOut = False
                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

