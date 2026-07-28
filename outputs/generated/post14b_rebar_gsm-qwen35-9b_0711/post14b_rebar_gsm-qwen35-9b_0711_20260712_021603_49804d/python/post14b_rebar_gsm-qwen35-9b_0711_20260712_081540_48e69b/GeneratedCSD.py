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
        d_2_phase1Budget_: int
        d_2_phase1Budget_ = 0
        if (maxSteps) > (20):
            d_2_phase1Budget_ = (maxSteps) - (20)
        elif True:
            d_2_phase1Budget_ = maxSteps
        if ((d_2_phase1Budget_) > (0)) and (not(insideConstrainedOut)):
            d_3_genOut_: _dafny.Seq
            d_4_stoppedOnOpenSpan_: bool
            d_5_stoppedOnEos_: bool
            d_6_stepsUsed_: int
            out0_: _dafny.Seq
            out1_: bool
            out2_: bool
            out3_: int
            out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_2_phase1Budget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
            d_3_genOut_ = out0_
            d_4_stoppedOnOpenSpan_ = out1_
            d_5_stoppedOnEos_ = out2_
            d_6_stepsUsed_ = out3_
            generated = d_3_genOut_
            d_1_steps_ = (d_1_steps_) + (d_6_stepsUsed_)
            if d_4_stoppedOnOpenSpan_:
                d_7_g2_: _dafny.Seq
                d_8_i2_: bool
                d_9_c2_: _dafny.Seq
                out4_: _dafny.Seq
                out5_: bool
                out6_: _dafny.Seq
                out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                d_7_g2_ = out4_
                d_8_i2_ = out5_
                d_9_c2_ = out6_
                generated = d_7_g2_
                insideConstrainedOut = d_8_i2_
                currentConstrainedOut = d_9_c2_
        with _dafny.label("0"):
            while ((d_1_steps_) < (maxSteps)) and (insideConstrainedOut):
                with _dafny.c_label("0"):
                    d_10_cg_: _dafny.Seq
                    d_11_ci_: bool
                    d_12_cc_: _dafny.Seq
                    d_13_closed_: bool
                    out7_: _dafny.Seq
                    out8_: bool
                    out9_: _dafny.Seq
                    out10_: bool
                    out7_, out8_, out9_, out10_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                    d_10_cg_ = out7_
                    d_11_ci_ = out8_
                    d_12_cc_ = out9_
                    d_13_closed_ = out10_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if d_13_closed_:
                        generated = d_10_cg_
                        insideConstrainedOut = d_11_ci_
                        currentConstrainedOut = d_12_cc_
                    elif True:
                        d_14_constrainedPrompt_: _dafny.Seq
                        d_14_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_15_next_: _dafny.Seq
                        out11_: _dafny.Seq
                        out11_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_14_constrainedPrompt_, currentConstrainedOut, eosToken)
                        d_15_next_ = out11_
                        if (d_15_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_16_ag_: _dafny.Seq
                            d_17_ai_: bool
                            d_18_ac_: _dafny.Seq
                            out12_: _dafny.Seq
                            out13_: bool
                            out14_: _dafny.Seq
                            out12_, out13_, out14_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_15_next_)
                            d_16_ag_ = out12_
                            d_17_ai_ = out13_
                            d_18_ac_ = out14_
                            generated = d_16_ag_
                            insideConstrainedOut = d_17_ai_
                            currentConstrainedOut = d_18_ac_
                    pass
            pass
        if (not(insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
            d_19_remainBudget_: int
            d_19_remainBudget_ = (maxSteps) - (d_1_steps_)
            d_20_genOut2_: _dafny.Seq
            d_21_stoppedOnOpenSpan2_: bool
            d_22_stoppedOnEos2_: bool
            d_23_stepsUsed2_: int
            out15_: _dafny.Seq
            out16_: bool
            out17_: bool
            out18_: int
            out15_, out16_, out17_, out18_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_19_remainBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
            d_20_genOut2_ = out15_
            d_21_stoppedOnOpenSpan2_ = out16_
            d_22_stoppedOnEos2_ = out17_
            d_23_stepsUsed2_ = out18_
            generated = d_20_genOut2_
            d_1_steps_ = (d_1_steps_) + (d_23_stepsUsed2_)
            if d_21_stoppedOnOpenSpan2_:
                d_24_g3_: _dafny.Seq
                d_25_i3_: bool
                d_26_c3_: _dafny.Seq
                out19_: _dafny.Seq
                out20_: bool
                out21_: _dafny.Seq
                out19_, out20_, out21_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                d_24_g3_ = out19_
                d_25_i3_ = out20_
                d_26_c3_ = out21_
                generated = d_24_g3_
                insideConstrainedOut = d_25_i3_
                currentConstrainedOut = d_26_c3_
                with _dafny.label("4_0_0"):
                    while ((d_1_steps_) < (maxSteps)) and (insideConstrainedOut):
                        with _dafny.c_label("4_0_0"):
                            d_27_cg2_: _dafny.Seq
                            d_28_ci2_: bool
                            d_29_cc2_: _dafny.Seq
                            d_30_closed2_: bool
                            out22_: _dafny.Seq
                            out23_: bool
                            out24_: _dafny.Seq
                            out25_: bool
                            out22_, out23_, out24_, out25_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                            d_27_cg2_ = out22_
                            d_28_ci2_ = out23_
                            d_29_cc2_ = out24_
                            d_30_closed2_ = out25_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if d_30_closed2_:
                                generated = d_27_cg2_
                                insideConstrainedOut = d_28_ci2_
                                currentConstrainedOut = d_29_cc2_
                            elif True:
                                d_31_constrainedPrompt2_: _dafny.Seq
                                d_31_constrainedPrompt2_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_32_next2_: _dafny.Seq
                                out26_: _dafny.Seq
                                out26_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_31_constrainedPrompt2_, currentConstrainedOut, eosToken)
                                d_32_next2_ = out26_
                                if (d_32_next2_) == (eosToken):
                                    raise _dafny.Break("4_0_0")
                                elif True:
                                    d_33_ag2_: _dafny.Seq
                                    d_34_ai2_: bool
                                    d_35_ac2_: _dafny.Seq
                                    out27_: _dafny.Seq
                                    out28_: bool
                                    out29_: _dafny.Seq
                                    out27_, out28_, out29_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_32_next2_)
                                    d_33_ag2_ = out27_
                                    d_34_ai2_ = out28_
                                    d_35_ac2_ = out29_
                                    generated = d_33_ag2_
                                    insideConstrainedOut = d_34_ai2_
                                    currentConstrainedOut = d_35_ac2_
                            pass
                    pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

