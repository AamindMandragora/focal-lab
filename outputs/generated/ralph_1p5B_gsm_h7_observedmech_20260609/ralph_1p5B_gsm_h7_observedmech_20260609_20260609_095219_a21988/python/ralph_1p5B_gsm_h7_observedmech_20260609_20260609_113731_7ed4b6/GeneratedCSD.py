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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. Write the final numeric answer inside << >> delimiters. The content between << and >> must be a valid arithmetic expression using only numbers, variables, +, -, *, /, (, ). Example: <<42>> or <<n1 * r + b>>.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_hasSeenOpenSpan_: bool
        d_2_hasSeenOpenSpan_ = insideConstrained
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_3_remaining_: int
                        d_3_remaining_ = (maxSteps) - (d_1_steps_)
                        if ((d_3_remaining_) <= (60)) and (not(d_2_hasSeenOpenSpan_)):
                            d_4_g2_: _dafny.Seq
                            d_5_i2_: bool
                            d_6_c2_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_4_g2_ = out0_
                            d_5_i2_ = out1_
                            d_6_c2_ = out2_
                            generated = d_4_g2_
                            insideConstrainedOut = d_5_i2_
                            currentConstrainedOut = d_6_c2_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_2_hasSeenOpenSpan_ = True
                        elif True:
                            d_7_chunkBudget_: int
                            if (d_3_remaining_) < (25):
                                d_7_chunkBudget_ = d_3_remaining_
                            elif True:
                                d_7_chunkBudget_ = 25
                            d_8_chunkGenerated_: _dafny.Seq
                            d_9_stoppedOnOpenSpan_: bool
                            d_10_stoppedOnEos_: bool
                            d_11_stepsUsed_: int
                            out3_: _dafny.Seq
                            out4_: bool
                            out5_: bool
                            out6_: int
                            out3_, out4_, out5_, out6_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_7_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_8_chunkGenerated_ = out3_
                            d_9_stoppedOnOpenSpan_ = out4_
                            d_10_stoppedOnEos_ = out5_
                            d_11_stepsUsed_ = out6_
                            generated = d_8_chunkGenerated_
                            d_1_steps_ = (d_1_steps_) + (d_11_stepsUsed_)
                            if d_10_stoppedOnEos_:
                                if (not(d_2_hasSeenOpenSpan_)) and (((d_1_steps_) + (3)) <= (maxSteps)):
                                    d_12_g2_: _dafny.Seq
                                    d_13_i2_: bool
                                    d_14_c2_: _dafny.Seq
                                    out7_: _dafny.Seq
                                    out8_: bool
                                    out9_: _dafny.Seq
                                    out7_, out8_, out9_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                    d_12_g2_ = out7_
                                    d_13_i2_ = out8_
                                    d_14_c2_ = out9_
                                    generated = d_12_g2_
                                    insideConstrainedOut = d_13_i2_
                                    currentConstrainedOut = d_14_c2_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    d_2_hasSeenOpenSpan_ = True
                                elif True:
                                    raise _dafny.Break("0")
                            elif d_9_stoppedOnOpenSpan_:
                                d_15_g2_: _dafny.Seq
                                d_16_i2_: bool
                                d_17_c2_: _dafny.Seq
                                out10_: _dafny.Seq
                                out11_: bool
                                out12_: _dafny.Seq
                                out10_, out11_, out12_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_15_g2_ = out10_
                                d_16_i2_ = out11_
                                d_17_c2_ = out12_
                                generated = d_15_g2_
                                insideConstrainedOut = d_16_i2_
                                currentConstrainedOut = d_17_c2_
                                d_2_hasSeenOpenSpan_ = True
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_18_g2_: _dafny.Seq
                        d_19_i2_: bool
                        d_20_c2_: _dafny.Seq
                        out13_: _dafny.Seq
                        out14_: bool
                        out15_: _dafny.Seq
                        out13_, out14_, out15_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_18_g2_ = out13_
                        d_19_i2_ = out14_
                        d_20_c2_ = out15_
                        generated = d_18_g2_
                        insideConstrainedOut = d_19_i2_
                        currentConstrainedOut = d_20_c2_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_21_constrainedPrompt_: _dafny.Seq
                        d_21_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_22_next_: _dafny.Seq
                        d_23_wasConstrained_: bool
                        out16_: _dafny.Seq
                        out17_: bool
                        out16_, out17_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, d_21_constrainedPrompt_, currentConstrainedOut, eosToken)
                        d_22_next_ = out16_
                        d_23_wasConstrained_ = out17_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_22_next_) == (eosToken):
                            d_24_gRolled_: _dafny.Seq
                            d_25_cRolled_: _dafny.Seq
                            out18_: _dafny.Seq
                            out19_: _dafny.Seq
                            out18_, out19_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                            d_24_gRolled_ = out18_
                            d_25_cRolled_ = out19_
                            generated = d_24_gRolled_
                            currentConstrainedOut = d_25_cRolled_
                            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                d_26_g2_: _dafny.Seq
                                d_27_i2_: bool
                                d_28_c2_: _dafny.Seq
                                out20_: _dafny.Seq
                                out21_: bool
                                out22_: _dafny.Seq
                                out20_, out21_, out22_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_26_g2_ = out20_
                                d_27_i2_ = out21_
                                d_28_c2_ = out22_
                                generated = d_26_g2_
                                insideConstrainedOut = d_27_i2_
                                currentConstrainedOut = d_28_c2_
                                d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_29_g2_: _dafny.Seq
                            d_30_i2_: bool
                            d_31_c2_: _dafny.Seq
                            out23_: _dafny.Seq
                            out24_: bool
                            out25_: _dafny.Seq
                            out23_, out24_, out25_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_22_next_)
                            d_29_g2_ = out23_
                            d_30_i2_ = out24_
                            d_31_c2_ = out25_
                            generated = d_29_g2_
                            insideConstrainedOut = d_30_i2_
                            currentConstrainedOut = d_31_c2_
                            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                d_32_g3_: _dafny.Seq
                                d_33_i3_: bool
                                d_34_c3_: _dafny.Seq
                                out26_: _dafny.Seq
                                out27_: bool
                                out28_: _dafny.Seq
                                out26_, out27_, out28_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_32_g3_ = out26_
                                d_33_i3_ = out27_
                                d_34_c3_ = out28_
                                generated = d_32_g3_
                                insideConstrainedOut = d_33_i3_
                                currentConstrainedOut = d_34_c3_
                                d_1_steps_ = (d_1_steps_) + (1)
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_35_gRolled_: _dafny.Seq
            d_36_cRolled_: _dafny.Seq
            out29_: _dafny.Seq
            out30_: _dafny.Seq
            out29_, out30_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
            d_35_gRolled_ = out29_
            d_36_cRolled_ = out30_
            generated = d_35_gRolled_
            currentConstrainedOut = d_36_cRolled_
            if (not((parser).IsCompletePrefix(currentConstrainedOut))) and (((d_1_steps_) + (1)) < (maxSteps)):
                d_37_constrainedPrompt2_: _dafny.Seq
                d_37_constrainedPrompt2_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                d_38_next2_: _dafny.Seq
                d_39_wasConstrained2_: bool
                out31_: _dafny.Seq
                out32_: bool
                out31_, out32_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, d_37_constrainedPrompt2_, currentConstrainedOut, eosToken)
                d_38_next2_ = out31_
                d_39_wasConstrained2_ = out32_
                d_1_steps_ = (d_1_steps_) + (1)
                if (d_38_next2_) != (eosToken):
                    d_40_g2_: _dafny.Seq
                    d_41_i2_: bool
                    d_42_c2_: _dafny.Seq
                    out33_: _dafny.Seq
                    out34_: bool
                    out35_: _dafny.Seq
                    out33_, out34_, out35_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_38_next2_)
                    d_40_g2_ = out33_
                    d_41_i2_ = out34_
                    d_42_c2_ = out35_
                    generated = d_40_g2_
                    insideConstrainedOut = d_41_i2_
                    currentConstrainedOut = d_42_c2_
            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                d_43_g2_: _dafny.Seq
                d_44_i2_: bool
                d_45_c2_: _dafny.Seq
                out36_: _dafny.Seq
                out37_: bool
                out38_: _dafny.Seq
                out36_, out37_, out38_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_43_g2_ = out36_
                d_44_i2_ = out37_
                d_45_c2_ = out38_
                generated = d_43_g2_
                insideConstrainedOut = d_44_i2_
                currentConstrainedOut = d_45_c2_
                d_1_steps_ = (d_1_steps_) + (1)
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

