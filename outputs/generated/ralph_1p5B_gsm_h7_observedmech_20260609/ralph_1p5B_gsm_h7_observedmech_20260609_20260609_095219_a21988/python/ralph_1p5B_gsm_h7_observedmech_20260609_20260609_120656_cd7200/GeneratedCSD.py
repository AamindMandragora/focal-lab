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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math problem step by step. Express each calculation as <<expression>>. Write the final numeric answer as <<number>>.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_freeChunkSize_: int
        d_2_freeChunkSize_ = 25
        d_3_spanTokensUsed_: int
        d_3_spanTokensUsed_ = 0
        d_4_spanMaxTokens_: int
        d_4_spanMaxTokens_ = 10
        d_5_hasSeenOpenSpan_: bool
        d_5_hasSeenOpenSpan_ = insideConstrained
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_6_remaining_: int
                        d_6_remaining_ = (maxSteps) - (d_1_steps_)
                        if (((d_6_remaining_) <= (60)) and (not(d_5_hasSeenOpenSpan_))) and ((d_6_remaining_) > (3)):
                            d_7_g2_: _dafny.Seq
                            d_8_i2_: bool
                            d_9_c2_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_7_g2_ = out0_
                            d_8_i2_ = out1_
                            d_9_c2_ = out2_
                            generated = d_7_g2_
                            insideConstrainedOut = d_8_i2_
                            currentConstrainedOut = d_9_c2_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_3_spanTokensUsed_ = 0
                            d_5_hasSeenOpenSpan_ = True
                        elif True:
                            d_10_chunkBudget_: int
                            if (d_6_remaining_) < (d_2_freeChunkSize_):
                                d_10_chunkBudget_ = d_6_remaining_
                            elif True:
                                d_10_chunkBudget_ = d_2_freeChunkSize_
                            if (d_10_chunkBudget_) == (0):
                                raise _dafny.Break("0")
                            d_11_chunkGenerated_: _dafny.Seq
                            d_12_stoppedOnOpenSpan_: bool
                            d_13_stoppedOnEos_: bool
                            d_14_stepsUsed_: int
                            out3_: _dafny.Seq
                            out4_: bool
                            out5_: bool
                            out6_: int
                            out3_, out4_, out5_, out6_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_10_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_11_chunkGenerated_ = out3_
                            d_12_stoppedOnOpenSpan_ = out4_
                            d_13_stoppedOnEos_ = out5_
                            d_14_stepsUsed_ = out6_
                            generated = d_11_chunkGenerated_
                            d_1_steps_ = (d_1_steps_) + (d_14_stepsUsed_)
                            if d_13_stoppedOnEos_:
                                if (not(d_5_hasSeenOpenSpan_)) and (((d_1_steps_) + (3)) <= (maxSteps)):
                                    d_15_g2_: _dafny.Seq
                                    d_16_i2_: bool
                                    d_17_c2_: _dafny.Seq
                                    out7_: _dafny.Seq
                                    out8_: bool
                                    out9_: _dafny.Seq
                                    out7_, out8_, out9_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                    d_15_g2_ = out7_
                                    d_16_i2_ = out8_
                                    d_17_c2_ = out9_
                                    generated = d_15_g2_
                                    insideConstrainedOut = d_16_i2_
                                    currentConstrainedOut = d_17_c2_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    d_3_spanTokensUsed_ = 0
                                    d_5_hasSeenOpenSpan_ = True
                                elif True:
                                    raise _dafny.Break("0")
                            elif d_12_stoppedOnOpenSpan_:
                                d_18_g2_: _dafny.Seq
                                d_19_i2_: bool
                                d_20_c2_: _dafny.Seq
                                out10_: _dafny.Seq
                                out11_: bool
                                out12_: _dafny.Seq
                                out10_, out11_, out12_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_18_g2_ = out10_
                                d_19_i2_ = out11_
                                d_20_c2_ = out12_
                                generated = d_18_g2_
                                insideConstrainedOut = d_19_i2_
                                currentConstrainedOut = d_20_c2_
                                d_3_spanTokensUsed_ = 0
                                d_5_hasSeenOpenSpan_ = True
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_21_g2_: _dafny.Seq
                        d_22_i2_: bool
                        d_23_c2_: _dafny.Seq
                        out13_: _dafny.Seq
                        out14_: bool
                        out15_: _dafny.Seq
                        out13_, out14_, out15_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_21_g2_ = out13_
                        d_22_i2_ = out14_
                        d_23_c2_ = out15_
                        generated = d_21_g2_
                        insideConstrainedOut = d_22_i2_
                        currentConstrainedOut = d_23_c2_
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_3_spanTokensUsed_ = 0
                    elif (d_3_spanTokensUsed_) >= (d_4_spanMaxTokens_):
                        d_24_gRolled_: _dafny.Seq
                        d_25_cRolled_: _dafny.Seq
                        out16_: _dafny.Seq
                        out17_: _dafny.Seq
                        out16_, out17_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                        d_24_gRolled_ = out16_
                        d_25_cRolled_ = out17_
                        generated = d_24_gRolled_
                        currentConstrainedOut = d_25_cRolled_
                        d_3_spanTokensUsed_ = 0
                        if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                            d_26_g2_: _dafny.Seq
                            d_27_i2_: bool
                            d_28_c2_: _dafny.Seq
                            out18_: _dafny.Seq
                            out19_: bool
                            out20_: _dafny.Seq
                            out18_, out19_, out20_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_26_g2_ = out18_
                            d_27_i2_ = out19_
                            d_28_c2_ = out20_
                            generated = d_26_g2_
                            insideConstrainedOut = d_27_i2_
                            currentConstrainedOut = d_28_c2_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif (d_1_steps_) < (maxSteps):
                            d_29_constrainedPrompt_: _dafny.Seq
                            d_29_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_30_next_: _dafny.Seq
                            out21_: _dafny.Seq
                            out21_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_29_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_30_next_ = out21_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_30_next_) != (eosToken):
                                d_31_g2_: _dafny.Seq
                                d_32_i2_: bool
                                d_33_c2_: _dafny.Seq
                                out22_: _dafny.Seq
                                out23_: bool
                                out24_: _dafny.Seq
                                out22_, out23_, out24_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_30_next_)
                                d_31_g2_ = out22_
                                d_32_i2_ = out23_
                                d_33_c2_ = out24_
                                generated = d_31_g2_
                                insideConstrainedOut = d_32_i2_
                                currentConstrainedOut = d_33_c2_
                                d_3_spanTokensUsed_ = 1
                    elif True:
                        d_34_constrainedPrompt_: _dafny.Seq
                        d_34_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_35_next_: _dafny.Seq
                        out25_: _dafny.Seq
                        out25_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_34_constrainedPrompt_, currentConstrainedOut, eosToken)
                        d_35_next_ = out25_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_35_next_) == (eosToken):
                            d_36_gRolled_: _dafny.Seq
                            d_37_cRolled_: _dafny.Seq
                            out26_: _dafny.Seq
                            out27_: _dafny.Seq
                            out26_, out27_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                            d_36_gRolled_ = out26_
                            d_37_cRolled_ = out27_
                            generated = d_36_gRolled_
                            currentConstrainedOut = d_37_cRolled_
                            d_3_spanTokensUsed_ = 0
                            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                d_38_g2_: _dafny.Seq
                                d_39_i2_: bool
                                d_40_c2_: _dafny.Seq
                                out28_: _dafny.Seq
                                out29_: bool
                                out30_: _dafny.Seq
                                out28_, out29_, out30_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_38_g2_ = out28_
                                d_39_i2_ = out29_
                                d_40_c2_ = out30_
                                generated = d_38_g2_
                                insideConstrainedOut = d_39_i2_
                                currentConstrainedOut = d_40_c2_
                                d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_41_g2_: _dafny.Seq
                            d_42_i2_: bool
                            d_43_c2_: _dafny.Seq
                            out31_: _dafny.Seq
                            out32_: bool
                            out33_: _dafny.Seq
                            out31_, out32_, out33_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_35_next_)
                            d_41_g2_ = out31_
                            d_42_i2_ = out32_
                            d_43_c2_ = out33_
                            generated = d_41_g2_
                            insideConstrainedOut = d_42_i2_
                            currentConstrainedOut = d_43_c2_
                            d_3_spanTokensUsed_ = (d_3_spanTokensUsed_) + (1)
                            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                d_44_g3_: _dafny.Seq
                                d_45_i3_: bool
                                d_46_c3_: _dafny.Seq
                                out34_: _dafny.Seq
                                out35_: bool
                                out36_: _dafny.Seq
                                out34_, out35_, out36_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_44_g3_ = out34_
                                d_45_i3_ = out35_
                                d_46_c3_ = out36_
                                generated = d_44_g3_
                                insideConstrainedOut = d_45_i3_
                                currentConstrainedOut = d_46_c3_
                                d_1_steps_ = (d_1_steps_) + (1)
                                d_3_spanTokensUsed_ = 0
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_47_gRolled_: _dafny.Seq
            d_48_cRolled_: _dafny.Seq
            out37_: _dafny.Seq
            out38_: _dafny.Seq
            out37_, out38_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
            d_47_gRolled_ = out37_
            d_48_cRolled_ = out38_
            generated = d_47_gRolled_
            currentConstrainedOut = d_48_cRolled_
            if (not((parser).IsCompletePrefix(currentConstrainedOut))) and (((d_1_steps_) + (1)) < (maxSteps)):
                d_49_constrainedPrompt_: _dafny.Seq
                d_49_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                d_50_next_: _dafny.Seq
                out39_: _dafny.Seq
                out39_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_49_constrainedPrompt_, currentConstrainedOut, eosToken)
                d_50_next_ = out39_
                d_1_steps_ = (d_1_steps_) + (1)
                if (d_50_next_) != (eosToken):
                    d_51_g2_: _dafny.Seq
                    d_52_i2_: bool
                    d_53_c2_: _dafny.Seq
                    out40_: _dafny.Seq
                    out41_: bool
                    out42_: _dafny.Seq
                    out40_, out41_, out42_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_50_next_)
                    d_51_g2_ = out40_
                    d_52_i2_ = out41_
                    d_53_c2_ = out42_
                    generated = d_51_g2_
                    insideConstrainedOut = d_52_i2_
                    currentConstrainedOut = d_53_c2_
            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                d_54_g2_: _dafny.Seq
                d_55_i2_: bool
                d_56_c2_: _dafny.Seq
                out43_: _dafny.Seq
                out44_: bool
                out45_: _dafny.Seq
                out43_, out44_, out45_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_54_g2_ = out43_
                d_55_i2_ = out44_
                d_56_c2_ = out45_
                generated = d_54_g2_
                insideConstrainedOut = d_55_i2_
                currentConstrainedOut = d_56_c2_
                d_1_steps_ = (d_1_steps_) + (1)
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

