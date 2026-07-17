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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. For each calculation and the final answer, write it inside << >> delimiters. Example: The total is <<3 + 5>> = 8. Final answer: <<8>>.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_freeChunkSize_: int
        d_2_freeChunkSize_ = 30
        d_3_spanTokensUsed_: int
        d_3_spanTokensUsed_ = 0
        d_4_spanMaxTokens_: int
        d_4_spanMaxTokens_ = 10
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_5_remaining_: int
                        d_5_remaining_ = (maxSteps) - (d_1_steps_)
                        if ((d_5_remaining_) <= (60)) and ((d_5_remaining_) > (1)):
                            d_6_g2_: _dafny.Seq
                            d_7_i2_: bool
                            d_8_c2_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_6_g2_ = out0_
                            d_7_i2_ = out1_
                            d_8_c2_ = out2_
                            generated = d_6_g2_
                            insideConstrainedOut = d_7_i2_
                            currentConstrainedOut = d_8_c2_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_3_spanTokensUsed_ = 0
                        elif True:
                            d_9_chunkBudget_: int
                            if (d_5_remaining_) < (d_2_freeChunkSize_):
                                d_9_chunkBudget_ = d_5_remaining_
                            elif True:
                                d_9_chunkBudget_ = d_2_freeChunkSize_
                            if (d_9_chunkBudget_) == (0):
                                raise _dafny.Break("0")
                            d_10_chunkGenerated_: _dafny.Seq
                            d_11_stoppedOnOpenSpan_: bool
                            d_12_stoppedOnEos_: bool
                            d_13_stepsUsed_: int
                            out3_: _dafny.Seq
                            out4_: bool
                            out5_: bool
                            out6_: int
                            out3_, out4_, out5_, out6_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_9_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_10_chunkGenerated_ = out3_
                            d_11_stoppedOnOpenSpan_ = out4_
                            d_12_stoppedOnEos_ = out5_
                            d_13_stepsUsed_ = out6_
                            generated = d_10_chunkGenerated_
                            d_1_steps_ = (d_1_steps_) + (d_13_stepsUsed_)
                            if d_12_stoppedOnEos_:
                                raise _dafny.Break("0")
                            elif d_11_stoppedOnOpenSpan_:
                                d_14_g2_: _dafny.Seq
                                d_15_i2_: bool
                                d_16_c2_: _dafny.Seq
                                out7_: _dafny.Seq
                                out8_: bool
                                out9_: _dafny.Seq
                                out7_, out8_, out9_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_14_g2_ = out7_
                                d_15_i2_ = out8_
                                d_16_c2_ = out9_
                                generated = d_14_g2_
                                insideConstrainedOut = d_15_i2_
                                currentConstrainedOut = d_16_c2_
                                d_3_spanTokensUsed_ = 0
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_17_g2_: _dafny.Seq
                        d_18_i2_: bool
                        d_19_c2_: _dafny.Seq
                        out10_: _dafny.Seq
                        out11_: bool
                        out12_: _dafny.Seq
                        out10_, out11_, out12_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_17_g2_ = out10_
                        d_18_i2_ = out11_
                        d_19_c2_ = out12_
                        generated = d_17_g2_
                        insideConstrainedOut = d_18_i2_
                        currentConstrainedOut = d_19_c2_
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_3_spanTokensUsed_ = 0
                    elif True:
                        d_20_isDeadEnd_: bool
                        out13_: bool
                        out13_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 2)
                        d_20_isDeadEnd_ = out13_
                        if (d_20_isDeadEnd_) or ((d_3_spanTokensUsed_) >= (d_4_spanMaxTokens_)):
                            d_21_gRolled_: _dafny.Seq
                            d_22_cRolled_: _dafny.Seq
                            out14_: _dafny.Seq
                            out15_: _dafny.Seq
                            out14_, out15_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                            d_21_gRolled_ = out14_
                            d_22_cRolled_ = out15_
                            generated = d_21_gRolled_
                            currentConstrainedOut = d_22_cRolled_
                            d_3_spanTokensUsed_ = 0
                            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                d_23_g2_: _dafny.Seq
                                d_24_i2_: bool
                                d_25_c2_: _dafny.Seq
                                out16_: _dafny.Seq
                                out17_: bool
                                out18_: _dafny.Seq
                                out16_, out17_, out18_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_23_g2_ = out16_
                                d_24_i2_ = out17_
                                d_25_c2_ = out18_
                                generated = d_23_g2_
                                insideConstrainedOut = d_24_i2_
                                currentConstrainedOut = d_25_c2_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif (d_1_steps_) < (maxSteps):
                                d_26_constrainedPrompt_: _dafny.Seq
                                d_26_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_27_next_: _dafny.Seq
                                out19_: _dafny.Seq
                                out19_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_26_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_27_next_ = out19_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_27_next_) == (eosToken):
                                    d_28_gR2_: _dafny.Seq
                                    d_29_cR2_: _dafny.Seq
                                    out20_: _dafny.Seq
                                    out21_: _dafny.Seq
                                    out20_, out21_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                    d_28_gR2_ = out20_
                                    d_29_cR2_ = out21_
                                    generated = d_28_gR2_
                                    currentConstrainedOut = d_29_cR2_
                                    if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                        d_30_g2_: _dafny.Seq
                                        d_31_i2_: bool
                                        d_32_c2_: _dafny.Seq
                                        out22_: _dafny.Seq
                                        out23_: bool
                                        out24_: _dafny.Seq
                                        out22_, out23_, out24_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                        d_30_g2_ = out22_
                                        d_31_i2_ = out23_
                                        d_32_c2_ = out24_
                                        generated = d_30_g2_
                                        insideConstrainedOut = d_31_i2_
                                        currentConstrainedOut = d_32_c2_
                                        d_1_steps_ = (d_1_steps_) + (1)
                                    elif True:
                                        raise _dafny.Break("0")
                                elif True:
                                    d_33_g2_: _dafny.Seq
                                    d_34_i2_: bool
                                    d_35_c2_: _dafny.Seq
                                    out25_: _dafny.Seq
                                    out26_: bool
                                    out27_: _dafny.Seq
                                    out25_, out26_, out27_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_27_next_)
                                    d_33_g2_ = out25_
                                    d_34_i2_ = out26_
                                    d_35_c2_ = out27_
                                    generated = d_33_g2_
                                    insideConstrainedOut = d_34_i2_
                                    currentConstrainedOut = d_35_c2_
                                    d_3_spanTokensUsed_ = (d_3_spanTokensUsed_) + (1)
                                    if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                        d_36_g3_: _dafny.Seq
                                        d_37_i3_: bool
                                        d_38_c3_: _dafny.Seq
                                        out28_: _dafny.Seq
                                        out29_: bool
                                        out30_: _dafny.Seq
                                        out28_, out29_, out30_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                        d_36_g3_ = out28_
                                        d_37_i3_ = out29_
                                        d_38_c3_ = out30_
                                        generated = d_36_g3_
                                        insideConstrainedOut = d_37_i3_
                                        currentConstrainedOut = d_38_c3_
                                        d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_39_constrainedPrompt_: _dafny.Seq
                            d_39_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_40_next_: _dafny.Seq
                            d_41_wasConstrained_: bool
                            out31_: _dafny.Seq
                            out32_: bool
                            out31_, out32_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, d_39_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_40_next_ = out31_
                            d_41_wasConstrained_ = out32_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_40_next_) == (eosToken):
                                d_42_gRolled_: _dafny.Seq
                                d_43_cRolled_: _dafny.Seq
                                out33_: _dafny.Seq
                                out34_: _dafny.Seq
                                out33_, out34_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                d_42_gRolled_ = out33_
                                d_43_cRolled_ = out34_
                                generated = d_42_gRolled_
                                currentConstrainedOut = d_43_cRolled_
                                d_3_spanTokensUsed_ = 0
                                if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                    d_44_g2_: _dafny.Seq
                                    d_45_i2_: bool
                                    d_46_c2_: _dafny.Seq
                                    out35_: _dafny.Seq
                                    out36_: bool
                                    out37_: _dafny.Seq
                                    out35_, out36_, out37_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_44_g2_ = out35_
                                    d_45_i2_ = out36_
                                    d_46_c2_ = out37_
                                    generated = d_44_g2_
                                    insideConstrainedOut = d_45_i2_
                                    currentConstrainedOut = d_46_c2_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                elif True:
                                    raise _dafny.Break("0")
                            elif True:
                                d_47_g2_: _dafny.Seq
                                d_48_i2_: bool
                                d_49_c2_: _dafny.Seq
                                out38_: _dafny.Seq
                                out39_: bool
                                out40_: _dafny.Seq
                                out38_, out39_, out40_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_40_next_)
                                d_47_g2_ = out38_
                                d_48_i2_ = out39_
                                d_49_c2_ = out40_
                                generated = d_47_g2_
                                insideConstrainedOut = d_48_i2_
                                currentConstrainedOut = d_49_c2_
                                d_3_spanTokensUsed_ = (d_3_spanTokensUsed_) + (1)
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_50_gRolled_: _dafny.Seq
            d_51_cRolled_: _dafny.Seq
            out41_: _dafny.Seq
            out42_: _dafny.Seq
            out41_, out42_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
            d_50_gRolled_ = out41_
            d_51_cRolled_ = out42_
            generated = d_50_gRolled_
            currentConstrainedOut = d_51_cRolled_
            if (parser).IsCompletePrefix(currentConstrainedOut):
                d_52_g2_: _dafny.Seq
                d_53_i2_: bool
                d_54_c2_: _dafny.Seq
                out43_: _dafny.Seq
                out44_: bool
                out45_: _dafny.Seq
                out43_, out44_, out45_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_52_g2_ = out43_
                d_53_i2_ = out44_
                d_54_c2_ = out45_
                generated = d_52_g2_
                insideConstrainedOut = d_53_i2_
                currentConstrainedOut = d_54_c2_
                d_1_steps_ = (d_1_steps_) + (1)
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

