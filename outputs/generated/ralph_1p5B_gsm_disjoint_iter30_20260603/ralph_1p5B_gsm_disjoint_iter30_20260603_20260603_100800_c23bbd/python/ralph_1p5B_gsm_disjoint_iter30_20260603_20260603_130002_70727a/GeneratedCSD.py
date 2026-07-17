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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. For each calculation and the final answer, write the expression inside << >>. Example: The total is <<3 * 4>>.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_freeChunkSize_: int
        d_2_freeChunkSize_ = 40
        d_3_spanTokensUsed_: int
        d_3_spanTokensUsed_ = 0
        d_4_spanMaxTokens_: int
        d_4_spanMaxTokens_ = 25
        d_5_lastToken_: _dafny.Seq
        d_5_lastToken_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
        d_6_repeatCount_: int
        d_6_repeatCount_ = 0
        d_7_maxRepeat_: int
        d_7_maxRepeat_ = 4
        if (maxSteps) > (0):
            if not(insideConstrainedOut):
                d_8_next_: _dafny.Seq
                out0_: _dafny.Seq
                out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                d_8_next_ = out0_
                d_1_steps_ = (d_1_steps_) + (1)
                if (d_8_next_) == (eosToken):
                    cost = d_1_steps_
                    return generated, insideConstrainedOut, currentConstrainedOut, cost
                elif True:
                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_8_next_]))
                    if (d_8_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                        d_9_g2_: _dafny.Seq
                        d_10_i2_: bool
                        d_11_c2_: _dafny.Seq
                        out1_: _dafny.Seq
                        out2_: bool
                        out3_: _dafny.Seq
                        out1_, out2_, out3_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                        d_9_g2_ = out1_
                        d_10_i2_ = out2_
                        d_11_c2_ = out3_
                        generated = d_9_g2_
                        insideConstrainedOut = d_10_i2_
                        currentConstrainedOut = d_11_c2_
                        d_3_spanTokensUsed_ = 0
                        d_5_lastToken_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                        d_6_repeatCount_ = 0
            elif (parser).IsCompletePrefix(currentConstrainedOut):
                d_12_g2_: _dafny.Seq
                d_13_i2_: bool
                d_14_c2_: _dafny.Seq
                out4_: _dafny.Seq
                out5_: bool
                out6_: _dafny.Seq
                out4_, out5_, out6_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_12_g2_ = out4_
                d_13_i2_ = out5_
                d_14_c2_ = out6_
                generated = d_12_g2_
                insideConstrainedOut = d_13_i2_
                currentConstrainedOut = d_14_c2_
                d_1_steps_ = (d_1_steps_) + (1)
                d_3_spanTokensUsed_ = 0
                d_5_lastToken_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                d_6_repeatCount_ = 0
            elif True:
                d_15_constrainedPrompt_: _dafny.Seq
                d_15_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                d_16_next_: _dafny.Seq
                d_17_wasConstrained_: bool
                out7_: _dafny.Seq
                out8_: bool
                out7_, out8_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, d_15_constrainedPrompt_, currentConstrainedOut, eosToken)
                d_16_next_ = out7_
                d_17_wasConstrained_ = out8_
                d_1_steps_ = (d_1_steps_) + (1)
                if (d_16_next_) == (eosToken):
                    cost = d_1_steps_
                    return generated, insideConstrainedOut, currentConstrainedOut, cost
                elif True:
                    if (d_16_next_) == (d_5_lastToken_):
                        d_6_repeatCount_ = (d_6_repeatCount_) + (1)
                    elif True:
                        d_5_lastToken_ = d_16_next_
                        d_6_repeatCount_ = 1
                    d_18_g2_: _dafny.Seq
                    d_19_i2_: bool
                    d_20_c2_: _dafny.Seq
                    out9_: _dafny.Seq
                    out10_: bool
                    out11_: _dafny.Seq
                    out9_, out10_, out11_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_16_next_)
                    d_18_g2_ = out9_
                    d_19_i2_ = out10_
                    d_20_c2_ = out11_
                    generated = d_18_g2_
                    insideConstrainedOut = d_19_i2_
                    currentConstrainedOut = d_20_c2_
                    d_3_spanTokensUsed_ = (d_3_spanTokensUsed_) + (1)
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_21_remaining_: int
                        d_21_remaining_ = (maxSteps) - (d_1_steps_)
                        if ((d_21_remaining_) <= (35)) and ((d_21_remaining_) > (0)):
                            d_22_g2_: _dafny.Seq
                            d_23_i2_: bool
                            d_24_c2_: _dafny.Seq
                            out12_: _dafny.Seq
                            out13_: bool
                            out14_: _dafny.Seq
                            out12_, out13_, out14_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_22_g2_ = out12_
                            d_23_i2_ = out13_
                            d_24_c2_ = out14_
                            generated = d_22_g2_
                            insideConstrainedOut = d_23_i2_
                            currentConstrainedOut = d_24_c2_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_3_spanTokensUsed_ = 0
                            d_5_lastToken_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                            d_6_repeatCount_ = 0
                        elif True:
                            d_25_chunkBudget_: int
                            if (d_21_remaining_) < (d_2_freeChunkSize_):
                                d_25_chunkBudget_ = d_21_remaining_
                            elif True:
                                d_25_chunkBudget_ = d_2_freeChunkSize_
                            if (d_25_chunkBudget_) == (0):
                                raise _dafny.Break("0")
                            d_26_chunkGenerated_: _dafny.Seq
                            d_27_stoppedOnOpenSpan_: bool
                            d_28_stoppedOnEos_: bool
                            d_29_stepsUsed_: int
                            out15_: _dafny.Seq
                            out16_: bool
                            out17_: bool
                            out18_: int
                            out15_, out16_, out17_, out18_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_25_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_26_chunkGenerated_ = out15_
                            d_27_stoppedOnOpenSpan_ = out16_
                            d_28_stoppedOnEos_ = out17_
                            d_29_stepsUsed_ = out18_
                            generated = d_26_chunkGenerated_
                            d_1_steps_ = (d_1_steps_) + (d_29_stepsUsed_)
                            if d_28_stoppedOnEos_:
                                raise _dafny.Break("0")
                            elif d_27_stoppedOnOpenSpan_:
                                d_30_g2_: _dafny.Seq
                                d_31_i2_: bool
                                d_32_c2_: _dafny.Seq
                                out19_: _dafny.Seq
                                out20_: bool
                                out21_: _dafny.Seq
                                out19_, out20_, out21_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_30_g2_ = out19_
                                d_31_i2_ = out20_
                                d_32_c2_ = out21_
                                generated = d_30_g2_
                                insideConstrainedOut = d_31_i2_
                                currentConstrainedOut = d_32_c2_
                                d_3_spanTokensUsed_ = 0
                                d_5_lastToken_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                                d_6_repeatCount_ = 0
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_33_g2_: _dafny.Seq
                        d_34_i2_: bool
                        d_35_c2_: _dafny.Seq
                        out22_: _dafny.Seq
                        out23_: bool
                        out24_: _dafny.Seq
                        out22_, out23_, out24_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_33_g2_ = out22_
                        d_34_i2_ = out23_
                        d_35_c2_ = out24_
                        generated = d_33_g2_
                        insideConstrainedOut = d_34_i2_
                        currentConstrainedOut = d_35_c2_
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_3_spanTokensUsed_ = 0
                        d_5_lastToken_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                        d_6_repeatCount_ = 0
                    elif ((d_3_spanTokensUsed_) >= (d_4_spanMaxTokens_)) or ((d_6_repeatCount_) >= (d_7_maxRepeat_)):
                        d_36_gRolled_: _dafny.Seq
                        d_37_cRolled_: _dafny.Seq
                        out25_: _dafny.Seq
                        out26_: _dafny.Seq
                        out25_, out26_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                        d_36_gRolled_ = out25_
                        d_37_cRolled_ = out26_
                        generated = d_36_gRolled_
                        currentConstrainedOut = d_37_cRolled_
                        d_3_spanTokensUsed_ = 0
                        d_5_lastToken_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                        d_6_repeatCount_ = 0
                        if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                            d_38_g2_: _dafny.Seq
                            d_39_i2_: bool
                            d_40_c2_: _dafny.Seq
                            out27_: _dafny.Seq
                            out28_: bool
                            out29_: _dafny.Seq
                            out27_, out28_, out29_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_38_g2_ = out27_
                            d_39_i2_ = out28_
                            d_40_c2_ = out29_
                            generated = d_38_g2_
                            insideConstrainedOut = d_39_i2_
                            currentConstrainedOut = d_40_c2_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif (d_1_steps_) < (maxSteps):
                            d_41_constrainedPrompt_: _dafny.Seq
                            d_41_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_42_next_: _dafny.Seq
                            d_43_wasConstrained_: bool
                            out30_: _dafny.Seq
                            out31_: bool
                            out30_, out31_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, d_41_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_42_next_ = out30_
                            d_43_wasConstrained_ = out31_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_42_next_) != (eosToken):
                                d_44_g2_: _dafny.Seq
                                d_45_i2_: bool
                                d_46_c2_: _dafny.Seq
                                out32_: _dafny.Seq
                                out33_: bool
                                out34_: _dafny.Seq
                                out32_, out33_, out34_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_42_next_)
                                d_44_g2_ = out32_
                                d_45_i2_ = out33_
                                d_46_c2_ = out34_
                                generated = d_44_g2_
                                insideConstrainedOut = d_45_i2_
                                currentConstrainedOut = d_46_c2_
                                d_3_spanTokensUsed_ = (d_3_spanTokensUsed_) + (1)
                                if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                    d_47_g3_: _dafny.Seq
                                    d_48_i3_: bool
                                    d_49_c3_: _dafny.Seq
                                    out35_: _dafny.Seq
                                    out36_: bool
                                    out37_: _dafny.Seq
                                    out35_, out36_, out37_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_47_g3_ = out35_
                                    d_48_i3_ = out36_
                                    d_49_c3_ = out37_
                                    generated = d_47_g3_
                                    insideConstrainedOut = d_48_i3_
                                    currentConstrainedOut = d_49_c3_
                                    d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_50_constrainedPrompt_: _dafny.Seq
                        d_50_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_51_next_: _dafny.Seq
                        d_52_wasConstrained_: bool
                        out38_: _dafny.Seq
                        out39_: bool
                        out38_, out39_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, d_50_constrainedPrompt_, currentConstrainedOut, eosToken)
                        d_51_next_ = out38_
                        d_52_wasConstrained_ = out39_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_51_next_) == (eosToken):
                            d_53_gRolled_: _dafny.Seq
                            d_54_cRolled_: _dafny.Seq
                            out40_: _dafny.Seq
                            out41_: _dafny.Seq
                            out40_, out41_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                            d_53_gRolled_ = out40_
                            d_54_cRolled_ = out41_
                            generated = d_53_gRolled_
                            currentConstrainedOut = d_54_cRolled_
                            d_3_spanTokensUsed_ = 0
                            d_5_lastToken_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                            d_6_repeatCount_ = 0
                            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                d_55_g2_: _dafny.Seq
                                d_56_i2_: bool
                                d_57_c2_: _dafny.Seq
                                out42_: _dafny.Seq
                                out43_: bool
                                out44_: _dafny.Seq
                                out42_, out43_, out44_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_55_g2_ = out42_
                                d_56_i2_ = out43_
                                d_57_c2_ = out44_
                                generated = d_55_g2_
                                insideConstrainedOut = d_56_i2_
                                currentConstrainedOut = d_57_c2_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                raise _dafny.Break("0")
                        elif True:
                            if (d_51_next_) == (d_5_lastToken_):
                                d_6_repeatCount_ = (d_6_repeatCount_) + (1)
                            elif True:
                                d_5_lastToken_ = d_51_next_
                                d_6_repeatCount_ = 1
                            d_58_g2_: _dafny.Seq
                            d_59_i2_: bool
                            d_60_c2_: _dafny.Seq
                            out45_: _dafny.Seq
                            out46_: bool
                            out47_: _dafny.Seq
                            out45_, out46_, out47_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_51_next_)
                            d_58_g2_ = out45_
                            d_59_i2_ = out46_
                            d_60_c2_ = out47_
                            generated = d_58_g2_
                            insideConstrainedOut = d_59_i2_
                            currentConstrainedOut = d_60_c2_
                            d_3_spanTokensUsed_ = (d_3_spanTokensUsed_) + (1)
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_61_gRolled_: _dafny.Seq
            d_62_cRolled_: _dafny.Seq
            out48_: _dafny.Seq
            out49_: _dafny.Seq
            out48_, out49_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
            d_61_gRolled_ = out48_
            d_62_cRolled_ = out49_
            generated = d_61_gRolled_
            currentConstrainedOut = d_62_cRolled_
            if (parser).IsCompletePrefix(currentConstrainedOut):
                d_63_g2_: _dafny.Seq
                d_64_i2_: bool
                d_65_c2_: _dafny.Seq
                out50_: _dafny.Seq
                out51_: bool
                out52_: _dafny.Seq
                out50_, out51_, out52_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_63_g2_ = out50_
                d_64_i2_ = out51_
                d_65_c2_ = out52_
                generated = d_63_g2_
                insideConstrainedOut = d_64_i2_
                currentConstrainedOut = d_65_c2_
                d_1_steps_ = (d_1_steps_) + (1)
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

