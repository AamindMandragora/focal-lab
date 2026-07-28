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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. Wrap each arithmetic expression and the final answer in << >>. Keep expressions simple: <<a + b>>, <<x * y>>.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_freeChunkSize_: int
        d_2_freeChunkSize_ = 45
        d_3_spanTokensUsed_: int
        d_3_spanTokensUsed_ = 0
        d_4_spanMaxTokens_: int
        d_4_spanMaxTokens_ = 8
        d_5_lastToken_: _dafny.Seq
        d_5_lastToken_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
        d_6_repeatCount_: int
        d_6_repeatCount_ = 0
        d_7_maxRepeat_: int
        d_7_maxRepeat_ = 3
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
                        if ((d_21_remaining_) <= (40)) and ((d_21_remaining_) > (0)):
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
                    elif True:
                        d_36_isDeadEnd_: bool
                        out25_: bool
                        out25_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 2)
                        d_36_isDeadEnd_ = out25_
                        if ((d_36_isDeadEnd_) or ((d_3_spanTokensUsed_) >= (d_4_spanMaxTokens_))) or ((d_6_repeatCount_) >= (d_7_maxRepeat_)):
                            d_37_gRolled_: _dafny.Seq
                            d_38_cRolled_: _dafny.Seq
                            out26_: _dafny.Seq
                            out27_: _dafny.Seq
                            out26_, out27_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                            d_37_gRolled_ = out26_
                            d_38_cRolled_ = out27_
                            generated = d_37_gRolled_
                            currentConstrainedOut = d_38_cRolled_
                            d_3_spanTokensUsed_ = 0
                            d_5_lastToken_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                            d_6_repeatCount_ = 0
                            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                d_39_g2_: _dafny.Seq
                                d_40_i2_: bool
                                d_41_c2_: _dafny.Seq
                                out28_: _dafny.Seq
                                out29_: bool
                                out30_: _dafny.Seq
                                out28_, out29_, out30_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_39_g2_ = out28_
                                d_40_i2_ = out29_
                                d_41_c2_ = out30_
                                generated = d_39_g2_
                                insideConstrainedOut = d_40_i2_
                                currentConstrainedOut = d_41_c2_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif (d_1_steps_) < (maxSteps):
                                d_42_constrainedPrompt_: _dafny.Seq
                                d_42_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_43_next_: _dafny.Seq
                                d_44_wasConstrained_: bool
                                out31_: _dafny.Seq
                                out32_: bool
                                out31_, out32_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, d_42_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_43_next_ = out31_
                                d_44_wasConstrained_ = out32_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_43_next_) != (eosToken):
                                    d_45_g2_: _dafny.Seq
                                    d_46_i2_: bool
                                    d_47_c2_: _dafny.Seq
                                    out33_: _dafny.Seq
                                    out34_: bool
                                    out35_: _dafny.Seq
                                    out33_, out34_, out35_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_43_next_)
                                    d_45_g2_ = out33_
                                    d_46_i2_ = out34_
                                    d_47_c2_ = out35_
                                    generated = d_45_g2_
                                    insideConstrainedOut = d_46_i2_
                                    currentConstrainedOut = d_47_c2_
                                    d_3_spanTokensUsed_ = (d_3_spanTokensUsed_) + (1)
                                    if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                        d_48_g3_: _dafny.Seq
                                        d_49_i3_: bool
                                        d_50_c3_: _dafny.Seq
                                        out36_: _dafny.Seq
                                        out37_: bool
                                        out38_: _dafny.Seq
                                        out36_, out37_, out38_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                        d_48_g3_ = out36_
                                        d_49_i3_ = out37_
                                        d_50_c3_ = out38_
                                        generated = d_48_g3_
                                        insideConstrainedOut = d_49_i3_
                                        currentConstrainedOut = d_50_c3_
                                        d_1_steps_ = (d_1_steps_) + (1)
                                elif True:
                                    d_51_gR2_: _dafny.Seq
                                    d_52_cR2_: _dafny.Seq
                                    out39_: _dafny.Seq
                                    out40_: _dafny.Seq
                                    out39_, out40_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                    d_51_gR2_ = out39_
                                    d_52_cR2_ = out40_
                                    generated = d_51_gR2_
                                    currentConstrainedOut = d_52_cR2_
                                    if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                        d_53_g2_: _dafny.Seq
                                        d_54_i2_: bool
                                        d_55_c2_: _dafny.Seq
                                        out41_: _dafny.Seq
                                        out42_: bool
                                        out43_: _dafny.Seq
                                        out41_, out42_, out43_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                        d_53_g2_ = out41_
                                        d_54_i2_ = out42_
                                        d_55_c2_ = out43_
                                        generated = d_53_g2_
                                        insideConstrainedOut = d_54_i2_
                                        currentConstrainedOut = d_55_c2_
                                        d_1_steps_ = (d_1_steps_) + (1)
                                    elif True:
                                        raise _dafny.Break("0")
                        elif True:
                            d_56_constrainedPrompt_: _dafny.Seq
                            d_56_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_57_next_: _dafny.Seq
                            d_58_wasConstrained_: bool
                            out44_: _dafny.Seq
                            out45_: bool
                            out44_, out45_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, d_56_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_57_next_ = out44_
                            d_58_wasConstrained_ = out45_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_57_next_) == (eosToken):
                                d_59_gRolled_: _dafny.Seq
                                d_60_cRolled_: _dafny.Seq
                                out46_: _dafny.Seq
                                out47_: _dafny.Seq
                                out46_, out47_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                d_59_gRolled_ = out46_
                                d_60_cRolled_ = out47_
                                generated = d_59_gRolled_
                                currentConstrainedOut = d_60_cRolled_
                                d_3_spanTokensUsed_ = 0
                                d_5_lastToken_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                                d_6_repeatCount_ = 0
                                if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                    d_61_g2_: _dafny.Seq
                                    d_62_i2_: bool
                                    d_63_c2_: _dafny.Seq
                                    out48_: _dafny.Seq
                                    out49_: bool
                                    out50_: _dafny.Seq
                                    out48_, out49_, out50_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_61_g2_ = out48_
                                    d_62_i2_ = out49_
                                    d_63_c2_ = out50_
                                    generated = d_61_g2_
                                    insideConstrainedOut = d_62_i2_
                                    currentConstrainedOut = d_63_c2_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                elif True:
                                    raise _dafny.Break("0")
                            elif True:
                                if (d_57_next_) == (d_5_lastToken_):
                                    d_6_repeatCount_ = (d_6_repeatCount_) + (1)
                                elif True:
                                    d_5_lastToken_ = d_57_next_
                                    d_6_repeatCount_ = 1
                                d_64_g2_: _dafny.Seq
                                d_65_i2_: bool
                                d_66_c2_: _dafny.Seq
                                out51_: _dafny.Seq
                                out52_: bool
                                out53_: _dafny.Seq
                                out51_, out52_, out53_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_57_next_)
                                d_64_g2_ = out51_
                                d_65_i2_ = out52_
                                d_66_c2_ = out53_
                                generated = d_64_g2_
                                insideConstrainedOut = d_65_i2_
                                currentConstrainedOut = d_66_c2_
                                d_3_spanTokensUsed_ = (d_3_spanTokensUsed_) + (1)
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_67_gRolled_: _dafny.Seq
            d_68_cRolled_: _dafny.Seq
            out54_: _dafny.Seq
            out55_: _dafny.Seq
            out54_, out55_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
            d_67_gRolled_ = out54_
            d_68_cRolled_ = out55_
            generated = d_67_gRolled_
            currentConstrainedOut = d_68_cRolled_
            if (parser).IsCompletePrefix(currentConstrainedOut):
                d_69_g2_: _dafny.Seq
                d_70_i2_: bool
                d_71_c2_: _dafny.Seq
                out56_: _dafny.Seq
                out57_: bool
                out58_: _dafny.Seq
                out56_, out57_, out58_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_69_g2_ = out56_
                d_70_i2_ = out57_
                d_71_c2_ = out58_
                generated = d_69_g2_
                insideConstrainedOut = d_70_i2_
                currentConstrainedOut = d_71_c2_
                d_1_steps_ = (d_1_steps_) + (1)
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

