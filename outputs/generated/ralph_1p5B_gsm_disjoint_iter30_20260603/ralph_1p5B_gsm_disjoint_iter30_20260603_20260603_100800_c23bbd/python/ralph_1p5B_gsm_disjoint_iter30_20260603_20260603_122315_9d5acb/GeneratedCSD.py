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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. Wrap each arithmetic expression and the final answer in << >>.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_freeChunkSize_: int
        d_2_freeChunkSize_ = 60
        d_3_spanCount_: int
        d_3_spanCount_ = 0
        d_4_maxSpans_: int
        d_4_maxSpans_ = 5
        d_5_spanTokensUsed_: int
        d_5_spanTokensUsed_ = 0
        d_6_spanMaxTokens_: int
        d_6_spanMaxTokens_ = 25
        d_7_lastToken_: _dafny.Seq
        d_7_lastToken_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
        d_8_repeatCount_: int
        d_8_repeatCount_ = 0
        d_9_maxRepeat_: int
        d_9_maxRepeat_ = 3
        if (maxSteps) > (0):
            if not(insideConstrainedOut):
                d_10_next_: _dafny.Seq
                out0_: _dafny.Seq
                out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                d_10_next_ = out0_
                d_1_steps_ = (d_1_steps_) + (1)
                if (d_10_next_) == (eosToken):
                    cost = d_1_steps_
                    return generated, insideConstrainedOut, currentConstrainedOut, cost
                elif True:
                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_10_next_]))
                    if (d_10_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                        d_11_g2_: _dafny.Seq
                        d_12_i2_: bool
                        d_13_c2_: _dafny.Seq
                        out1_: _dafny.Seq
                        out2_: bool
                        out3_: _dafny.Seq
                        out1_, out2_, out3_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                        d_11_g2_ = out1_
                        d_12_i2_ = out2_
                        d_13_c2_ = out3_
                        generated = d_11_g2_
                        insideConstrainedOut = d_12_i2_
                        currentConstrainedOut = d_13_c2_
                        d_3_spanCount_ = (d_3_spanCount_) + (1)
                        d_5_spanTokensUsed_ = 0
                        d_7_lastToken_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                        d_8_repeatCount_ = 0
            elif (parser).IsCompletePrefix(currentConstrainedOut):
                d_14_g2_: _dafny.Seq
                d_15_i2_: bool
                d_16_c2_: _dafny.Seq
                out4_: _dafny.Seq
                out5_: bool
                out6_: _dafny.Seq
                out4_, out5_, out6_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_14_g2_ = out4_
                d_15_i2_ = out5_
                d_16_c2_ = out6_
                generated = d_14_g2_
                insideConstrainedOut = d_15_i2_
                currentConstrainedOut = d_16_c2_
                d_1_steps_ = (d_1_steps_) + (1)
                d_5_spanTokensUsed_ = 0
                d_7_lastToken_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                d_8_repeatCount_ = 0
                cost = d_1_steps_
                return generated, insideConstrainedOut, currentConstrainedOut, cost
            elif True:
                d_17_constrainedPrompt_: _dafny.Seq
                d_17_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                d_18_next_: _dafny.Seq
                out7_: _dafny.Seq
                out7_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_17_constrainedPrompt_, currentConstrainedOut, eosToken)
                d_18_next_ = out7_
                d_1_steps_ = (d_1_steps_) + (1)
                if (d_18_next_) == (eosToken):
                    cost = d_1_steps_
                    return generated, insideConstrainedOut, currentConstrainedOut, cost
                elif True:
                    if (d_18_next_) == (d_7_lastToken_):
                        d_8_repeatCount_ = (d_8_repeatCount_) + (1)
                    elif True:
                        d_7_lastToken_ = d_18_next_
                        d_8_repeatCount_ = 1
                    d_19_g2_: _dafny.Seq
                    d_20_i2_: bool
                    d_21_c2_: _dafny.Seq
                    out8_: _dafny.Seq
                    out9_: bool
                    out10_: _dafny.Seq
                    out8_, out9_, out10_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_18_next_)
                    d_19_g2_ = out8_
                    d_20_i2_ = out9_
                    d_21_c2_ = out10_
                    generated = d_19_g2_
                    insideConstrainedOut = d_20_i2_
                    currentConstrainedOut = d_21_c2_
                    d_5_spanTokensUsed_ = (d_5_spanTokensUsed_) + (1)
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_22_remaining_: int
                        d_22_remaining_ = (maxSteps) - (d_1_steps_)
                        d_23_chunkBudget_: int
                        if (d_22_remaining_) < (d_2_freeChunkSize_):
                            d_23_chunkBudget_ = d_22_remaining_
                        elif True:
                            d_23_chunkBudget_ = d_2_freeChunkSize_
                        if (d_23_chunkBudget_) == (0):
                            raise _dafny.Break("0")
                        d_24_chunkGenerated_: _dafny.Seq
                        d_25_stoppedOnOpenSpan_: bool
                        d_26_stoppedOnEos_: bool
                        d_27_stepsUsed_: int
                        out11_: _dafny.Seq
                        out12_: bool
                        out13_: bool
                        out14_: int
                        out11_, out12_, out13_, out14_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_23_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_24_chunkGenerated_ = out11_
                        d_25_stoppedOnOpenSpan_ = out12_
                        d_26_stoppedOnEos_ = out13_
                        d_27_stepsUsed_ = out14_
                        generated = d_24_chunkGenerated_
                        d_1_steps_ = (d_1_steps_) + (d_27_stepsUsed_)
                        if d_26_stoppedOnEos_:
                            raise _dafny.Break("0")
                        elif d_25_stoppedOnOpenSpan_:
                            d_28_g2_: _dafny.Seq
                            d_29_i2_: bool
                            d_30_c2_: _dafny.Seq
                            out15_: _dafny.Seq
                            out16_: bool
                            out17_: _dafny.Seq
                            out15_, out16_, out17_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                            d_28_g2_ = out15_
                            d_29_i2_ = out16_
                            d_30_c2_ = out17_
                            generated = d_28_g2_
                            insideConstrainedOut = d_29_i2_
                            currentConstrainedOut = d_30_c2_
                            d_3_spanCount_ = (d_3_spanCount_) + (1)
                            d_5_spanTokensUsed_ = 0
                            d_7_lastToken_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                            d_8_repeatCount_ = 0
                        elif True:
                            if ((d_3_spanCount_) < (d_4_maxSpans_)) and ((d_1_steps_) < (maxSteps)):
                                d_31_g2_: _dafny.Seq
                                d_32_i2_: bool
                                d_33_c2_: _dafny.Seq
                                out18_: _dafny.Seq
                                out19_: bool
                                out20_: _dafny.Seq
                                out18_, out19_, out20_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                d_31_g2_ = out18_
                                d_32_i2_ = out19_
                                d_33_c2_ = out20_
                                generated = d_31_g2_
                                insideConstrainedOut = d_32_i2_
                                currentConstrainedOut = d_33_c2_
                                d_1_steps_ = (d_1_steps_) + (1)
                                d_3_spanCount_ = (d_3_spanCount_) + (1)
                                d_5_spanTokensUsed_ = 0
                                d_7_lastToken_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                                d_8_repeatCount_ = 0
                            elif True:
                                if (d_1_steps_) < (maxSteps):
                                    d_34_next_: _dafny.Seq
                                    out21_: _dafny.Seq
                                    out21_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                                    d_34_next_ = out21_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    if (d_34_next_) == (eosToken):
                                        raise _dafny.Break("0")
                                    elif True:
                                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_34_next_]))
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_35_g2_: _dafny.Seq
                        d_36_i2_: bool
                        d_37_c2_: _dafny.Seq
                        out22_: _dafny.Seq
                        out23_: bool
                        out24_: _dafny.Seq
                        out22_, out23_, out24_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_35_g2_ = out22_
                        d_36_i2_ = out23_
                        d_37_c2_ = out24_
                        generated = d_35_g2_
                        insideConstrainedOut = d_36_i2_
                        currentConstrainedOut = d_37_c2_
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_5_spanTokensUsed_ = 0
                        d_7_lastToken_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                        d_8_repeatCount_ = 0
                    elif ((d_5_spanTokensUsed_) >= (d_6_spanMaxTokens_)) or ((d_8_repeatCount_) >= (d_9_maxRepeat_)):
                        d_38_gRolled_: _dafny.Seq
                        d_39_cRolled_: _dafny.Seq
                        out25_: _dafny.Seq
                        out26_: _dafny.Seq
                        out25_, out26_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                        d_38_gRolled_ = out25_
                        d_39_cRolled_ = out26_
                        generated = d_38_gRolled_
                        currentConstrainedOut = d_39_cRolled_
                        d_5_spanTokensUsed_ = 0
                        d_7_lastToken_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                        d_8_repeatCount_ = 0
                        if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                            d_40_g2_: _dafny.Seq
                            d_41_i2_: bool
                            d_42_c2_: _dafny.Seq
                            out27_: _dafny.Seq
                            out28_: bool
                            out29_: _dafny.Seq
                            out27_, out28_, out29_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_40_g2_ = out27_
                            d_41_i2_ = out28_
                            d_42_c2_ = out29_
                            generated = d_40_g2_
                            insideConstrainedOut = d_41_i2_
                            currentConstrainedOut = d_42_c2_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif (d_1_steps_) < (maxSteps):
                            d_43_constrainedPrompt_: _dafny.Seq
                            d_43_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_44_next_: _dafny.Seq
                            out30_: _dafny.Seq
                            out30_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_43_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_44_next_ = out30_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_44_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_45_g2_: _dafny.Seq
                                d_46_i2_: bool
                                d_47_c2_: _dafny.Seq
                                out31_: _dafny.Seq
                                out32_: bool
                                out33_: _dafny.Seq
                                out31_, out32_, out33_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_44_next_)
                                d_45_g2_ = out31_
                                d_46_i2_ = out32_
                                d_47_c2_ = out33_
                                generated = d_45_g2_
                                insideConstrainedOut = d_46_i2_
                                currentConstrainedOut = d_47_c2_
                                d_5_spanTokensUsed_ = (d_5_spanTokensUsed_) + (1)
                        elif True:
                            raise _dafny.Break("0")
                    elif True:
                        d_48_constrainedPrompt_: _dafny.Seq
                        d_48_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_49_next_: _dafny.Seq
                        out34_: _dafny.Seq
                        out34_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_48_constrainedPrompt_, currentConstrainedOut, eosToken)
                        d_49_next_ = out34_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_49_next_) == (eosToken):
                            d_50_gRolled_: _dafny.Seq
                            d_51_cRolled_: _dafny.Seq
                            out35_: _dafny.Seq
                            out36_: _dafny.Seq
                            out35_, out36_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                            d_50_gRolled_ = out35_
                            d_51_cRolled_ = out36_
                            generated = d_50_gRolled_
                            currentConstrainedOut = d_51_cRolled_
                            d_5_spanTokensUsed_ = 0
                            d_7_lastToken_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                            d_8_repeatCount_ = 0
                            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                d_52_g2_: _dafny.Seq
                                d_53_i2_: bool
                                d_54_c2_: _dafny.Seq
                                out37_: _dafny.Seq
                                out38_: bool
                                out39_: _dafny.Seq
                                out37_, out38_, out39_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_52_g2_ = out37_
                                d_53_i2_ = out38_
                                d_54_c2_ = out39_
                                generated = d_52_g2_
                                insideConstrainedOut = d_53_i2_
                                currentConstrainedOut = d_54_c2_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif (d_1_steps_) < (maxSteps):
                                d_55_constrainedPrompt2_: _dafny.Seq
                                d_55_constrainedPrompt2_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_56_next2_: _dafny.Seq
                                out40_: _dafny.Seq
                                out40_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_55_constrainedPrompt2_, currentConstrainedOut, eosToken)
                                d_56_next2_ = out40_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_56_next2_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_57_g2_: _dafny.Seq
                                    d_58_i2_: bool
                                    d_59_c2_: _dafny.Seq
                                    out41_: _dafny.Seq
                                    out42_: bool
                                    out43_: _dafny.Seq
                                    out41_, out42_, out43_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_56_next2_)
                                    d_57_g2_ = out41_
                                    d_58_i2_ = out42_
                                    d_59_c2_ = out43_
                                    generated = d_57_g2_
                                    insideConstrainedOut = d_58_i2_
                                    currentConstrainedOut = d_59_c2_
                                    d_5_spanTokensUsed_ = (d_5_spanTokensUsed_) + (1)
                            elif True:
                                raise _dafny.Break("0")
                        elif True:
                            if (d_49_next_) == (d_7_lastToken_):
                                d_8_repeatCount_ = (d_8_repeatCount_) + (1)
                            elif True:
                                d_7_lastToken_ = d_49_next_
                                d_8_repeatCount_ = 1
                            d_60_g2_: _dafny.Seq
                            d_61_i2_: bool
                            d_62_c2_: _dafny.Seq
                            out44_: _dafny.Seq
                            out45_: bool
                            out46_: _dafny.Seq
                            out44_, out45_, out46_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_49_next_)
                            d_60_g2_ = out44_
                            d_61_i2_ = out45_
                            d_62_c2_ = out46_
                            generated = d_60_g2_
                            insideConstrainedOut = d_61_i2_
                            currentConstrainedOut = d_62_c2_
                            d_5_spanTokensUsed_ = (d_5_spanTokensUsed_) + (1)
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

