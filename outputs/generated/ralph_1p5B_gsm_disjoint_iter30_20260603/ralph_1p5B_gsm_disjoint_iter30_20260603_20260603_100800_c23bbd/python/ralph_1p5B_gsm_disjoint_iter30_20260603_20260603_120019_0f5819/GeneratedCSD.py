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
        d_2_freeChunkSize_ = 10
        d_3_spanTokensUsed_: int
        d_3_spanTokensUsed_ = 0
        d_4_spanMaxTokens_: int
        d_4_spanMaxTokens_ = 50
        d_5_lastToken_: _dafny.Seq
        d_5_lastToken_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
        d_6_repeatCount_: int
        d_6_repeatCount_ = 0
        d_7_maxRepeat_: int
        d_7_maxRepeat_ = 5
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
                cost = d_1_steps_
                return generated, insideConstrainedOut, currentConstrainedOut, cost
            elif True:
                d_15_constrainedPrompt_: _dafny.Seq
                d_15_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                d_16_next_: _dafny.Seq
                out7_: _dafny.Seq
                out7_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_15_constrainedPrompt_, currentConstrainedOut, eosToken)
                d_16_next_ = out7_
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
                    d_17_g2_: _dafny.Seq
                    d_18_i2_: bool
                    d_19_c2_: _dafny.Seq
                    out8_: _dafny.Seq
                    out9_: bool
                    out10_: _dafny.Seq
                    out8_, out9_, out10_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_16_next_)
                    d_17_g2_ = out8_
                    d_18_i2_ = out9_
                    d_19_c2_ = out10_
                    generated = d_17_g2_
                    insideConstrainedOut = d_18_i2_
                    currentConstrainedOut = d_19_c2_
                    d_3_spanTokensUsed_ = (d_3_spanTokensUsed_) + (1)
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_20_remaining_: int
                        d_20_remaining_ = (maxSteps) - (d_1_steps_)
                        d_21_chunkBudget_: int
                        if (d_20_remaining_) < (d_2_freeChunkSize_):
                            d_21_chunkBudget_ = d_20_remaining_
                        elif True:
                            d_21_chunkBudget_ = d_2_freeChunkSize_
                        if (d_21_chunkBudget_) == (0):
                            raise _dafny.Break("0")
                        d_22_chunkGenerated_: _dafny.Seq
                        d_23_stoppedOnOpenSpan_: bool
                        d_24_stoppedOnEos_: bool
                        d_25_stepsUsed_: int
                        out11_: _dafny.Seq
                        out12_: bool
                        out13_: bool
                        out14_: int
                        out11_, out12_, out13_, out14_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_21_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_22_chunkGenerated_ = out11_
                        d_23_stoppedOnOpenSpan_ = out12_
                        d_24_stoppedOnEos_ = out13_
                        d_25_stepsUsed_ = out14_
                        generated = d_22_chunkGenerated_
                        d_1_steps_ = (d_1_steps_) + (d_25_stepsUsed_)
                        if d_24_stoppedOnEos_:
                            raise _dafny.Break("0")
                        elif d_23_stoppedOnOpenSpan_:
                            d_26_g2_: _dafny.Seq
                            d_27_i2_: bool
                            d_28_c2_: _dafny.Seq
                            out15_: _dafny.Seq
                            out16_: bool
                            out17_: _dafny.Seq
                            out15_, out16_, out17_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                            d_26_g2_ = out15_
                            d_27_i2_ = out16_
                            d_28_c2_ = out17_
                            generated = d_26_g2_
                            insideConstrainedOut = d_27_i2_
                            currentConstrainedOut = d_28_c2_
                            d_3_spanTokensUsed_ = 0
                            d_5_lastToken_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                            d_6_repeatCount_ = 0
                        elif True:
                            if (d_1_steps_) < (maxSteps):
                                d_29_g2_: _dafny.Seq
                                d_30_i2_: bool
                                d_31_c2_: _dafny.Seq
                                out18_: _dafny.Seq
                                out19_: bool
                                out20_: _dafny.Seq
                                out18_, out19_, out20_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                d_29_g2_ = out18_
                                d_30_i2_ = out19_
                                d_31_c2_ = out20_
                                generated = d_29_g2_
                                insideConstrainedOut = d_30_i2_
                                currentConstrainedOut = d_31_c2_
                                d_1_steps_ = (d_1_steps_) + (1)
                                d_3_spanTokensUsed_ = 0
                                d_5_lastToken_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                                d_6_repeatCount_ = 0
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_32_g2_: _dafny.Seq
                        d_33_i2_: bool
                        d_34_c2_: _dafny.Seq
                        out21_: _dafny.Seq
                        out22_: bool
                        out23_: _dafny.Seq
                        out21_, out22_, out23_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_32_g2_ = out21_
                        d_33_i2_ = out22_
                        d_34_c2_ = out23_
                        generated = d_32_g2_
                        insideConstrainedOut = d_33_i2_
                        currentConstrainedOut = d_34_c2_
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_3_spanTokensUsed_ = 0
                        d_5_lastToken_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                        d_6_repeatCount_ = 0
                    elif ((d_3_spanTokensUsed_) >= (d_4_spanMaxTokens_)) or ((d_6_repeatCount_) >= (d_7_maxRepeat_)):
                        d_35_gRolled_: _dafny.Seq
                        d_36_cRolled_: _dafny.Seq
                        out24_: _dafny.Seq
                        out25_: _dafny.Seq
                        out24_, out25_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                        d_35_gRolled_ = out24_
                        d_36_cRolled_ = out25_
                        generated = d_35_gRolled_
                        currentConstrainedOut = d_36_cRolled_
                        d_3_spanTokensUsed_ = 0
                        d_5_lastToken_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                        d_6_repeatCount_ = 0
                        if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                            d_37_g2_: _dafny.Seq
                            d_38_i2_: bool
                            d_39_c2_: _dafny.Seq
                            out26_: _dafny.Seq
                            out27_: bool
                            out28_: _dafny.Seq
                            out26_, out27_, out28_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_37_g2_ = out26_
                            d_38_i2_ = out27_
                            d_39_c2_ = out28_
                            generated = d_37_g2_
                            insideConstrainedOut = d_38_i2_
                            currentConstrainedOut = d_39_c2_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            raise _dafny.Break("0")
                    elif True:
                        d_40_constrainedPrompt_: _dafny.Seq
                        d_40_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_41_next_: _dafny.Seq
                        out29_: _dafny.Seq
                        out29_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_40_constrainedPrompt_, currentConstrainedOut, eosToken)
                        d_41_next_ = out29_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_41_next_) == (eosToken):
                            d_42_gRolled_: _dafny.Seq
                            d_43_cRolled_: _dafny.Seq
                            out30_: _dafny.Seq
                            out31_: _dafny.Seq
                            out30_, out31_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                            d_42_gRolled_ = out30_
                            d_43_cRolled_ = out31_
                            generated = d_42_gRolled_
                            currentConstrainedOut = d_43_cRolled_
                            d_3_spanTokensUsed_ = 0
                            d_5_lastToken_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                            d_6_repeatCount_ = 0
                            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                d_44_g2_: _dafny.Seq
                                d_45_i2_: bool
                                d_46_c2_: _dafny.Seq
                                out32_: _dafny.Seq
                                out33_: bool
                                out34_: _dafny.Seq
                                out32_, out33_, out34_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_44_g2_ = out32_
                                d_45_i2_ = out33_
                                d_46_c2_ = out34_
                                generated = d_44_g2_
                                insideConstrainedOut = d_45_i2_
                                currentConstrainedOut = d_46_c2_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                raise _dafny.Break("0")
                        elif True:
                            if (d_41_next_) == (d_5_lastToken_):
                                d_6_repeatCount_ = (d_6_repeatCount_) + (1)
                            elif True:
                                d_5_lastToken_ = d_41_next_
                                d_6_repeatCount_ = 1
                            d_47_g2_: _dafny.Seq
                            d_48_i2_: bool
                            d_49_c2_: _dafny.Seq
                            out35_: _dafny.Seq
                            out36_: bool
                            out37_: _dafny.Seq
                            out35_, out36_, out37_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_41_next_)
                            d_47_g2_ = out35_
                            d_48_i2_ = out36_
                            d_49_c2_ = out37_
                            generated = d_47_g2_
                            insideConstrainedOut = d_48_i2_
                            currentConstrainedOut = d_49_c2_
                            d_3_spanTokensUsed_ = (d_3_spanTokensUsed_) + (1)
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

