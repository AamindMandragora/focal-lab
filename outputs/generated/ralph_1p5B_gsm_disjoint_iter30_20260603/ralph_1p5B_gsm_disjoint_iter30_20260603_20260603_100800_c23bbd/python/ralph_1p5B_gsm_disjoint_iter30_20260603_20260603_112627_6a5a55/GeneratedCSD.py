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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. For each calculation, wrap the expression in << >>. End with #### <<final_expression>>.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_freeChunkSize_: int
        d_2_freeChunkSize_ = 20
        d_3_spanCount_: int
        d_3_spanCount_ = 0
        d_4_maxSpans_: int
        d_4_maxSpans_ = 20
        d_5_spanTokensUsed_: int
        d_5_spanTokensUsed_ = 0
        d_6_spanMaxTokens_: int
        d_6_spanMaxTokens_ = 15
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
            elif True:
                d_17_constrainedPrompt_: _dafny.Seq
                d_17_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                d_18_next_: _dafny.Seq
                d_19_wasConstrained_: bool
                out7_: _dafny.Seq
                out8_: bool
                out7_, out8_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, d_17_constrainedPrompt_, currentConstrainedOut, eosToken)
                d_18_next_ = out7_
                d_19_wasConstrained_ = out8_
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
                    d_20_g2_: _dafny.Seq
                    d_21_i2_: bool
                    d_22_c2_: _dafny.Seq
                    out9_: _dafny.Seq
                    out10_: bool
                    out11_: _dafny.Seq
                    out9_, out10_, out11_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_18_next_)
                    d_20_g2_ = out9_
                    d_21_i2_ = out10_
                    d_22_c2_ = out11_
                    generated = d_20_g2_
                    insideConstrainedOut = d_21_i2_
                    currentConstrainedOut = d_22_c2_
                    d_5_spanTokensUsed_ = (d_5_spanTokensUsed_) + (1)
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    d_23_remaining_: int
                    d_23_remaining_ = (maxSteps) - (d_1_steps_)
                    if (insideConstrainedOut) and ((d_23_remaining_) <= (8)):
                        d_24_gRolled_: _dafny.Seq
                        d_25_cRolled_: _dafny.Seq
                        out12_: _dafny.Seq
                        out13_: _dafny.Seq
                        out12_, out13_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                        d_24_gRolled_ = out12_
                        d_25_cRolled_ = out13_
                        generated = d_24_gRolled_
                        currentConstrainedOut = d_25_cRolled_
                        d_5_spanTokensUsed_ = 0
                        d_7_lastToken_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                        d_8_repeatCount_ = 0
                        if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                            d_26_g2_: _dafny.Seq
                            d_27_i2_: bool
                            d_28_c2_: _dafny.Seq
                            out14_: _dafny.Seq
                            out15_: bool
                            out16_: _dafny.Seq
                            out14_, out15_, out16_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_26_g2_ = out14_
                            d_27_i2_ = out15_
                            d_28_c2_ = out16_
                            generated = d_26_g2_
                            insideConstrainedOut = d_27_i2_
                            currentConstrainedOut = d_28_c2_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            raise _dafny.Break("0")
                    elif not(insideConstrainedOut):
                        d_29_chunkBudget_: int
                        if (d_23_remaining_) < (d_2_freeChunkSize_):
                            d_29_chunkBudget_ = d_23_remaining_
                        elif True:
                            d_29_chunkBudget_ = d_2_freeChunkSize_
                        d_30_chunkGenerated_: _dafny.Seq
                        d_31_stoppedOnOpenSpan_: bool
                        d_32_stoppedOnEos_: bool
                        d_33_stepsUsed_: int
                        out17_: _dafny.Seq
                        out18_: bool
                        out19_: bool
                        out20_: int
                        out17_, out18_, out19_, out20_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_29_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_30_chunkGenerated_ = out17_
                        d_31_stoppedOnOpenSpan_ = out18_
                        d_32_stoppedOnEos_ = out19_
                        d_33_stepsUsed_ = out20_
                        generated = d_30_chunkGenerated_
                        d_1_steps_ = (d_1_steps_) + (d_33_stepsUsed_)
                        if d_32_stoppedOnEos_:
                            raise _dafny.Break("0")
                        elif d_31_stoppedOnOpenSpan_:
                            d_34_g2_: _dafny.Seq
                            d_35_i2_: bool
                            d_36_c2_: _dafny.Seq
                            out21_: _dafny.Seq
                            out22_: bool
                            out23_: _dafny.Seq
                            out21_, out22_, out23_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                            d_34_g2_ = out21_
                            d_35_i2_ = out22_
                            d_36_c2_ = out23_
                            generated = d_34_g2_
                            insideConstrainedOut = d_35_i2_
                            currentConstrainedOut = d_36_c2_
                            d_3_spanCount_ = (d_3_spanCount_) + (1)
                            d_5_spanTokensUsed_ = 0
                            d_7_lastToken_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                            d_8_repeatCount_ = 0
                        elif True:
                            if ((d_3_spanCount_) < (d_4_maxSpans_)) and ((d_1_steps_) < (maxSteps)):
                                d_37_g2_: _dafny.Seq
                                d_38_i2_: bool
                                d_39_c2_: _dafny.Seq
                                out24_: _dafny.Seq
                                out25_: bool
                                out26_: _dafny.Seq
                                out24_, out25_, out26_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                d_37_g2_ = out24_
                                d_38_i2_ = out25_
                                d_39_c2_ = out26_
                                generated = d_37_g2_
                                insideConstrainedOut = d_38_i2_
                                currentConstrainedOut = d_39_c2_
                                d_1_steps_ = (d_1_steps_) + (1)
                                d_3_spanCount_ = (d_3_spanCount_) + (1)
                                d_5_spanTokensUsed_ = 0
                                d_7_lastToken_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                                d_8_repeatCount_ = 0
                            elif (d_1_steps_) < (maxSteps):
                                d_40_next_: _dafny.Seq
                                out27_: _dafny.Seq
                                out27_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                                d_40_next_ = out27_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_40_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_40_next_]))
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_41_g2_: _dafny.Seq
                        d_42_i2_: bool
                        d_43_c2_: _dafny.Seq
                        out28_: _dafny.Seq
                        out29_: bool
                        out30_: _dafny.Seq
                        out28_, out29_, out30_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_41_g2_ = out28_
                        d_42_i2_ = out29_
                        d_43_c2_ = out30_
                        generated = d_41_g2_
                        insideConstrainedOut = d_42_i2_
                        currentConstrainedOut = d_43_c2_
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_5_spanTokensUsed_ = 0
                        d_7_lastToken_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                        d_8_repeatCount_ = 0
                    elif ((d_5_spanTokensUsed_) >= (d_6_spanMaxTokens_)) or ((d_8_repeatCount_) >= (d_9_maxRepeat_)):
                        d_44_gRolled_: _dafny.Seq
                        d_45_cRolled_: _dafny.Seq
                        out31_: _dafny.Seq
                        out32_: _dafny.Seq
                        out31_, out32_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                        d_44_gRolled_ = out31_
                        d_45_cRolled_ = out32_
                        generated = d_44_gRolled_
                        currentConstrainedOut = d_45_cRolled_
                        d_5_spanTokensUsed_ = 0
                        d_7_lastToken_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                        d_8_repeatCount_ = 0
                        if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                            d_46_g2_: _dafny.Seq
                            d_47_i2_: bool
                            d_48_c2_: _dafny.Seq
                            out33_: _dafny.Seq
                            out34_: bool
                            out35_: _dafny.Seq
                            out33_, out34_, out35_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_46_g2_ = out33_
                            d_47_i2_ = out34_
                            d_48_c2_ = out35_
                            generated = d_46_g2_
                            insideConstrainedOut = d_47_i2_
                            currentConstrainedOut = d_48_c2_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            raise _dafny.Break("0")
                    elif True:
                        d_49_constrainedPrompt_: _dafny.Seq
                        d_49_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_50_next_: _dafny.Seq
                        d_51_wasConstrained_: bool
                        out36_: _dafny.Seq
                        out37_: bool
                        out36_, out37_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, d_49_constrainedPrompt_, currentConstrainedOut, eosToken)
                        d_50_next_ = out36_
                        d_51_wasConstrained_ = out37_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_50_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            if (d_50_next_) == (d_7_lastToken_):
                                d_8_repeatCount_ = (d_8_repeatCount_) + (1)
                            elif True:
                                d_7_lastToken_ = d_50_next_
                                d_8_repeatCount_ = 1
                            d_52_g2_: _dafny.Seq
                            d_53_i2_: bool
                            d_54_c2_: _dafny.Seq
                            out38_: _dafny.Seq
                            out39_: bool
                            out40_: _dafny.Seq
                            out38_, out39_, out40_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_50_next_)
                            d_52_g2_ = out38_
                            d_53_i2_ = out39_
                            d_54_c2_ = out40_
                            generated = d_52_g2_
                            insideConstrainedOut = d_53_i2_
                            currentConstrainedOut = d_54_c2_
                            d_5_spanTokensUsed_ = (d_5_spanTokensUsed_) + (1)
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

