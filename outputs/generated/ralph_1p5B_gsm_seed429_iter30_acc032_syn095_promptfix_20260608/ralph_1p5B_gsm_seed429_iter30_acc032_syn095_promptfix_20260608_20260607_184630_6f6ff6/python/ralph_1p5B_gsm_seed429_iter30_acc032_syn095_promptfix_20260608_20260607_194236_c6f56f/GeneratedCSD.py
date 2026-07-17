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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step. Put ONLY arithmetic expressions inside << >> delimiters. Do not put words or text inside << >>. Each << >> span must contain a valid arithmetic expression like <<3 + 4>> or <<n * k>>. The final answer must be inside << >>. Never put text, words, or template variables inside << >>.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_freeChunkSize_: int
        d_2_freeChunkSize_ = 5
        d_3_freeTokensSinceSpan_: int
        d_3_freeTokensSinceSpan_ = 0
        d_4_spanTokenCount_: int
        d_4_spanTokenCount_ = 0
        d_5_maxSpanTokens_: int
        d_5_maxSpanTokens_ = 12
        d_6_forceOpenThreshold_: int
        d_6_forceOpenThreshold_ = 15
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if ((d_3_freeTokensSinceSpan_) >= (d_6_forceOpenThreshold_)) and (((d_1_steps_) + (2)) <= (maxSteps)):
                            d_7_openGenerated_: _dafny.Seq
                            d_8_openInside_: bool
                            d_9_openCurrent_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_7_openGenerated_ = out0_
                            d_8_openInside_ = out1_
                            d_9_openCurrent_ = out2_
                            generated = d_7_openGenerated_
                            insideConstrainedOut = d_8_openInside_
                            currentConstrainedOut = d_9_openCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_3_freeTokensSinceSpan_ = 0
                            d_4_spanTokenCount_ = 0
                        elif True:
                            d_10_remainingSteps_: int
                            d_10_remainingSteps_ = (maxSteps) - (d_1_steps_)
                            d_11_chunkSize_: int
                            d_11_chunkSize_ = d_2_freeChunkSize_
                            if (d_11_chunkSize_) > (d_10_remainingSteps_):
                                d_11_chunkSize_ = d_10_remainingSteps_
                            if (d_11_chunkSize_) == (0):
                                raise _dafny.Break("0")
                            d_12_chunkGenerated_: _dafny.Seq
                            d_13_stoppedOnOpenSpan_: bool
                            d_14_stoppedOnEos_: bool
                            d_15_stepsUsed_: int
                            out3_: _dafny.Seq
                            out4_: bool
                            out5_: bool
                            out6_: int
                            out3_, out4_, out5_, out6_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_11_chunkSize_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_12_chunkGenerated_ = out3_
                            d_13_stoppedOnOpenSpan_ = out4_
                            d_14_stoppedOnEos_ = out5_
                            d_15_stepsUsed_ = out6_
                            generated = d_12_chunkGenerated_
                            d_1_steps_ = (d_1_steps_) + (d_15_stepsUsed_)
                            d_3_freeTokensSinceSpan_ = (d_3_freeTokensSinceSpan_) + (d_15_stepsUsed_)
                            if d_14_stoppedOnEos_:
                                raise _dafny.Break("0")
                            if d_13_stoppedOnOpenSpan_:
                                d_16_enterGenerated_: _dafny.Seq
                                d_17_enterInside_: bool
                                d_18_enterCurrent_: _dafny.Seq
                                out7_: _dafny.Seq
                                out8_: bool
                                out9_: _dafny.Seq
                                out7_, out8_, out9_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_16_enterGenerated_ = out7_
                                d_17_enterInside_ = out8_
                                d_18_enterCurrent_ = out9_
                                generated = d_16_enterGenerated_
                                insideConstrainedOut = d_17_enterInside_
                                currentConstrainedOut = d_18_enterCurrent_
                                d_3_freeTokensSinceSpan_ = 0
                                d_4_spanTokenCount_ = 0
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_19_closedGenerated_: _dafny.Seq
                        d_20_closedInside_: bool
                        d_21_closedCurrent_: _dafny.Seq
                        out10_: _dafny.Seq
                        out11_: bool
                        out12_: _dafny.Seq
                        out10_, out11_, out12_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_19_closedGenerated_ = out10_
                        d_20_closedInside_ = out11_
                        d_21_closedCurrent_ = out12_
                        generated = d_19_closedGenerated_
                        insideConstrainedOut = d_20_closedInside_
                        currentConstrainedOut = d_21_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_3_freeTokensSinceSpan_ = 0
                        d_4_spanTokenCount_ = 0
                    elif (d_4_spanTokenCount_) >= (d_5_maxSpanTokens_):
                        d_22_rolledGenerated_: _dafny.Seq
                        d_23_rolledCurrent_: _dafny.Seq
                        out13_: _dafny.Seq
                        out14_: _dafny.Seq
                        out13_, out14_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                        d_22_rolledGenerated_ = out13_
                        d_23_rolledCurrent_ = out14_
                        generated = d_22_rolledGenerated_
                        currentConstrainedOut = d_23_rolledCurrent_
                        d_4_spanTokenCount_ = len(currentConstrainedOut)
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            d_24_closedGenerated_: _dafny.Seq
                            d_25_closedInside_: bool
                            d_26_closedCurrent_: _dafny.Seq
                            out15_: _dafny.Seq
                            out16_: bool
                            out17_: _dafny.Seq
                            out15_, out16_, out17_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_24_closedGenerated_ = out15_
                            d_25_closedInside_ = out16_
                            d_26_closedCurrent_ = out17_
                            generated = d_24_closedGenerated_
                            insideConstrainedOut = d_25_closedInside_
                            currentConstrainedOut = d_26_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_3_freeTokensSinceSpan_ = 0
                            d_4_spanTokenCount_ = 0
                        elif True:
                            if (d_1_steps_) < (maxSteps):
                                d_27_constrainedPrompt_: _dafny.Seq
                                d_27_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_28_next_: _dafny.Seq
                                d_29_wasConstrained_: bool
                                out18_: _dafny.Seq
                                out19_: bool
                                out18_, out19_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, d_27_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_28_next_ = out18_
                                d_29_wasConstrained_ = out19_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_28_next_) == (eosToken):
                                    d_30_rolledGenerated2_: _dafny.Seq
                                    d_31_rolledCurrent2_: _dafny.Seq
                                    out20_: _dafny.Seq
                                    out21_: _dafny.Seq
                                    out20_, out21_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                    d_30_rolledGenerated2_ = out20_
                                    d_31_rolledCurrent2_ = out21_
                                    generated = d_30_rolledGenerated2_
                                    currentConstrainedOut = d_31_rolledCurrent2_
                                    if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                        d_32_closedGenerated2_: _dafny.Seq
                                        d_33_closedInside2_: bool
                                        d_34_closedCurrent2_: _dafny.Seq
                                        out22_: _dafny.Seq
                                        out23_: bool
                                        out24_: _dafny.Seq
                                        out22_, out23_, out24_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                        d_32_closedGenerated2_ = out22_
                                        d_33_closedInside2_ = out23_
                                        d_34_closedCurrent2_ = out24_
                                        generated = d_32_closedGenerated2_
                                        insideConstrainedOut = d_33_closedInside2_
                                        currentConstrainedOut = d_34_closedCurrent2_
                                        d_1_steps_ = (d_1_steps_) + (1)
                                        d_3_freeTokensSinceSpan_ = 0
                                        d_4_spanTokenCount_ = 0
                                    elif True:
                                        raise _dafny.Break("0")
                                elif True:
                                    d_35_appendedGenerated_: _dafny.Seq
                                    d_36_appendedInside_: bool
                                    d_37_appendedCurrent_: _dafny.Seq
                                    out25_: _dafny.Seq
                                    out26_: bool
                                    out27_: _dafny.Seq
                                    out25_, out26_, out27_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_28_next_)
                                    d_35_appendedGenerated_ = out25_
                                    d_36_appendedInside_ = out26_
                                    d_37_appendedCurrent_ = out27_
                                    generated = d_35_appendedGenerated_
                                    insideConstrainedOut = d_36_appendedInside_
                                    currentConstrainedOut = d_37_appendedCurrent_
                                    d_4_spanTokenCount_ = (d_4_spanTokenCount_) + (1)
                    elif True:
                        d_38_isDeadEnd_: bool
                        out28_: bool
                        out28_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                        d_38_isDeadEnd_ = out28_
                        if d_38_isDeadEnd_:
                            d_39_rolledGenerated_: _dafny.Seq
                            d_40_rolledCurrent_: _dafny.Seq
                            out29_: _dafny.Seq
                            out30_: _dafny.Seq
                            out29_, out30_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                            d_39_rolledGenerated_ = out29_
                            d_40_rolledCurrent_ = out30_
                            generated = d_39_rolledGenerated_
                            currentConstrainedOut = d_40_rolledCurrent_
                            d_4_spanTokenCount_ = len(currentConstrainedOut)
                            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                d_41_closedGenerated_: _dafny.Seq
                                d_42_closedInside_: bool
                                d_43_closedCurrent_: _dafny.Seq
                                out31_: _dafny.Seq
                                out32_: bool
                                out33_: _dafny.Seq
                                out31_, out32_, out33_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_41_closedGenerated_ = out31_
                                d_42_closedInside_ = out32_
                                d_43_closedCurrent_ = out33_
                                generated = d_41_closedGenerated_
                                insideConstrainedOut = d_42_closedInside_
                                currentConstrainedOut = d_43_closedCurrent_
                                d_1_steps_ = (d_1_steps_) + (1)
                                d_3_freeTokensSinceSpan_ = 0
                                d_4_spanTokenCount_ = 0
                        elif True:
                            d_44_constrainedPrompt_: _dafny.Seq
                            d_44_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_45_next_: _dafny.Seq
                            d_46_wasConstrained_: bool
                            out34_: _dafny.Seq
                            out35_: bool
                            out34_, out35_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, d_44_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_45_next_ = out34_
                            d_46_wasConstrained_ = out35_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_45_next_) == (eosToken):
                                d_47_rolledGenerated_: _dafny.Seq
                                d_48_rolledCurrent_: _dafny.Seq
                                out36_: _dafny.Seq
                                out37_: _dafny.Seq
                                out36_, out37_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                d_47_rolledGenerated_ = out36_
                                d_48_rolledCurrent_ = out37_
                                generated = d_47_rolledGenerated_
                                currentConstrainedOut = d_48_rolledCurrent_
                                if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                    d_49_closedGenerated_: _dafny.Seq
                                    d_50_closedInside_: bool
                                    d_51_closedCurrent_: _dafny.Seq
                                    out38_: _dafny.Seq
                                    out39_: bool
                                    out40_: _dafny.Seq
                                    out38_, out39_, out40_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_49_closedGenerated_ = out38_
                                    d_50_closedInside_ = out39_
                                    d_51_closedCurrent_ = out40_
                                    generated = d_49_closedGenerated_
                                    insideConstrainedOut = d_50_closedInside_
                                    currentConstrainedOut = d_51_closedCurrent_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    d_3_freeTokensSinceSpan_ = 0
                                    d_4_spanTokenCount_ = 0
                                elif True:
                                    raise _dafny.Break("0")
                            elif True:
                                d_52_appendedGenerated_: _dafny.Seq
                                d_53_appendedInside_: bool
                                d_54_appendedCurrent_: _dafny.Seq
                                out41_: _dafny.Seq
                                out42_: bool
                                out43_: _dafny.Seq
                                out41_, out42_, out43_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_45_next_)
                                d_52_appendedGenerated_ = out41_
                                d_53_appendedInside_ = out42_
                                d_54_appendedCurrent_ = out43_
                                generated = d_52_appendedGenerated_
                                insideConstrainedOut = d_53_appendedInside_
                                currentConstrainedOut = d_54_appendedCurrent_
                                d_4_spanTokenCount_ = (d_4_spanTokenCount_) + (1)
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

