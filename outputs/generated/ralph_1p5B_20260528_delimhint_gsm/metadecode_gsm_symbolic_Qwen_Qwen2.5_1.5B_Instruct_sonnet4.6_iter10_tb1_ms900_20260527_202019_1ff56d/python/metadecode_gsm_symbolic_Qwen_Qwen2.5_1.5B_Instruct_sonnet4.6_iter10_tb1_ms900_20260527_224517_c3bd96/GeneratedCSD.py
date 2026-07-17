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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve this math word problem step by step, computing actual numeric values. IMPORTANT: Inside << >> delimiters, write ONLY arithmetic with actual numbers and the operators +, -, *, /, (, ), =. Example: <<3 * 5 = 15>>. Final answer: <<42>>. NEVER write variable names like {n}, template placeholders, or currency symbols like $ inside << >>. Every << >> span must contain only digits, spaces, and the operators listed above.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_spanSteps_: int
        d_2_spanSteps_ = 0
        d_3_maxSpanSteps_: int
        d_3_maxSpanSteps_ = 28
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_4_remaining_: int
                        d_4_remaining_ = (maxSteps) - (d_1_steps_)
                        d_5_chunkBudget_: int
                        if (d_4_remaining_) < (35):
                            d_5_chunkBudget_ = d_4_remaining_
                        elif True:
                            d_5_chunkBudget_ = 35
                        d_6_g_: _dafny.Seq
                        d_7_stoppedOnOpen_: bool
                        d_8_stoppedOnEos_: bool
                        d_9_stepsUsed_: int
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: bool
                        out3_: int
                        out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_5_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_6_g_ = out0_
                        d_7_stoppedOnOpen_ = out1_
                        d_8_stoppedOnEos_ = out2_
                        d_9_stepsUsed_ = out3_
                        generated = d_6_g_
                        d_1_steps_ = (d_1_steps_) + (d_9_stepsUsed_)
                        if d_8_stoppedOnEos_:
                            raise _dafny.Break("0")
                        elif d_7_stoppedOnOpen_:
                            d_10_g2_: _dafny.Seq
                            d_11_i2_: bool
                            d_12_c2_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                            d_10_g2_ = out4_
                            d_11_i2_ = out5_
                            d_12_c2_ = out6_
                            generated = d_10_g2_
                            insideConstrainedOut = d_11_i2_
                            currentConstrainedOut = d_12_c2_
                            d_2_spanSteps_ = 0
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_13_g_: _dafny.Seq
                        d_14_i_: bool
                        d_15_c_: _dafny.Seq
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: _dafny.Seq
                        out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_13_g_ = out7_
                        d_14_i_ = out8_
                        d_15_c_ = out9_
                        generated = d_13_g_
                        insideConstrainedOut = d_14_i_
                        currentConstrainedOut = d_15_c_
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_2_spanSteps_ = 0
                    elif (d_2_spanSteps_) >= (d_3_maxSpanSteps_):
                        d_16_rolledG_: _dafny.Seq
                        d_17_rolledC_: _dafny.Seq
                        out10_: _dafny.Seq
                        out11_: _dafny.Seq
                        out10_, out11_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                        d_16_rolledG_ = out10_
                        d_17_rolledC_ = out11_
                        generated = d_16_rolledG_
                        currentConstrainedOut = d_17_rolledC_
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            d_18_g_: _dafny.Seq
                            d_19_i_: bool
                            d_20_c_: _dafny.Seq
                            out12_: _dafny.Seq
                            out13_: bool
                            out14_: _dafny.Seq
                            out12_, out13_, out14_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_18_g_ = out12_
                            d_19_i_ = out13_
                            d_20_c_ = out14_
                            generated = d_18_g_
                            insideConstrainedOut = d_19_i_
                            currentConstrainedOut = d_20_c_
                        elif True:
                            insideConstrainedOut = False
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_2_spanSteps_ = 0
                    elif True:
                        d_21_constrainedPrompt_: _dafny.Seq
                        d_21_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_22_narrow_: bool
                        out15_: bool
                        out15_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 6)
                        d_22_narrow_ = out15_
                        if d_22_narrow_:
                            d_23_next_: _dafny.Seq
                            out16_: _dafny.Seq
                            out16_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_21_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                            d_23_next_ = out16_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_2_spanSteps_ = (d_2_spanSteps_) + (1)
                            if (d_23_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_24_g_: _dafny.Seq
                                d_25_i_: bool
                                d_26_c_: _dafny.Seq
                                out17_: _dafny.Seq
                                out18_: bool
                                out19_: _dafny.Seq
                                out17_, out18_, out19_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_23_next_)
                                d_24_g_ = out17_
                                d_25_i_ = out18_
                                d_26_c_ = out19_
                                generated = d_24_g_
                                insideConstrainedOut = d_25_i_
                                currentConstrainedOut = d_26_c_
                        elif True:
                            d_27_next_: _dafny.Seq
                            d_28_wasConstrained_: bool
                            out20_: _dafny.Seq
                            out21_: bool
                            out20_, out21_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, d_21_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_27_next_ = out20_
                            d_28_wasConstrained_ = out21_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_2_spanSteps_ = (d_2_spanSteps_) + (1)
                            if (d_27_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_29_g_: _dafny.Seq
                                d_30_i_: bool
                                d_31_c_: _dafny.Seq
                                out22_: _dafny.Seq
                                out23_: bool
                                out24_: _dafny.Seq
                                out22_, out23_, out24_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_27_next_)
                                d_29_g_ = out22_
                                d_30_i_ = out23_
                                d_31_c_ = out24_
                                generated = d_29_g_
                                insideConstrainedOut = d_30_i_
                                currentConstrainedOut = d_31_c_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

