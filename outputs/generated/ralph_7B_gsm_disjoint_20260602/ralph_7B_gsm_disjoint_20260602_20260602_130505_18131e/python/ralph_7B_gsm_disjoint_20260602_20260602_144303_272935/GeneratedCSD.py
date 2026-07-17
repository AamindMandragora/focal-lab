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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math problem step by step. For EVERY arithmetic expression and the final answer, you MUST use << >> delimiters. Example: <<3+4=7>>. End with #### <<final_answer>>. Every << must be closed with >>.")))
        d_1_penaltyTokens_: _dafny.Seq
        d_1_penaltyTokens_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "!")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "?")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "#")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "@")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "$")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "%")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "^")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "&")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "*")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "~")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "`"))])
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_effectiveCap_: int
        d_3_effectiveCap_ = 220
        d_4_unconstrainedTokens_: int
        d_4_unconstrainedTokens_ = 0
        with _dafny.label("0"):
            while ((d_2_steps_) < (maxSteps)) and ((d_2_steps_) < (d_3_effectiveCap_)):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if (d_4_unconstrainedTokens_) >= (30):
                            d_5_remaining_: int
                            if ((maxSteps) - (d_2_steps_)) < ((d_3_effectiveCap_) - (d_2_steps_)):
                                d_5_remaining_ = (maxSteps) - (d_2_steps_)
                            elif True:
                                d_5_remaining_ = (d_3_effectiveCap_) - (d_2_steps_)
                            if (d_5_remaining_) == (0):
                                raise _dafny.Break("0")
                            d_6_g2_: _dafny.Seq
                            d_7_ins2_: bool
                            d_8_cur2_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_6_g2_ = out0_
                            d_7_ins2_ = out1_
                            d_8_cur2_ = out2_
                            generated = d_6_g2_
                            insideConstrainedOut = d_7_ins2_
                            currentConstrainedOut = d_8_cur2_
                            d_2_steps_ = (d_2_steps_) + (1)
                            d_4_unconstrainedTokens_ = 0
                        elif True:
                            d_9_remaining_: int
                            if ((maxSteps) - (d_2_steps_)) < ((d_3_effectiveCap_) - (d_2_steps_)):
                                d_9_remaining_ = (maxSteps) - (d_2_steps_)
                            elif True:
                                d_9_remaining_ = (d_3_effectiveCap_) - (d_2_steps_)
                            d_10_chunkBudget_: int
                            if (d_9_remaining_) < (25):
                                d_10_chunkBudget_ = d_9_remaining_
                            elif True:
                                d_10_chunkBudget_ = 25
                            if (d_10_chunkBudget_) == (0):
                                raise _dafny.Break("0")
                            d_11_generatedOut_: _dafny.Seq
                            d_12_stoppedOnOpenSpan_: bool
                            d_13_stoppedOnEos_: bool
                            d_14_stepsUsed_: int
                            out3_: _dafny.Seq
                            out4_: bool
                            out5_: bool
                            out6_: int
                            out3_, out4_, out5_, out6_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_10_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_11_generatedOut_ = out3_
                            d_12_stoppedOnOpenSpan_ = out4_
                            d_13_stoppedOnEos_ = out5_
                            d_14_stepsUsed_ = out6_
                            generated = d_11_generatedOut_
                            d_2_steps_ = (d_2_steps_) + (d_14_stepsUsed_)
                            d_4_unconstrainedTokens_ = (d_4_unconstrainedTokens_) + (d_14_stepsUsed_)
                            if d_13_stoppedOnEos_:
                                raise _dafny.Break("0")
                            elif d_12_stoppedOnOpenSpan_:
                                d_15_g2_: _dafny.Seq
                                d_16_ins2_: bool
                                d_17_cur2_: _dafny.Seq
                                out7_: _dafny.Seq
                                out8_: bool
                                out9_: _dafny.Seq
                                out7_, out8_, out9_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_15_g2_ = out7_
                                d_16_ins2_ = out8_
                                d_17_cur2_ = out9_
                                generated = d_15_g2_
                                insideConstrainedOut = d_16_ins2_
                                currentConstrainedOut = d_17_cur2_
                                d_4_unconstrainedTokens_ = 0
                            elif True:
                                if ((len(generated)) > (0)) and (((generated)[(len(generated)) - (1)]) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))):
                                    d_18_g2_: _dafny.Seq
                                    d_19_ins2_: bool
                                    d_20_cur2_: _dafny.Seq
                                    out10_: _dafny.Seq
                                    out11_: bool
                                    out12_: _dafny.Seq
                                    out10_, out11_, out12_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                    d_18_g2_ = out10_
                                    d_19_ins2_ = out11_
                                    d_20_cur2_ = out12_
                                    generated = d_18_g2_
                                    insideConstrainedOut = d_19_ins2_
                                    currentConstrainedOut = d_20_cur2_
                                    d_4_unconstrainedTokens_ = 0
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_21_remaining_: int
                        if ((maxSteps) - (d_2_steps_)) < ((d_3_effectiveCap_) - (d_2_steps_)):
                            d_21_remaining_ = (maxSteps) - (d_2_steps_)
                        elif True:
                            d_21_remaining_ = (d_3_effectiveCap_) - (d_2_steps_)
                        if (d_21_remaining_) == (0):
                            raise _dafny.Break("0")
                        d_22_closedGenerated_: _dafny.Seq
                        d_23_closedInside_: bool
                        d_24_closedCurrent_: _dafny.Seq
                        out13_: _dafny.Seq
                        out14_: bool
                        out15_: _dafny.Seq
                        out13_, out14_, out15_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_22_closedGenerated_ = out13_
                        d_23_closedInside_ = out14_
                        d_24_closedCurrent_ = out15_
                        generated = d_22_closedGenerated_
                        insideConstrainedOut = d_23_closedInside_
                        currentConstrainedOut = d_24_closedCurrent_
                        d_2_steps_ = (d_2_steps_) + (1)
                    elif True:
                        d_25_remaining_: int
                        if ((maxSteps) - (d_2_steps_)) < ((d_3_effectiveCap_) - (d_2_steps_)):
                            d_25_remaining_ = (maxSteps) - (d_2_steps_)
                        elif True:
                            d_25_remaining_ = (d_3_effectiveCap_) - (d_2_steps_)
                        if (d_25_remaining_) == (0):
                            raise _dafny.Break("0")
                        d_26_constrainedPrompt_: _dafny.Seq
                        d_26_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_27_nextPen_: _dafny.Seq
                        out16_: _dafny.Seq
                        out16_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_26_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('8e0'), d_1_penaltyTokens_, _dafny.BigRational('6e0'), 12, eosToken)
                        d_27_nextPen_ = out16_
                        d_2_steps_ = (d_2_steps_) + (1)
                        if (d_27_nextPen_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_28_appendedGenerated_: _dafny.Seq
                            d_29_appendedInside_: bool
                            d_30_appendedCurrent_: _dafny.Seq
                            out17_: _dafny.Seq
                            out18_: bool
                            out19_: _dafny.Seq
                            out17_, out18_, out19_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_27_nextPen_)
                            d_28_appendedGenerated_ = out17_
                            d_29_appendedInside_ = out18_
                            d_30_appendedCurrent_ = out19_
                            generated = d_28_appendedGenerated_
                            insideConstrainedOut = d_29_appendedInside_
                            currentConstrainedOut = d_30_appendedCurrent_
                    pass
            pass
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

