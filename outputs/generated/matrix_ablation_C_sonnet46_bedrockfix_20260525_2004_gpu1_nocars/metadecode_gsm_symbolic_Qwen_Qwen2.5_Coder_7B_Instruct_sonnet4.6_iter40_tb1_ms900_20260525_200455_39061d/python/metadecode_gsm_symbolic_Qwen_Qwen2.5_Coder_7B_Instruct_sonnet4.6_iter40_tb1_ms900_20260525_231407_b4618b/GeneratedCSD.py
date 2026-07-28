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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem concisely. Write your answer as a single Python arithmetic expression inside << >> delimiters at the end. Use the variable names from the problem. The answer must be just the expression, no equals sign, no units. Example: The answer is <<n * k + d>>.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_maxConstrainedLen_: int
        d_2_maxConstrainedLen_ = 40
        d_3_maxUnconstrainedTotal_: int
        d_3_maxUnconstrainedTotal_ = 160
        d_4_unconstrainedUsed_: int
        d_4_unconstrainedUsed_ = 0
        d_5_effectiveBudget_: int
        if (maxSteps) > (200):
            d_5_effectiveBudget_ = 200
        elif True:
            d_5_effectiveBudget_ = maxSteps
        with _dafny.label("0"):
            while (d_1_steps_) < (d_5_effectiveBudget_):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if (d_4_unconstrainedUsed_) >= (d_3_maxUnconstrainedTotal_):
                            d_6_openedGenerated_: _dafny.Seq
                            d_7_openedInside_: bool
                            d_8_openedCurrent_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_6_openedGenerated_ = out0_
                            d_7_openedInside_ = out1_
                            d_8_openedCurrent_ = out2_
                            generated = d_6_openedGenerated_
                            insideConstrainedOut = d_7_openedInside_
                            currentConstrainedOut = d_8_openedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_9_remaining_: int
                            d_9_remaining_ = (d_5_effectiveBudget_) - (d_1_steps_)
                            d_10_unconstrainedRemaining_: int
                            d_10_unconstrainedRemaining_ = (d_3_maxUnconstrainedTotal_) - (d_4_unconstrainedUsed_)
                            d_11_chunkCap_: int
                            if (d_10_unconstrainedRemaining_) < (50):
                                d_11_chunkCap_ = d_10_unconstrainedRemaining_
                            elif True:
                                d_11_chunkCap_ = 50
                            d_12_chunkBudget_: int
                            if (d_9_remaining_) < (d_11_chunkCap_):
                                d_12_chunkBudget_ = d_9_remaining_
                            elif True:
                                d_12_chunkBudget_ = d_11_chunkCap_
                            if (d_12_chunkBudget_) == (0):
                                raise _dafny.Break("0")
                            d_13_chunkedG_: _dafny.Seq
                            d_14_stoppedOpen_: bool
                            d_15_stoppedEos_: bool
                            d_16_stepsUsed_: int
                            out3_: _dafny.Seq
                            out4_: bool
                            out5_: bool
                            out6_: int
                            out3_, out4_, out5_, out6_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_12_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_13_chunkedG_ = out3_
                            d_14_stoppedOpen_ = out4_
                            d_15_stoppedEos_ = out5_
                            d_16_stepsUsed_ = out6_
                            generated = d_13_chunkedG_
                            d_1_steps_ = (d_1_steps_) + (d_16_stepsUsed_)
                            d_4_unconstrainedUsed_ = (d_4_unconstrainedUsed_) + (d_16_stepsUsed_)
                            if d_14_stoppedOpen_:
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                            elif d_15_stoppedEos_:
                                raise _dafny.Break("0")
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_17_closedGenerated_: _dafny.Seq
                        d_18_closedInside_: bool
                        d_19_closedCurrent_: _dafny.Seq
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: _dafny.Seq
                        out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_17_closedGenerated_ = out7_
                        d_18_closedInside_ = out8_
                        d_19_closedCurrent_ = out9_
                        generated = d_17_closedGenerated_
                        insideConstrainedOut = d_18_closedInside_
                        currentConstrainedOut = d_19_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                        raise _dafny.Break("0")
                    elif (len(currentConstrainedOut)) >= (d_2_maxConstrainedLen_):
                        insideConstrainedOut = False
                        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_4_unconstrainedUsed_) > (20):
                            d_4_unconstrainedUsed_ = (d_4_unconstrainedUsed_) - (20)
                        elif True:
                            d_4_unconstrainedUsed_ = 0
                    elif True:
                        d_20_constrainedPrompt_: _dafny.Seq
                        d_20_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_21_next_: _dafny.Seq
                        out10_: _dafny.Seq
                        out10_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_20_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('4e0'), eosToken)
                        d_21_next_ = out10_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_21_next_) == (eosToken):
                            insideConstrainedOut = False
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                        elif True:
                            d_22_appendedGenerated_: _dafny.Seq
                            d_23_appendedInside_: bool
                            d_24_appendedCurrent_: _dafny.Seq
                            out11_: _dafny.Seq
                            out12_: bool
                            out13_: _dafny.Seq
                            out11_, out12_, out13_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_21_next_)
                            d_22_appendedGenerated_ = out11_
                            d_23_appendedInside_ = out12_
                            d_24_appendedCurrent_ = out13_
                            generated = d_22_appendedGenerated_
                            insideConstrainedOut = d_23_appendedInside_
                            currentConstrainedOut = d_24_appendedCurrent_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            if (parser).IsCompletePrefix(currentConstrainedOut):
                d_25_closedGenerated_: _dafny.Seq
                d_26_closedInside_: bool
                d_27_closedCurrent_: _dafny.Seq
                out14_: _dafny.Seq
                out15_: bool
                out16_: _dafny.Seq
                out14_, out15_, out16_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_25_closedGenerated_ = out14_
                d_26_closedInside_ = out15_
                d_27_closedCurrent_ = out16_
                generated = d_25_closedGenerated_
                insideConstrainedOut = d_26_closedInside_
                currentConstrainedOut = d_27_closedCurrent_
                d_1_steps_ = (d_1_steps_) + (1)
            elif True:
                insideConstrainedOut = False
                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

