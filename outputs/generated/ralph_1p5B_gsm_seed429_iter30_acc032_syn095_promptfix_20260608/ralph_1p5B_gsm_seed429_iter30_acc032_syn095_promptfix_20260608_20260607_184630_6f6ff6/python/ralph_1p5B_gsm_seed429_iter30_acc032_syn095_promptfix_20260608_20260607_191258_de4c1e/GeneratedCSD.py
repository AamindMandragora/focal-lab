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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. Use << >> for arithmetic only. Each <<expr>> must be a valid arithmetic expression with numbers and +,-,*,/,(). Example: <<3 * 4 + 2>>. Final answer: <<number>>.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_freeChunkSize_: int
        d_2_freeChunkSize_ = 6
        d_3_tokensSinceLastSpan_: int
        d_3_tokensSinceLastSpan_ = 0
        d_4_spanTokenCount_: int
        d_4_spanTokenCount_ = 0
        d_5_maxSpanTokens_: int
        d_5_maxSpanTokens_ = 25
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_6_remainingSteps_: int
                        d_6_remainingSteps_ = (maxSteps) - (d_1_steps_)
                        d_7_chunkSize_: int
                        d_7_chunkSize_ = d_2_freeChunkSize_
                        if (d_7_chunkSize_) > (d_6_remainingSteps_):
                            d_7_chunkSize_ = d_6_remainingSteps_
                        if (d_7_chunkSize_) == (0):
                            raise _dafny.Break("0")
                        d_8_chunkGenerated_: _dafny.Seq
                        d_9_stoppedOnOpenSpan_: bool
                        d_10_stoppedOnEos_: bool
                        d_11_stepsUsed_: int
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: bool
                        out3_: int
                        out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_7_chunkSize_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_8_chunkGenerated_ = out0_
                        d_9_stoppedOnOpenSpan_ = out1_
                        d_10_stoppedOnEos_ = out2_
                        d_11_stepsUsed_ = out3_
                        generated = d_8_chunkGenerated_
                        d_1_steps_ = (d_1_steps_) + (d_11_stepsUsed_)
                        d_3_tokensSinceLastSpan_ = (d_3_tokensSinceLastSpan_) + (d_11_stepsUsed_)
                        if d_10_stoppedOnEos_:
                            raise _dafny.Break("0")
                        if d_9_stoppedOnOpenSpan_:
                            d_12_enterGenerated_: _dafny.Seq
                            d_13_enterInside_: bool
                            d_14_enterCurrent_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                            d_12_enterGenerated_ = out4_
                            d_13_enterInside_ = out5_
                            d_14_enterCurrent_ = out6_
                            generated = d_12_enterGenerated_
                            insideConstrainedOut = d_13_enterInside_
                            currentConstrainedOut = d_14_enterCurrent_
                            d_3_tokensSinceLastSpan_ = 0
                            d_4_spanTokenCount_ = 0
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_15_closedGenerated_: _dafny.Seq
                        d_16_closedInside_: bool
                        d_17_closedCurrent_: _dafny.Seq
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: _dafny.Seq
                        out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_15_closedGenerated_ = out7_
                        d_16_closedInside_ = out8_
                        d_17_closedCurrent_ = out9_
                        generated = d_15_closedGenerated_
                        insideConstrainedOut = d_16_closedInside_
                        currentConstrainedOut = d_17_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_3_tokensSinceLastSpan_ = 0
                        d_4_spanTokenCount_ = 0
                    elif (d_4_spanTokenCount_) >= (d_5_maxSpanTokens_):
                        d_18_rolledGenerated_: _dafny.Seq
                        d_19_rolledCurrent_: _dafny.Seq
                        out10_: _dafny.Seq
                        out11_: _dafny.Seq
                        out10_, out11_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                        d_18_rolledGenerated_ = out10_
                        d_19_rolledCurrent_ = out11_
                        generated = d_18_rolledGenerated_
                        currentConstrainedOut = d_19_rolledCurrent_
                        d_4_spanTokenCount_ = len(currentConstrainedOut)
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            d_20_closedGenerated_: _dafny.Seq
                            d_21_closedInside_: bool
                            d_22_closedCurrent_: _dafny.Seq
                            out12_: _dafny.Seq
                            out13_: bool
                            out14_: _dafny.Seq
                            out12_, out13_, out14_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_20_closedGenerated_ = out12_
                            d_21_closedInside_ = out13_
                            d_22_closedCurrent_ = out14_
                            generated = d_20_closedGenerated_
                            insideConstrainedOut = d_21_closedInside_
                            currentConstrainedOut = d_22_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_3_tokensSinceLastSpan_ = 0
                            d_4_spanTokenCount_ = 0
                        elif True:
                            d_23_constrainedPrompt_: _dafny.Seq
                            d_23_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_24_next_: _dafny.Seq
                            out15_: _dafny.Seq
                            out15_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_23_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_24_next_ = out15_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_24_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_25_appendedGenerated_: _dafny.Seq
                                d_26_appendedInside_: bool
                                d_27_appendedCurrent_: _dafny.Seq
                                out16_: _dafny.Seq
                                out17_: bool
                                out18_: _dafny.Seq
                                out16_, out17_, out18_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_24_next_)
                                d_25_appendedGenerated_ = out16_
                                d_26_appendedInside_ = out17_
                                d_27_appendedCurrent_ = out18_
                                generated = d_25_appendedGenerated_
                                insideConstrainedOut = d_26_appendedInside_
                                currentConstrainedOut = d_27_appendedCurrent_
                                d_4_spanTokenCount_ = (d_4_spanTokenCount_) + (1)
                    elif True:
                        d_28_constrainedPrompt_: _dafny.Seq
                        d_28_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_29_next_: _dafny.Seq
                        out19_: _dafny.Seq
                        out19_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_28_constrainedPrompt_, currentConstrainedOut, eosToken)
                        d_29_next_ = out19_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_29_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_30_appendedGenerated_: _dafny.Seq
                            d_31_appendedInside_: bool
                            d_32_appendedCurrent_: _dafny.Seq
                            out20_: _dafny.Seq
                            out21_: bool
                            out22_: _dafny.Seq
                            out20_, out21_, out22_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_29_next_)
                            d_30_appendedGenerated_ = out20_
                            d_31_appendedInside_ = out21_
                            d_32_appendedCurrent_ = out22_
                            generated = d_30_appendedGenerated_
                            insideConstrainedOut = d_31_appendedInside_
                            currentConstrainedOut = d_32_appendedCurrent_
                            d_4_spanTokenCount_ = (d_4_spanTokenCount_) + (1)
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

