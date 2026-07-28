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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. Place each arithmetic expression and the final answer inside << >> delimiters. Use only valid arithmetic tokens inside << >>. Final answer must be in a << >> span.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_spanTokens_: int
        d_2_spanTokens_ = 0
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_3_remainingSteps_: int
                        d_3_remainingSteps_ = (maxSteps) - (d_1_steps_)
                        d_4_chunkSize_: int
                        d_4_chunkSize_ = 30
                        if (d_4_chunkSize_) > (d_3_remainingSteps_):
                            d_4_chunkSize_ = d_3_remainingSteps_
                        if (d_4_chunkSize_) == (0):
                            raise _dafny.Break("0")
                        d_5_chunkGenerated_: _dafny.Seq
                        d_6_stoppedOnOpenSpan_: bool
                        d_7_stoppedOnEos_: bool
                        d_8_stepsUsed_: int
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: bool
                        out3_: int
                        out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_4_chunkSize_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_5_chunkGenerated_ = out0_
                        d_6_stoppedOnOpenSpan_ = out1_
                        d_7_stoppedOnEos_ = out2_
                        d_8_stepsUsed_ = out3_
                        generated = d_5_chunkGenerated_
                        d_1_steps_ = (d_1_steps_) + (d_8_stepsUsed_)
                        if d_7_stoppedOnEos_:
                            raise _dafny.Break("0")
                        if d_6_stoppedOnOpenSpan_:
                            d_9_enterGenerated_: _dafny.Seq
                            d_10_enterInside_: bool
                            d_11_enterCurrent_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                            d_9_enterGenerated_ = out4_
                            d_10_enterInside_ = out5_
                            d_11_enterCurrent_ = out6_
                            generated = d_9_enterGenerated_
                            insideConstrainedOut = d_10_enterInside_
                            currentConstrainedOut = d_11_enterCurrent_
                            d_2_spanTokens_ = 0
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_12_closedGenerated_: _dafny.Seq
                        d_13_closedInside_: bool
                        d_14_closedCurrent_: _dafny.Seq
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: _dafny.Seq
                        out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_12_closedGenerated_ = out7_
                        d_13_closedInside_ = out8_
                        d_14_closedCurrent_ = out9_
                        generated = d_12_closedGenerated_
                        insideConstrainedOut = d_13_closedInside_
                        currentConstrainedOut = d_14_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_2_spanTokens_ = 0
                    elif True:
                        d_15_isDeadEnd_: bool
                        out10_: bool
                        out10_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                        d_15_isDeadEnd_ = out10_
                        if d_15_isDeadEnd_:
                            d_16_rolledGenerated_: _dafny.Seq
                            d_17_rolledCurrent_: _dafny.Seq
                            out11_: _dafny.Seq
                            out12_: _dafny.Seq
                            out11_, out12_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                            d_16_rolledGenerated_ = out11_
                            d_17_rolledCurrent_ = out12_
                            generated = d_16_rolledGenerated_
                            currentConstrainedOut = d_17_rolledCurrent_
                            d_18_constrainedPrompt_: _dafny.Seq
                            d_18_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_19_next_: _dafny.Seq
                            out13_: _dafny.Seq
                            out13_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_18_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_19_next_ = out13_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_19_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_20_appendedGenerated_: _dafny.Seq
                                d_21_appendedInside_: bool
                                d_22_appendedCurrent_: _dafny.Seq
                                out14_: _dafny.Seq
                                out15_: bool
                                out16_: _dafny.Seq
                                out14_, out15_, out16_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_19_next_)
                                d_20_appendedGenerated_ = out14_
                                d_21_appendedInside_ = out15_
                                d_22_appendedCurrent_ = out16_
                                generated = d_20_appendedGenerated_
                                insideConstrainedOut = d_21_appendedInside_
                                currentConstrainedOut = d_22_appendedCurrent_
                                d_2_spanTokens_ = (d_2_spanTokens_) + (1)
                        elif True:
                            d_23_constrainedPrompt_: _dafny.Seq
                            d_23_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_24_next_: _dafny.Seq
                            out17_: _dafny.Seq
                            out17_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_23_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_24_next_ = out17_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_24_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_25_appendedGenerated_: _dafny.Seq
                                d_26_appendedInside_: bool
                                d_27_appendedCurrent_: _dafny.Seq
                                out18_: _dafny.Seq
                                out19_: bool
                                out20_: _dafny.Seq
                                out18_, out19_, out20_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_24_next_)
                                d_25_appendedGenerated_ = out18_
                                d_26_appendedInside_ = out19_
                                d_27_appendedCurrent_ = out20_
                                generated = d_25_appendedGenerated_
                                insideConstrainedOut = d_26_appendedInside_
                                currentConstrainedOut = d_27_appendedCurrent_
                                d_2_spanTokens_ = (d_2_spanTokens_) + (1)
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

