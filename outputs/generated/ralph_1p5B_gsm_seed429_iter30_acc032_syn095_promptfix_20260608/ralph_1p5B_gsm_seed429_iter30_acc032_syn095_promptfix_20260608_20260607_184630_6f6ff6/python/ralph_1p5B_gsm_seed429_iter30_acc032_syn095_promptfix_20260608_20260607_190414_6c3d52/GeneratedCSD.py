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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. Write arithmetic expressions inside << >>. Final answer must be <<number>>. Keep expressions simple: use only numbers and operators + - * / ( ). Example: <<3 * 4 + 2>>. #### answer")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_freeChunkSize_: int
        d_2_freeChunkSize_ = 15
        d_3_spanTokenCount_: int
        d_3_spanTokenCount_ = 0
        d_4_maxSpanTokens_: int
        d_4_maxSpanTokens_ = 40
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_5_remainingSteps_: int
                        d_5_remainingSteps_ = (maxSteps) - (d_1_steps_)
                        d_6_chunkSize_: int
                        d_6_chunkSize_ = d_2_freeChunkSize_
                        if (d_6_chunkSize_) > (d_5_remainingSteps_):
                            d_6_chunkSize_ = d_5_remainingSteps_
                        if (d_6_chunkSize_) == (0):
                            raise _dafny.Break("0")
                        d_7_chunkGenerated_: _dafny.Seq
                        d_8_stoppedOnOpenSpan_: bool
                        d_9_stoppedOnEos_: bool
                        d_10_stepsUsed_: int
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: bool
                        out3_: int
                        out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_6_chunkSize_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_7_chunkGenerated_ = out0_
                        d_8_stoppedOnOpenSpan_ = out1_
                        d_9_stoppedOnEos_ = out2_
                        d_10_stepsUsed_ = out3_
                        generated = d_7_chunkGenerated_
                        d_1_steps_ = (d_1_steps_) + (d_10_stepsUsed_)
                        if d_9_stoppedOnEos_:
                            raise _dafny.Break("0")
                        if d_8_stoppedOnOpenSpan_:
                            d_11_enterGenerated_: _dafny.Seq
                            d_12_enterInside_: bool
                            d_13_enterCurrent_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                            d_11_enterGenerated_ = out4_
                            d_12_enterInside_ = out5_
                            d_13_enterCurrent_ = out6_
                            generated = d_11_enterGenerated_
                            insideConstrainedOut = d_12_enterInside_
                            currentConstrainedOut = d_13_enterCurrent_
                            d_3_spanTokenCount_ = 0
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_14_closedGenerated_: _dafny.Seq
                        d_15_closedInside_: bool
                        d_16_closedCurrent_: _dafny.Seq
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: _dafny.Seq
                        out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_14_closedGenerated_ = out7_
                        d_15_closedInside_ = out8_
                        d_16_closedCurrent_ = out9_
                        generated = d_14_closedGenerated_
                        insideConstrainedOut = d_15_closedInside_
                        currentConstrainedOut = d_16_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_3_spanTokenCount_ = 0
                    elif (d_3_spanTokenCount_) >= (d_4_maxSpanTokens_):
                        d_17_rolledGenerated_: _dafny.Seq
                        d_18_rolledCurrent_: _dafny.Seq
                        out10_: _dafny.Seq
                        out11_: _dafny.Seq
                        out10_, out11_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                        d_17_rolledGenerated_ = out10_
                        d_18_rolledCurrent_ = out11_
                        generated = d_17_rolledGenerated_
                        currentConstrainedOut = d_18_rolledCurrent_
                        d_3_spanTokenCount_ = len(currentConstrainedOut)
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            d_19_closedGenerated_: _dafny.Seq
                            d_20_closedInside_: bool
                            d_21_closedCurrent_: _dafny.Seq
                            out12_: _dafny.Seq
                            out13_: bool
                            out14_: _dafny.Seq
                            out12_, out13_, out14_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_19_closedGenerated_ = out12_
                            d_20_closedInside_ = out13_
                            d_21_closedCurrent_ = out14_
                            generated = d_19_closedGenerated_
                            insideConstrainedOut = d_20_closedInside_
                            currentConstrainedOut = d_21_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_3_spanTokenCount_ = 0
                        elif True:
                            d_22_constrainedPrompt_: _dafny.Seq
                            d_22_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_23_next_: _dafny.Seq
                            d_24_wasConstrained_: bool
                            out15_: _dafny.Seq
                            out16_: bool
                            out15_, out16_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, d_22_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_23_next_ = out15_
                            d_24_wasConstrained_ = out16_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_23_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_25_appendedGenerated_: _dafny.Seq
                                d_26_appendedInside_: bool
                                d_27_appendedCurrent_: _dafny.Seq
                                out17_: _dafny.Seq
                                out18_: bool
                                out19_: _dafny.Seq
                                out17_, out18_, out19_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_23_next_)
                                d_25_appendedGenerated_ = out17_
                                d_26_appendedInside_ = out18_
                                d_27_appendedCurrent_ = out19_
                                generated = d_25_appendedGenerated_
                                insideConstrainedOut = d_26_appendedInside_
                                currentConstrainedOut = d_27_appendedCurrent_
                                d_3_spanTokenCount_ = (d_3_spanTokenCount_) + (1)
                    elif True:
                        d_28_constrainedPrompt_: _dafny.Seq
                        d_28_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_29_next_: _dafny.Seq
                        d_30_wasConstrained_: bool
                        out20_: _dafny.Seq
                        out21_: bool
                        out20_, out21_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, d_28_constrainedPrompt_, currentConstrainedOut, eosToken)
                        d_29_next_ = out20_
                        d_30_wasConstrained_ = out21_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_29_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_31_appendedGenerated_: _dafny.Seq
                            d_32_appendedInside_: bool
                            d_33_appendedCurrent_: _dafny.Seq
                            out22_: _dafny.Seq
                            out23_: bool
                            out24_: _dafny.Seq
                            out22_, out23_, out24_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_29_next_)
                            d_31_appendedGenerated_ = out22_
                            d_32_appendedInside_ = out23_
                            d_33_appendedCurrent_ = out24_
                            generated = d_31_appendedGenerated_
                            insideConstrainedOut = d_32_appendedInside_
                            currentConstrainedOut = d_33_appendedCurrent_
                            d_3_spanTokenCount_ = (d_3_spanTokenCount_) + (1)
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

