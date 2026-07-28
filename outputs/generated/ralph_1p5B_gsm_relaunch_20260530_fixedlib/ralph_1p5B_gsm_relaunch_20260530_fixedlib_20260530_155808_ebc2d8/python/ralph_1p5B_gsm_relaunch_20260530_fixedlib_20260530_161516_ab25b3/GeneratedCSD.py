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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step. Show each arithmetic step. Wrap every intermediate arithmetic expression and the final numeric answer inside << >> delimiters. Do NOT nest << >> inside each other. Keep each << >> span to a single arithmetic expression. End with #### followed by the final numeric answer.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_freeChunkSize_: int
        d_2_freeChunkSize_ = 12
        d_3_tokensSinceLastSpan_: int
        d_3_tokensSinceLastSpan_ = 0
        d_4_nestedDelimPenalty_: _dafny.Seq
        d_4_nestedDelimPenalty_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">"))])
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if ((d_3_tokensSinceLastSpan_) >= (d_2_freeChunkSize_)) and (((d_1_steps_) + (2)) <= (maxSteps)):
                            d_5_openGenerated_: _dafny.Seq
                            d_6_openInside_: bool
                            d_7_openCurrent_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_5_openGenerated_ = out0_
                            d_6_openInside_ = out1_
                            d_7_openCurrent_ = out2_
                            generated = d_5_openGenerated_
                            insideConstrainedOut = d_6_openInside_
                            currentConstrainedOut = d_7_openCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_3_tokensSinceLastSpan_ = 0
                        elif True:
                            d_8_remainingSteps_: int
                            d_8_remainingSteps_ = (maxSteps) - (d_1_steps_)
                            d_9_chunkSize_: int
                            d_9_chunkSize_ = d_2_freeChunkSize_
                            if (d_9_chunkSize_) > (d_8_remainingSteps_):
                                d_9_chunkSize_ = d_8_remainingSteps_
                            if (d_9_chunkSize_) == (0):
                                raise _dafny.Break("0")
                            d_10_chunkGenerated_: _dafny.Seq
                            d_11_stoppedOnOpenSpan_: bool
                            d_12_stoppedOnEos_: bool
                            d_13_stepsUsed_: int
                            out3_: _dafny.Seq
                            out4_: bool
                            out5_: bool
                            out6_: int
                            out3_, out4_, out5_, out6_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_9_chunkSize_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_10_chunkGenerated_ = out3_
                            d_11_stoppedOnOpenSpan_ = out4_
                            d_12_stoppedOnEos_ = out5_
                            d_13_stepsUsed_ = out6_
                            generated = d_10_chunkGenerated_
                            d_1_steps_ = (d_1_steps_) + (d_13_stepsUsed_)
                            d_3_tokensSinceLastSpan_ = (d_3_tokensSinceLastSpan_) + (d_13_stepsUsed_)
                            if d_12_stoppedOnEos_:
                                raise _dafny.Break("0")
                            if d_11_stoppedOnOpenSpan_:
                                d_14_enterGenerated_: _dafny.Seq
                                d_15_enterInside_: bool
                                d_16_enterCurrent_: _dafny.Seq
                                out7_: _dafny.Seq
                                out8_: bool
                                out9_: _dafny.Seq
                                out7_, out8_, out9_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_14_enterGenerated_ = out7_
                                d_15_enterInside_ = out8_
                                d_16_enterCurrent_ = out9_
                                generated = d_14_enterGenerated_
                                insideConstrainedOut = d_15_enterInside_
                                currentConstrainedOut = d_16_enterCurrent_
                                d_3_tokensSinceLastSpan_ = 0
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_17_closedGenerated_: _dafny.Seq
                        d_18_closedInside_: bool
                        d_19_closedCurrent_: _dafny.Seq
                        out10_: _dafny.Seq
                        out11_: bool
                        out12_: _dafny.Seq
                        out10_, out11_, out12_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_17_closedGenerated_ = out10_
                        d_18_closedInside_ = out11_
                        d_19_closedCurrent_ = out12_
                        generated = d_17_closedGenerated_
                        insideConstrainedOut = d_18_closedInside_
                        currentConstrainedOut = d_19_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_3_tokensSinceLastSpan_ = 0
                    elif True:
                        d_20_constrainedPrompt_: _dafny.Seq
                        d_20_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_21_next_: _dafny.Seq
                        out13_: _dafny.Seq
                        out13_ = (d_0_helpers_).SafePenalizedConstrainedStep(lm, parser, d_20_constrainedPrompt_, currentConstrainedOut, d_4_nestedDelimPenalty_, _dafny.BigRational('8e0'), eosToken)
                        d_21_next_ = out13_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_21_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_22_appendedGenerated_: _dafny.Seq
                            d_23_appendedInside_: bool
                            d_24_appendedCurrent_: _dafny.Seq
                            out14_: _dafny.Seq
                            out15_: bool
                            out16_: _dafny.Seq
                            out14_, out15_, out16_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_21_next_)
                            d_22_appendedGenerated_ = out14_
                            d_23_appendedInside_ = out15_
                            d_24_appendedCurrent_ = out16_
                            generated = d_22_appendedGenerated_
                            insideConstrainedOut = d_23_appendedInside_
                            currentConstrainedOut = d_24_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

