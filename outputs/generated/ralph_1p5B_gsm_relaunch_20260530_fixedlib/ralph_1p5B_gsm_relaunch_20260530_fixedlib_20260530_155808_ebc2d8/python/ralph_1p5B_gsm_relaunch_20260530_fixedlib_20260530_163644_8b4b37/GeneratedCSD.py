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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step. Wrap each arithmetic expression and the final answer in << >> delimiters. Keep each << >> span short (one expression only). Example: <<3+4>> = 7. Final answer: <<7>>.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_freeChunkSize_: int
        d_2_freeChunkSize_ = 8
        d_3_tokensSinceLastSpan_: int
        d_3_tokensSinceLastSpan_ = 0
        d_4_maxSpanTokens_: int
        d_4_maxSpanTokens_ = 15
        d_5_spanTokensUsed_: int
        d_5_spanTokensUsed_ = 0
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if ((d_3_tokensSinceLastSpan_) >= (d_2_freeChunkSize_)) and (((d_1_steps_) + (2)) <= (maxSteps)):
                            d_6_openGenerated_: _dafny.Seq
                            d_7_openInside_: bool
                            d_8_openCurrent_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_6_openGenerated_ = out0_
                            d_7_openInside_ = out1_
                            d_8_openCurrent_ = out2_
                            generated = d_6_openGenerated_
                            insideConstrainedOut = d_7_openInside_
                            currentConstrainedOut = d_8_openCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_3_tokensSinceLastSpan_ = 0
                            d_5_spanTokensUsed_ = 0
                        elif True:
                            d_9_remainingSteps_: int
                            d_9_remainingSteps_ = (maxSteps) - (d_1_steps_)
                            d_10_chunkSize_: int
                            d_10_chunkSize_ = d_2_freeChunkSize_
                            if (d_10_chunkSize_) > (d_9_remainingSteps_):
                                d_10_chunkSize_ = d_9_remainingSteps_
                            if (d_10_chunkSize_) == (0):
                                raise _dafny.Break("0")
                            d_11_chunkGenerated_: _dafny.Seq
                            d_12_stoppedOnOpenSpan_: bool
                            d_13_stoppedOnEos_: bool
                            d_14_stepsUsed_: int
                            out3_: _dafny.Seq
                            out4_: bool
                            out5_: bool
                            out6_: int
                            out3_, out4_, out5_, out6_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_10_chunkSize_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_11_chunkGenerated_ = out3_
                            d_12_stoppedOnOpenSpan_ = out4_
                            d_13_stoppedOnEos_ = out5_
                            d_14_stepsUsed_ = out6_
                            generated = d_11_chunkGenerated_
                            d_1_steps_ = (d_1_steps_) + (d_14_stepsUsed_)
                            d_3_tokensSinceLastSpan_ = (d_3_tokensSinceLastSpan_) + (d_14_stepsUsed_)
                            if d_13_stoppedOnEos_:
                                raise _dafny.Break("0")
                            if d_12_stoppedOnOpenSpan_:
                                d_15_enterGenerated_: _dafny.Seq
                                d_16_enterInside_: bool
                                d_17_enterCurrent_: _dafny.Seq
                                out7_: _dafny.Seq
                                out8_: bool
                                out9_: _dafny.Seq
                                out7_, out8_, out9_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_15_enterGenerated_ = out7_
                                d_16_enterInside_ = out8_
                                d_17_enterCurrent_ = out9_
                                generated = d_15_enterGenerated_
                                insideConstrainedOut = d_16_enterInside_
                                currentConstrainedOut = d_17_enterCurrent_
                                d_3_tokensSinceLastSpan_ = 0
                                d_5_spanTokensUsed_ = 0
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_18_closedGenerated_: _dafny.Seq
                        d_19_closedInside_: bool
                        d_20_closedCurrent_: _dafny.Seq
                        out10_: _dafny.Seq
                        out11_: bool
                        out12_: _dafny.Seq
                        out10_, out11_, out12_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_18_closedGenerated_ = out10_
                        d_19_closedInside_ = out11_
                        d_20_closedCurrent_ = out12_
                        generated = d_18_closedGenerated_
                        insideConstrainedOut = d_19_closedInside_
                        currentConstrainedOut = d_20_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_3_tokensSinceLastSpan_ = 0
                        d_5_spanTokensUsed_ = 0
                    elif (d_5_spanTokensUsed_) >= (d_4_maxSpanTokens_):
                        d_21_rolledGenerated_: _dafny.Seq
                        d_22_rolledCurrent_: _dafny.Seq
                        out13_: _dafny.Seq
                        out14_: _dafny.Seq
                        out13_, out14_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                        d_21_rolledGenerated_ = out13_
                        d_22_rolledCurrent_ = out14_
                        generated = d_21_rolledGenerated_
                        currentConstrainedOut = d_22_rolledCurrent_
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            d_23_closedGenerated_: _dafny.Seq
                            d_24_closedInside_: bool
                            d_25_closedCurrent_: _dafny.Seq
                            out15_: _dafny.Seq
                            out16_: bool
                            out17_: _dafny.Seq
                            out15_, out16_, out17_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_23_closedGenerated_ = out15_
                            d_24_closedInside_ = out16_
                            d_25_closedCurrent_ = out17_
                            generated = d_23_closedGenerated_
                            insideConstrainedOut = d_24_closedInside_
                            currentConstrainedOut = d_25_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_3_tokensSinceLastSpan_ = 0
                            d_5_spanTokensUsed_ = 0
                        elif True:
                            d_26_constrainedPrompt_: _dafny.Seq
                            d_26_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_27_next_: _dafny.Seq
                            d_28_wasConstrained_: bool
                            out18_: _dafny.Seq
                            out19_: bool
                            out18_, out19_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, d_26_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_27_next_ = out18_
                            d_28_wasConstrained_ = out19_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_5_spanTokensUsed_ = (d_5_spanTokensUsed_) + (1)
                            if (d_27_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_29_appendedGenerated_: _dafny.Seq
                                d_30_appendedInside_: bool
                                d_31_appendedCurrent_: _dafny.Seq
                                out20_: _dafny.Seq
                                out21_: bool
                                out22_: _dafny.Seq
                                out20_, out21_, out22_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_27_next_)
                                d_29_appendedGenerated_ = out20_
                                d_30_appendedInside_ = out21_
                                d_31_appendedCurrent_ = out22_
                                generated = d_29_appendedGenerated_
                                insideConstrainedOut = d_30_appendedInside_
                                currentConstrainedOut = d_31_appendedCurrent_
                    elif True:
                        d_32_constrainedPrompt_: _dafny.Seq
                        d_32_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_33_next_: _dafny.Seq
                        d_34_wasConstrained_: bool
                        out23_: _dafny.Seq
                        out24_: bool
                        out23_, out24_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, d_32_constrainedPrompt_, currentConstrainedOut, eosToken)
                        d_33_next_ = out23_
                        d_34_wasConstrained_ = out24_
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_5_spanTokensUsed_ = (d_5_spanTokensUsed_) + (1)
                        if (d_33_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_35_appendedGenerated_: _dafny.Seq
                            d_36_appendedInside_: bool
                            d_37_appendedCurrent_: _dafny.Seq
                            out25_: _dafny.Seq
                            out26_: bool
                            out27_: _dafny.Seq
                            out25_, out26_, out27_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_33_next_)
                            d_35_appendedGenerated_ = out25_
                            d_36_appendedInside_ = out26_
                            d_37_appendedCurrent_ = out27_
                            generated = d_35_appendedGenerated_
                            insideConstrainedOut = d_36_appendedInside_
                            currentConstrainedOut = d_37_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

