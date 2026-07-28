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
        d_1_guidance_: _dafny.Seq
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step. For each calculation, wrap the symbolic expression in << >> delimiters like <<3 + 4>> or <<n1 + n2>>. End your answer with the final numeric expression as <<answer>>. Example: She has <<n1 + n2>> apples total. The answer is <<n1 + n2>>."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_spanTokens_: int
        d_3_spanTokens_ = 0
        d_4_maxSpanTokens_: int
        d_4_maxSpanTokens_ = 30
        d_5_freeTokensSinceSpan_: int
        d_5_freeTokensSinceSpan_ = 0
        d_6_maxFreeBeforeForce_: int
        d_6_maxFreeBeforeForce_ = 80
        with _dafny.label("0"):
            while (d_2_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_7_remaining_: int
                        d_7_remaining_ = (maxSteps) - (d_2_steps_)
                        d_8_chunkSize_: int
                        d_8_chunkSize_ = 8
                        if (d_7_remaining_) < (d_8_chunkSize_):
                            d_8_chunkSize_ = d_7_remaining_
                        if (d_8_chunkSize_) == (0):
                            raise _dafny.Break("0")
                        d_9_genOut_: _dafny.Seq
                        d_10_stoppedOnOpen_: bool
                        d_11_stoppedOnEos_: bool
                        d_12_stepsUsed_: int
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: bool
                        out3_: int
                        out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_8_chunkSize_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_9_genOut_ = out0_
                        d_10_stoppedOnOpen_ = out1_
                        d_11_stoppedOnEos_ = out2_
                        d_12_stepsUsed_ = out3_
                        d_2_steps_ = (d_2_steps_) + (d_12_stepsUsed_)
                        generated = d_9_genOut_
                        if d_11_stoppedOnEos_:
                            raise _dafny.Break("0")
                        elif d_10_stoppedOnOpen_:
                            d_13_newGenerated_: _dafny.Seq
                            d_14_newInside_: bool
                            d_15_newCurrent_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                            d_13_newGenerated_ = out4_
                            d_14_newInside_ = out5_
                            d_15_newCurrent_ = out6_
                            generated = d_13_newGenerated_
                            insideConstrainedOut = d_14_newInside_
                            currentConstrainedOut = d_15_newCurrent_
                            d_3_spanTokens_ = 0
                            d_5_freeTokensSinceSpan_ = 0
                        elif True:
                            d_5_freeTokensSinceSpan_ = (d_5_freeTokensSinceSpan_) + (d_12_stepsUsed_)
                            if ((d_5_freeTokensSinceSpan_) >= (d_6_maxFreeBeforeForce_)) and ((d_2_steps_) < (maxSteps)):
                                d_16_forcedGenerated_: _dafny.Seq
                                d_17_forcedInside_: bool
                                d_18_forcedCurrent_: _dafny.Seq
                                out7_: _dafny.Seq
                                out8_: bool
                                out9_: _dafny.Seq
                                out7_, out8_, out9_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                d_16_forcedGenerated_ = out7_
                                d_17_forcedInside_ = out8_
                                d_18_forcedCurrent_ = out9_
                                generated = d_16_forcedGenerated_
                                insideConstrainedOut = d_17_forcedInside_
                                currentConstrainedOut = d_18_forcedCurrent_
                                d_2_steps_ = (d_2_steps_) + (1)
                                d_3_spanTokens_ = 0
                                d_5_freeTokensSinceSpan_ = 0
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
                        d_2_steps_ = (d_2_steps_) + (1)
                        d_3_spanTokens_ = 0
                        d_5_freeTokensSinceSpan_ = 0
                    elif (d_3_spanTokens_) >= (d_4_maxSpanTokens_):
                        if (parser).IsCompletePrefix(currentConstrainedOut):
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
                            d_3_spanTokens_ = 0
                        elif True:
                            d_25_rolledGenerated_: _dafny.Seq
                            d_26_rolledCurrent_: _dafny.Seq
                            out16_: _dafny.Seq
                            out17_: _dafny.Seq
                            out16_, out17_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                            d_25_rolledGenerated_ = out16_
                            d_26_rolledCurrent_ = out17_
                            generated = d_25_rolledGenerated_
                            currentConstrainedOut = d_26_rolledCurrent_
                            if (parser).IsCompletePrefix(currentConstrainedOut):
                                d_27_closedGenerated_: _dafny.Seq
                                d_28_closedInside_: bool
                                d_29_closedCurrent_: _dafny.Seq
                                out18_: _dafny.Seq
                                out19_: bool
                                out20_: _dafny.Seq
                                out18_, out19_, out20_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_27_closedGenerated_ = out18_
                                d_28_closedInside_ = out19_
                                d_29_closedCurrent_ = out20_
                                generated = d_27_closedGenerated_
                                insideConstrainedOut = d_28_closedInside_
                                currentConstrainedOut = d_29_closedCurrent_
                                d_2_steps_ = (d_2_steps_) + (1)
                                d_3_spanTokens_ = 0
                            elif True:
                                insideConstrainedOut = False
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                d_2_steps_ = (d_2_steps_) + (1)
                                d_3_spanTokens_ = 0
                    elif True:
                        d_30_constrainedPrompt_: _dafny.Seq
                        d_30_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_31_next_: _dafny.Seq
                        out21_: _dafny.Seq
                        out21_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_30_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('3e0'), eosToken)
                        d_31_next_ = out21_
                        d_2_steps_ = (d_2_steps_) + (1)
                        d_3_spanTokens_ = (d_3_spanTokens_) + (1)
                        if (d_31_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_32_appendedGenerated_: _dafny.Seq
                            d_33_appendedInside_: bool
                            d_34_appendedCurrent_: _dafny.Seq
                            out22_: _dafny.Seq
                            out23_: bool
                            out24_: _dafny.Seq
                            out22_, out23_, out24_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_31_next_)
                            d_32_appendedGenerated_ = out22_
                            d_33_appendedInside_ = out23_
                            d_34_appendedCurrent_ = out24_
                            generated = d_32_appendedGenerated_
                            insideConstrainedOut = d_33_appendedInside_
                            currentConstrainedOut = d_34_appendedCurrent_
                    pass
            pass
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

