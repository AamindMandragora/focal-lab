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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. Wrap EVERY arithmetic expression in << >> like <<a + b>> or <<x * y - z>>. Final answer must be <<expression>>. Use variables from the problem. Example: Total = <<n1 + n2>>. Answer: <<n1 + n2>>."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_spanTokens_: int
        d_3_spanTokens_ = 0
        d_4_maxSpanTokens_: int
        d_4_maxSpanTokens_ = 20
        d_5_freeTokensSinceSpan_: int
        d_5_freeTokensSinceSpan_ = 0
        d_6_maxFreeBeforeForce_: int
        d_6_maxFreeBeforeForce_ = 30
        d_7_effectiveMaxSteps_: int
        d_7_effectiveMaxSteps_ = maxSteps
        if (d_7_effectiveMaxSteps_) > (400):
            d_7_effectiveMaxSteps_ = 400
        with _dafny.label("0"):
            while (d_2_steps_) < (d_7_effectiveMaxSteps_):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_8_remaining_: int
                        d_8_remaining_ = (d_7_effectiveMaxSteps_) - (d_2_steps_)
                        d_9_chunkSize_: int
                        d_9_chunkSize_ = 4
                        if (d_8_remaining_) < (d_9_chunkSize_):
                            d_9_chunkSize_ = d_8_remaining_
                        if (d_9_chunkSize_) == (0):
                            raise _dafny.Break("0")
                        d_10_genOut_: _dafny.Seq
                        d_11_stoppedOnOpen_: bool
                        d_12_stoppedOnEos_: bool
                        d_13_stepsUsed_: int
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: bool
                        out3_: int
                        out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_9_chunkSize_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_10_genOut_ = out0_
                        d_11_stoppedOnOpen_ = out1_
                        d_12_stoppedOnEos_ = out2_
                        d_13_stepsUsed_ = out3_
                        d_2_steps_ = (d_2_steps_) + (d_13_stepsUsed_)
                        generated = d_10_genOut_
                        if d_12_stoppedOnEos_:
                            raise _dafny.Break("0")
                        elif d_11_stoppedOnOpen_:
                            d_14_newGenerated_: _dafny.Seq
                            d_15_newInside_: bool
                            d_16_newCurrent_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                            d_14_newGenerated_ = out4_
                            d_15_newInside_ = out5_
                            d_16_newCurrent_ = out6_
                            generated = d_14_newGenerated_
                            insideConstrainedOut = d_15_newInside_
                            currentConstrainedOut = d_16_newCurrent_
                            d_3_spanTokens_ = 0
                            d_5_freeTokensSinceSpan_ = 0
                        elif True:
                            d_5_freeTokensSinceSpan_ = (d_5_freeTokensSinceSpan_) + (d_13_stepsUsed_)
                            if ((d_5_freeTokensSinceSpan_) >= (d_6_maxFreeBeforeForce_)) and ((d_2_steps_) < (d_7_effectiveMaxSteps_)):
                                d_17_forcedGenerated_: _dafny.Seq
                                d_18_forcedInside_: bool
                                d_19_forcedCurrent_: _dafny.Seq
                                out7_: _dafny.Seq
                                out8_: bool
                                out9_: _dafny.Seq
                                out7_, out8_, out9_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                d_17_forcedGenerated_ = out7_
                                d_18_forcedInside_ = out8_
                                d_19_forcedCurrent_ = out9_
                                generated = d_17_forcedGenerated_
                                insideConstrainedOut = d_18_forcedInside_
                                currentConstrainedOut = d_19_forcedCurrent_
                                d_2_steps_ = (d_2_steps_) + (1)
                                d_3_spanTokens_ = 0
                                d_5_freeTokensSinceSpan_ = 0
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_20_closedGenerated_: _dafny.Seq
                        d_21_closedInside_: bool
                        d_22_closedCurrent_: _dafny.Seq
                        out10_: _dafny.Seq
                        out11_: bool
                        out12_: _dafny.Seq
                        out10_, out11_, out12_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_20_closedGenerated_ = out10_
                        d_21_closedInside_ = out11_
                        d_22_closedCurrent_ = out12_
                        generated = d_20_closedGenerated_
                        insideConstrainedOut = d_21_closedInside_
                        currentConstrainedOut = d_22_closedCurrent_
                        d_2_steps_ = (d_2_steps_) + (1)
                        d_3_spanTokens_ = 0
                        d_5_freeTokensSinceSpan_ = 0
                    elif (d_3_spanTokens_) >= (d_4_maxSpanTokens_):
                        d_23_rolledGenerated_: _dafny.Seq
                        d_24_rolledCurrent_: _dafny.Seq
                        out13_: _dafny.Seq
                        out14_: _dafny.Seq
                        out13_, out14_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                        d_23_rolledGenerated_ = out13_
                        d_24_rolledCurrent_ = out14_
                        generated = d_23_rolledGenerated_
                        currentConstrainedOut = d_24_rolledCurrent_
                        if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_2_steps_) < (maxSteps)):
                            d_25_closedGenerated_: _dafny.Seq
                            d_26_closedInside_: bool
                            d_27_closedCurrent_: _dafny.Seq
                            out15_: _dafny.Seq
                            out16_: bool
                            out17_: _dafny.Seq
                            out15_, out16_, out17_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_25_closedGenerated_ = out15_
                            d_26_closedInside_ = out16_
                            d_27_closedCurrent_ = out17_
                            generated = d_25_closedGenerated_
                            insideConstrainedOut = d_26_closedInside_
                            currentConstrainedOut = d_27_closedCurrent_
                            d_2_steps_ = (d_2_steps_) + (1)
                            d_3_spanTokens_ = 0
                            d_5_freeTokensSinceSpan_ = 0
                        elif True:
                            insideConstrainedOut = False
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                            d_3_spanTokens_ = 0
                            d_2_steps_ = (d_2_steps_) + (1)
                    elif True:
                        d_28_constrainedPrompt_: _dafny.Seq
                        d_28_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_29_next_: _dafny.Seq
                        out18_: _dafny.Seq
                        out18_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_28_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 15, eosToken)
                        d_29_next_ = out18_
                        d_2_steps_ = (d_2_steps_) + (1)
                        d_3_spanTokens_ = (d_3_spanTokens_) + (1)
                        if (d_29_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_30_appendedGenerated_: _dafny.Seq
                            d_31_appendedInside_: bool
                            d_32_appendedCurrent_: _dafny.Seq
                            out19_: _dafny.Seq
                            out20_: bool
                            out21_: _dafny.Seq
                            out19_, out20_, out21_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_29_next_)
                            d_30_appendedGenerated_ = out19_
                            d_31_appendedInside_ = out20_
                            d_32_appendedCurrent_ = out21_
                            generated = d_30_appendedGenerated_
                            insideConstrainedOut = d_31_appendedInside_
                            currentConstrainedOut = d_32_appendedCurrent_
                    pass
            pass
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

