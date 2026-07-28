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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step. For EVERY intermediate calculation and the final answer, you MUST wrap the expression in << >> delimiters. Example: <<3+4=7>>. End with the final numeric answer also in << >> delimiters. Do not skip the delimiters.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_freeTokensSinceSpan_: int
        d_2_freeTokensSinceSpan_ = 0
        d_3_maxFreeBeforeForce_: int
        d_3_maxFreeBeforeForce_ = 40
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_4_chunkBudget_: int
                        d_4_chunkBudget_ = 8
                        if ((maxSteps) - (d_1_steps_)) < (d_4_chunkBudget_):
                            d_4_chunkBudget_ = (maxSteps) - (d_1_steps_)
                        if (d_4_chunkBudget_) == (0):
                            raise _dafny.Break("0")
                        if (d_2_freeTokensSinceSpan_) >= (d_3_maxFreeBeforeForce_):
                            if (d_1_steps_) < (maxSteps):
                                d_5_openedGenerated_: _dafny.Seq
                                d_6_openedInside_: bool
                                d_7_openedCurrent_: _dafny.Seq
                                out0_: _dafny.Seq
                                out1_: bool
                                out2_: _dafny.Seq
                                out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                d_5_openedGenerated_ = out0_
                                d_6_openedInside_ = out1_
                                d_7_openedCurrent_ = out2_
                                generated = d_5_openedGenerated_
                                insideConstrainedOut = d_6_openedInside_
                                currentConstrainedOut = d_7_openedCurrent_
                                d_1_steps_ = (d_1_steps_) + (1)
                                d_2_freeTokensSinceSpan_ = 0
                        elif True:
                            d_8_chunkGenerated_: _dafny.Seq
                            d_9_stoppedOnOpenSpan_: bool
                            d_10_stoppedOnEos_: bool
                            d_11_stepsUsed_: int
                            out3_: _dafny.Seq
                            out4_: bool
                            out5_: bool
                            out6_: int
                            out3_, out4_, out5_, out6_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_4_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_8_chunkGenerated_ = out3_
                            d_9_stoppedOnOpenSpan_ = out4_
                            d_10_stoppedOnEos_ = out5_
                            d_11_stepsUsed_ = out6_
                            generated = d_8_chunkGenerated_
                            d_1_steps_ = (d_1_steps_) + (d_11_stepsUsed_)
                            d_2_freeTokensSinceSpan_ = (d_2_freeTokensSinceSpan_) + (d_11_stepsUsed_)
                            if d_10_stoppedOnEos_:
                                raise _dafny.Break("0")
                            elif d_9_stoppedOnOpenSpan_:
                                d_12_enteredGenerated_: _dafny.Seq
                                d_13_enteredInside_: bool
                                d_14_enteredCurrent_: _dafny.Seq
                                out7_: _dafny.Seq
                                out8_: bool
                                out9_: _dafny.Seq
                                out7_, out8_, out9_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_12_enteredGenerated_ = out7_
                                d_13_enteredInside_ = out8_
                                d_14_enteredCurrent_ = out9_
                                generated = d_12_enteredGenerated_
                                insideConstrainedOut = d_13_enteredInside_
                                currentConstrainedOut = d_14_enteredCurrent_
                                d_2_freeTokensSinceSpan_ = 0
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_15_closedGenerated_: _dafny.Seq
                        d_16_closedInside_: bool
                        d_17_closedCurrent_: _dafny.Seq
                        out10_: _dafny.Seq
                        out11_: bool
                        out12_: _dafny.Seq
                        out10_, out11_, out12_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_15_closedGenerated_ = out10_
                        d_16_closedInside_ = out11_
                        d_17_closedCurrent_ = out12_
                        generated = d_15_closedGenerated_
                        insideConstrainedOut = d_16_closedInside_
                        currentConstrainedOut = d_17_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_2_freeTokensSinceSpan_ = 0
                    elif True:
                        d_18_constrainedPrompt_: _dafny.Seq
                        d_18_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_19_next_: _dafny.Seq
                        d_20_wasConstrained_: bool
                        out13_: _dafny.Seq
                        out14_: bool
                        out13_, out14_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, d_18_constrainedPrompt_, currentConstrainedOut, eosToken)
                        d_19_next_ = out13_
                        d_20_wasConstrained_ = out14_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_19_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_21_appendedGenerated_: _dafny.Seq
                            d_22_appendedInside_: bool
                            d_23_appendedCurrent_: _dafny.Seq
                            out15_: _dafny.Seq
                            out16_: bool
                            out17_: _dafny.Seq
                            out15_, out16_, out17_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_19_next_)
                            d_21_appendedGenerated_ = out15_
                            d_22_appendedInside_ = out16_
                            d_23_appendedCurrent_ = out17_
                            generated = d_21_appendedGenerated_
                            insideConstrainedOut = d_22_appendedInside_
                            currentConstrainedOut = d_23_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

