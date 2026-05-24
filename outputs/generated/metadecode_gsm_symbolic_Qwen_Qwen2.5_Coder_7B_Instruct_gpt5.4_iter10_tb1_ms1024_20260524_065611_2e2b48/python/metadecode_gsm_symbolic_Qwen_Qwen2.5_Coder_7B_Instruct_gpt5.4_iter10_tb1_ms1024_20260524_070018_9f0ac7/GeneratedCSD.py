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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step. Every arithmetic computation must appear inside visible << >> delimiters, and the final computation should also be inside << >>.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_outsideSinceSpan_: int
        d_2_outsideSinceSpan_ = 0
        d_3_openAfter_: int
        d_3_openAfter_ = 16
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if (d_2_outsideSinceSpan_) >= (d_3_openAfter_):
                            d_4_openedGenerated_: _dafny.Seq
                            d_5_openedInside_: bool
                            d_6_openedCurrent_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_4_openedGenerated_ = out0_
                            d_5_openedInside_ = out1_
                            d_6_openedCurrent_ = out2_
                            generated = d_4_openedGenerated_
                            insideConstrainedOut = d_5_openedInside_
                            currentConstrainedOut = d_6_openedCurrent_
                            d_2_outsideSinceSpan_ = 0
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_7_remaining_: int
                            d_7_remaining_ = (maxSteps) - (d_1_steps_)
                            d_8_chunkBudget_: int
                            if (d_7_remaining_) == (0):
                                d_8_chunkBudget_ = 0
                            elif (d_7_remaining_) < (8):
                                d_8_chunkBudget_ = d_7_remaining_
                            elif True:
                                d_8_chunkBudget_ = 8
                            if (d_8_chunkBudget_) == (0):
                                raise _dafny.Break("0")
                            elif True:
                                d_9_chunkedGenerated_: _dafny.Seq
                                d_10_stoppedOnOpenSpan_: bool
                                d_11_stoppedOnEos_: bool
                                d_12_stepsUsed_: int
                                out3_: _dafny.Seq
                                out4_: bool
                                out5_: bool
                                out6_: int
                                out3_, out4_, out5_, out6_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_8_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                                d_9_chunkedGenerated_ = out3_
                                d_10_stoppedOnOpenSpan_ = out4_
                                d_11_stoppedOnEos_ = out5_
                                d_12_stepsUsed_ = out6_
                                generated = d_9_chunkedGenerated_
                                d_1_steps_ = (d_1_steps_) + (d_12_stepsUsed_)
                                d_2_outsideSinceSpan_ = (d_2_outsideSinceSpan_) + (d_12_stepsUsed_)
                                if d_11_stoppedOnEos_:
                                    raise _dafny.Break("0")
                                elif d_10_stoppedOnOpenSpan_:
                                    d_13_enteredGenerated_: _dafny.Seq
                                    d_14_enteredInside_: bool
                                    d_15_enteredCurrent_: _dafny.Seq
                                    out7_: _dafny.Seq
                                    out8_: bool
                                    out9_: _dafny.Seq
                                    out7_, out8_, out9_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                    d_13_enteredGenerated_ = out7_
                                    d_14_enteredInside_ = out8_
                                    d_15_enteredCurrent_ = out9_
                                    generated = d_13_enteredGenerated_
                                    insideConstrainedOut = d_14_enteredInside_
                                    currentConstrainedOut = d_15_enteredCurrent_
                                    d_2_outsideSinceSpan_ = 0
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_16_closedGenerated_: _dafny.Seq
                        d_17_closedInside_: bool
                        d_18_closedCurrent_: _dafny.Seq
                        out10_: _dafny.Seq
                        out11_: bool
                        out12_: _dafny.Seq
                        out10_, out11_, out12_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_16_closedGenerated_ = out10_
                        d_17_closedInside_ = out11_
                        d_18_closedCurrent_ = out12_
                        generated = d_16_closedGenerated_
                        insideConstrainedOut = d_17_closedInside_
                        currentConstrainedOut = d_18_closedCurrent_
                        d_2_outsideSinceSpan_ = 0
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_19_stablePrefix_: _dafny.Seq
                        d_19_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_20_constrainedPrompt_: _dafny.Seq
                        d_20_constrainedPrompt_ = (prompt) + (d_19_stablePrefix_)
                        d_21_remainingInside_: int
                        d_21_remainingInside_ = (maxSteps) - (d_1_steps_)
                        d_22_symbolBudget_: int
                        if (stepTokenBudget) == (0):
                            d_22_symbolBudget_ = 1
                        elif True:
                            d_22_symbolBudget_ = stepTokenBudget
                        if (d_22_symbolBudget_) > (d_21_remainingInside_):
                            d_22_symbolBudget_ = d_21_remainingInside_
                        if (d_22_symbolBudget_) == (0):
                            raise _dafny.Break("0")
                        elif True:
                            d_23_symbolGenerated_: _dafny.Seq
                            d_24_symbolCurrent_: _dafny.Seq
                            d_25_hitEos_: bool
                            d_26_stepsUsed_: int
                            out13_: _dafny.Seq
                            out14_: _dafny.Seq
                            out15_: bool
                            out16_: int
                            out13_, out14_, out15_, out16_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_20_constrainedPrompt_, generated, currentConstrainedOut, d_22_symbolBudget_, eosToken)
                            d_23_symbolGenerated_ = out13_
                            d_24_symbolCurrent_ = out14_
                            d_25_hitEos_ = out15_
                            d_26_stepsUsed_ = out16_
                            generated = d_23_symbolGenerated_
                            currentConstrainedOut = d_24_symbolCurrent_
                            insideConstrainedOut = True
                            d_2_outsideSinceSpan_ = 0
                            d_1_steps_ = (d_1_steps_) + (d_26_stepsUsed_)
                            if d_25_hitEos_:
                                raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

