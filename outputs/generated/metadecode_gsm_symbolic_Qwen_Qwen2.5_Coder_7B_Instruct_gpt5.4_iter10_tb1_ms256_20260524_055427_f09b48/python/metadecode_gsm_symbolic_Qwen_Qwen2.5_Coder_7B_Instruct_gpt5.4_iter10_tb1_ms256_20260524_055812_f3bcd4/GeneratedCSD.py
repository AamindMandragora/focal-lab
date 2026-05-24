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
        d_3_openThreshold_: int
        d_3_openThreshold_ = 24
        d_4_chunkCap_: int
        d_4_chunkCap_ = 8
        d_5_narrowThreshold_: int
        d_5_narrowThreshold_ = 12
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if (d_2_outsideSinceSpan_) >= (d_3_openThreshold_):
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
                            d_2_outsideSinceSpan_ = 0
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_9_remaining_: int
                            d_9_remaining_ = (maxSteps) - (d_1_steps_)
                            d_10_chunkBudget_: int
                            if (d_9_remaining_) < (d_4_chunkCap_):
                                d_10_chunkBudget_ = d_9_remaining_
                            elif True:
                                d_10_chunkBudget_ = d_4_chunkCap_
                            d_11_oldLen_: int
                            d_11_oldLen_ = len(generated)
                            d_12_chunkedGenerated_: _dafny.Seq
                            d_13_stoppedOnOpenSpan_: bool
                            d_14_stoppedOnEos_: bool
                            d_15_stepsUsed_: int
                            out3_: _dafny.Seq
                            out4_: bool
                            out5_: bool
                            out6_: int
                            out3_, out4_, out5_, out6_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_10_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_12_chunkedGenerated_ = out3_
                            d_13_stoppedOnOpenSpan_ = out4_
                            d_14_stoppedOnEos_ = out5_
                            d_15_stepsUsed_ = out6_
                            generated = d_12_chunkedGenerated_
                            d_1_steps_ = (d_1_steps_) + (d_15_stepsUsed_)
                            d_2_outsideSinceSpan_ = (d_2_outsideSinceSpan_) + ((len(generated)) - (d_11_oldLen_))
                            if d_14_stoppedOnEos_:
                                raise _dafny.Break("0")
                            elif d_13_stoppedOnOpenSpan_:
                                d_16_enteredGenerated_: _dafny.Seq
                                d_17_enteredInside_: bool
                                d_18_enteredCurrent_: _dafny.Seq
                                out7_: _dafny.Seq
                                out8_: bool
                                out9_: _dafny.Seq
                                out7_, out8_, out9_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_16_enteredGenerated_ = out7_
                                d_17_enteredInside_ = out8_
                                d_18_enteredCurrent_ = out9_
                                generated = d_16_enteredGenerated_
                                insideConstrainedOut = d_17_enteredInside_
                                currentConstrainedOut = d_18_enteredCurrent_
                                d_2_outsideSinceSpan_ = 0
                    elif True:
                        d_19_isComplete_: bool
                        d_19_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_19_isComplete_:
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
                            d_2_outsideSinceSpan_ = 0
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_23_stablePrefix_: _dafny.Seq
                            d_23_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_24_constrainedPrompt_: _dafny.Seq
                            d_24_constrainedPrompt_ = (prompt) + (d_23_stablePrefix_)
                            d_25_validCount_: int
                            out13_: int
                            out13_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                            d_25_validCount_ = out13_
                            if (d_25_validCount_) <= (d_5_narrowThreshold_):
                                d_26_next_: _dafny.Seq
                                out14_: _dafny.Seq
                                out14_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_24_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_26_next_ = out14_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_26_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_27_appendedGenerated_: _dafny.Seq
                                    d_28_appendedInside_: bool
                                    d_29_appendedCurrent_: _dafny.Seq
                                    out15_: _dafny.Seq
                                    out16_: bool
                                    out17_: _dafny.Seq
                                    out15_, out16_, out17_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_26_next_)
                                    d_27_appendedGenerated_ = out15_
                                    d_28_appendedInside_ = out16_
                                    d_29_appendedCurrent_ = out17_
                                    generated = d_27_appendedGenerated_
                                    insideConstrainedOut = d_28_appendedInside_
                                    currentConstrainedOut = d_29_appendedCurrent_
                            elif True:
                                d_30_remainingInside_: int
                                d_30_remainingInside_ = (maxSteps) - (d_1_steps_)
                                d_31_symbolBudget_: int
                                if (stepTokenBudget) == (0):
                                    d_31_symbolBudget_ = 1
                                elif (stepTokenBudget) > (d_30_remainingInside_):
                                    d_31_symbolBudget_ = d_30_remainingInside_
                                elif True:
                                    d_31_symbolBudget_ = stepTokenBudget
                                d_32_symbolGenerated_: _dafny.Seq
                                d_33_symbolCurrent_: _dafny.Seq
                                d_34_hitEos_: bool
                                d_35_stepsUsed_: int
                                out18_: _dafny.Seq
                                out19_: _dafny.Seq
                                out20_: bool
                                out21_: int
                                out18_, out19_, out20_, out21_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_24_constrainedPrompt_, generated, currentConstrainedOut, d_31_symbolBudget_, eosToken)
                                d_32_symbolGenerated_ = out18_
                                d_33_symbolCurrent_ = out19_
                                d_34_hitEos_ = out20_
                                d_35_stepsUsed_ = out21_
                                generated = d_32_symbolGenerated_
                                currentConstrainedOut = d_33_symbolCurrent_
                                insideConstrainedOut = True
                                d_1_steps_ = (d_1_steps_) + (d_35_stepsUsed_)
                                if d_34_hitEos_:
                                    raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

