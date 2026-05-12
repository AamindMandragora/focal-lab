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
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_narrowThreshold_: int
        d_2_narrowThreshold_ = 12
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_3_remainingChunk_: int
                        d_3_remainingChunk_ = (maxSteps) - (d_1_steps_)
                        d_4_chunkBudget_: int
                        if (d_3_remainingChunk_) > (2):
                            d_4_chunkBudget_ = 2
                        elif True:
                            d_4_chunkBudget_ = d_3_remainingChunk_
                        if (d_4_chunkBudget_) > (0):
                            d_5_chunkedGenerated_: _dafny.Seq
                            d_6_stoppedOpen_: bool
                            d_7_stoppedEos_: bool
                            d_8_stepsUsed_: int
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: bool
                            out3_: int
                            out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_4_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_5_chunkedGenerated_ = out0_
                            d_6_stoppedOpen_ = out1_
                            d_7_stoppedEos_ = out2_
                            d_8_stepsUsed_ = out3_
                            generated = d_5_chunkedGenerated_
                            d_1_steps_ = (d_1_steps_) + (d_8_stepsUsed_)
                            if d_7_stoppedEos_:
                                raise _dafny.Break("0")
                            elif d_6_stoppedOpen_:
                                d_9_enteredGenerated_: _dafny.Seq
                                d_10_enteredInside_: bool
                                d_11_enteredCurrent_: _dafny.Seq
                                out4_: _dafny.Seq
                                out5_: bool
                                out6_: _dafny.Seq
                                out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_9_enteredGenerated_ = out4_
                                d_10_enteredInside_ = out5_
                                d_11_enteredCurrent_ = out6_
                                generated = d_9_enteredGenerated_
                                insideConstrainedOut = d_10_enteredInside_
                                currentConstrainedOut = d_11_enteredCurrent_
                            elif (d_1_steps_) < (maxSteps):
                                d_12_openCount_: int
                                out7_: int
                                out7_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))
                                d_12_openCount_ = out7_
                                if (d_12_openCount_) == (0):
                                    d_13_next_: _dafny.Seq
                                    out8_: _dafny.Seq
                                    out8_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                                    d_13_next_ = out8_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    if (d_13_next_) == (eosToken):
                                        raise _dafny.Break("0")
                                    elif True:
                                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_13_next_]))
                                        if (d_13_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                            insideConstrainedOut = True
                                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                elif True:
                                    d_14_openedGenerated_: _dafny.Seq
                                    d_15_openedInside_: bool
                                    d_16_openedCurrent_: _dafny.Seq
                                    out9_: _dafny.Seq
                                    out10_: bool
                                    out11_: _dafny.Seq
                                    out9_, out10_, out11_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                    d_14_openedGenerated_ = out9_
                                    d_15_openedInside_ = out10_
                                    d_16_openedCurrent_ = out11_
                                    generated = d_14_openedGenerated_
                                    insideConstrainedOut = d_15_openedInside_
                                    currentConstrainedOut = d_16_openedCurrent_
                                    d_1_steps_ = (d_1_steps_) + (1)
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_17_closedGenerated_: _dafny.Seq
                        d_18_closedInside_: bool
                        d_19_closedCurrent_: _dafny.Seq
                        out12_: _dafny.Seq
                        out13_: bool
                        out14_: _dafny.Seq
                        out12_, out13_, out14_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_17_closedGenerated_ = out12_
                        d_18_closedInside_ = out13_
                        d_19_closedCurrent_ = out14_
                        generated = d_17_closedGenerated_
                        insideConstrainedOut = d_18_closedInside_
                        currentConstrainedOut = d_19_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_20_deadEnd_: bool
                        out15_: bool
                        out15_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                        d_20_deadEnd_ = out15_
                        if d_20_deadEnd_:
                            d_21_repairedGenerated_: _dafny.Seq
                            d_22_repairedCurrent_: _dafny.Seq
                            out16_: _dafny.Seq
                            out17_: _dafny.Seq
                            out16_, out17_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                            d_21_repairedGenerated_ = out16_
                            d_22_repairedCurrent_ = out17_
                            generated = d_21_repairedGenerated_
                            insideConstrainedOut = True
                            currentConstrainedOut = d_22_repairedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_23_stablePrefix_: _dafny.Seq
                            d_23_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_24_constrainedPrompt_: _dafny.Seq
                            d_24_constrainedPrompt_ = (prompt) + (d_23_stablePrefix_)
                            d_25_validCount_: int
                            out18_: int
                            out18_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                            d_25_validCount_ = out18_
                            if ((d_25_validCount_) <= (d_2_narrowThreshold_)) or ((stepTokenBudget) <= (1)):
                                d_26_next_: _dafny.Seq
                                out19_: _dafny.Seq
                                out19_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_24_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_2_narrowThreshold_, eosToken)
                                d_26_next_ = out19_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_26_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_27_appendedGenerated_: _dafny.Seq
                                    d_28_appendedInside_: bool
                                    d_29_appendedCurrent_: _dafny.Seq
                                    out20_: _dafny.Seq
                                    out21_: bool
                                    out22_: _dafny.Seq
                                    out20_, out21_, out22_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_26_next_)
                                    d_27_appendedGenerated_ = out20_
                                    d_28_appendedInside_ = out21_
                                    d_29_appendedCurrent_ = out22_
                                    generated = d_27_appendedGenerated_
                                    insideConstrainedOut = d_28_appendedInside_
                                    currentConstrainedOut = d_29_appendedCurrent_
                            elif True:
                                d_30_remaining_: int
                                d_30_remaining_ = (maxSteps) - (d_1_steps_)
                                d_31_symbolBudget_: int
                                if (stepTokenBudget) > (d_30_remaining_):
                                    d_31_symbolBudget_ = d_30_remaining_
                                elif True:
                                    d_31_symbolBudget_ = stepTokenBudget
                                d_32_symbolGenerated_: _dafny.Seq
                                d_33_symbolCurrent_: _dafny.Seq
                                d_34_hitEos_: bool
                                d_35_symbolSteps_: int
                                out23_: _dafny.Seq
                                out24_: _dafny.Seq
                                out25_: bool
                                out26_: int
                                out23_, out24_, out25_, out26_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_24_constrainedPrompt_, generated, currentConstrainedOut, d_31_symbolBudget_, eosToken)
                                d_32_symbolGenerated_ = out23_
                                d_33_symbolCurrent_ = out24_
                                d_34_hitEos_ = out25_
                                d_35_symbolSteps_ = out26_
                                generated = d_32_symbolGenerated_
                                insideConstrainedOut = True
                                currentConstrainedOut = d_33_symbolCurrent_
                                d_1_steps_ = (d_1_steps_) + (d_35_symbolSteps_)
                                if d_34_hitEos_:
                                    raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

