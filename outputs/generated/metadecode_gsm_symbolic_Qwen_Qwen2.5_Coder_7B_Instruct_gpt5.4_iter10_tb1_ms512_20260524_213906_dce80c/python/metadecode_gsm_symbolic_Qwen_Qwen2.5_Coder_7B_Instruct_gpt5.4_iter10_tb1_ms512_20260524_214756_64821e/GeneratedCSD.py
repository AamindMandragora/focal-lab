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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. Put every arithmetic computation inside visible << and >> delimiters, and keep the arithmetic expression itself inside the delimiters.")))
        (d_0_helpers_).SetNonDeterministic(lm, False)
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_openArmed_: bool
        d_2_openArmed_ = False
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if d_2_openArmed_:
                            d_3_openedGenerated_: _dafny.Seq
                            d_4_openedInside_: bool
                            d_5_openedCurrent_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_3_openedGenerated_ = out0_
                            d_4_openedInside_ = out1_
                            d_5_openedCurrent_ = out2_
                            generated = d_3_openedGenerated_
                            insideConstrainedOut = d_4_openedInside_
                            currentConstrainedOut = d_5_openedCurrent_
                            d_2_openArmed_ = False
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_6_chunkBudget_: int
                            if ((maxSteps) - (d_1_steps_)) >= (4):
                                d_6_chunkBudget_ = 4
                            elif True:
                                d_6_chunkBudget_ = (maxSteps) - (d_1_steps_)
                            d_7_chunkedGenerated_: _dafny.Seq
                            d_8_stoppedOnOpenSpan_: bool
                            d_9_stoppedOnEos_: bool
                            d_10_stepsUsed_: int
                            out3_: _dafny.Seq
                            out4_: bool
                            out5_: bool
                            out6_: int
                            out3_, out4_, out5_, out6_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_6_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_7_chunkedGenerated_ = out3_
                            d_8_stoppedOnOpenSpan_ = out4_
                            d_9_stoppedOnEos_ = out5_
                            d_10_stepsUsed_ = out6_
                            generated = d_7_chunkedGenerated_
                            d_1_steps_ = (d_1_steps_) + (d_10_stepsUsed_)
                            if d_9_stoppedOnEos_:
                                raise _dafny.Break("0")
                            elif True:
                                if d_8_stoppedOnOpenSpan_:
                                    d_11_enteredGenerated_: _dafny.Seq
                                    d_12_enteredInside_: bool
                                    d_13_enteredCurrent_: _dafny.Seq
                                    out7_: _dafny.Seq
                                    out8_: bool
                                    out9_: _dafny.Seq
                                    out7_, out8_, out9_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                    d_11_enteredGenerated_ = out7_
                                    d_12_enteredInside_ = out8_
                                    d_13_enteredCurrent_ = out9_
                                    generated = d_11_enteredGenerated_
                                    insideConstrainedOut = d_12_enteredInside_
                                    currentConstrainedOut = d_13_enteredCurrent_
                                    d_2_openArmed_ = False
                                elif (len(generated)) > (0):
                                    d_14_lastTok_: _dafny.Seq
                                    d_14_lastTok_ = (generated)[(len(generated)) - (1)]
                                    if (((((d_14_lastTok_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")))) or ((d_14_lastTok_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "+"))))) or ((d_14_lastTok_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "-"))))) or ((d_14_lastTok_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "*"))))) or ((d_14_lastTok_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "/")))):
                                        d_2_openArmed_ = True
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
                        d_2_openArmed_ = False
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_18_deadEnd_: bool
                        out13_: bool
                        out13_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                        d_18_deadEnd_ = out13_
                        if d_18_deadEnd_:
                            d_19_rolledGenerated_: _dafny.Seq
                            d_20_rolledCurrent_: _dafny.Seq
                            out14_: _dafny.Seq
                            out15_: _dafny.Seq
                            out14_, out15_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                            d_19_rolledGenerated_ = out14_
                            d_20_rolledCurrent_ = out15_
                            generated = d_19_rolledGenerated_
                            insideConstrainedOut = True
                            currentConstrainedOut = d_20_rolledCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_21_stablePrefix_: _dafny.Seq
                            d_21_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_22_constrainedPrompt_: _dafny.Seq
                            d_22_constrainedPrompt_ = (prompt) + (d_21_stablePrefix_)
                            d_23_remaining_: int
                            d_23_remaining_ = (maxSteps) - (d_1_steps_)
                            d_24_symbolBudget_: int
                            if (stepTokenBudget) == (0):
                                d_24_symbolBudget_ = 1
                            elif (stepTokenBudget) > (d_23_remaining_):
                                d_24_symbolBudget_ = d_23_remaining_
                            elif True:
                                d_24_symbolBudget_ = stepTokenBudget
                            d_25_symbolGenerated_: _dafny.Seq
                            d_26_symbolCurrent_: _dafny.Seq
                            d_27_hitEos_: bool
                            d_28_symbolSteps_: int
                            out16_: _dafny.Seq
                            out17_: _dafny.Seq
                            out18_: bool
                            out19_: int
                            out16_, out17_, out18_, out19_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_22_constrainedPrompt_, generated, currentConstrainedOut, d_24_symbolBudget_, eosToken)
                            d_25_symbolGenerated_ = out16_
                            d_26_symbolCurrent_ = out17_
                            d_27_hitEos_ = out18_
                            d_28_symbolSteps_ = out19_
                            generated = d_25_symbolGenerated_
                            insideConstrainedOut = True
                            currentConstrainedOut = d_26_symbolCurrent_
                            d_1_steps_ = (d_1_steps_) + (d_28_symbolSteps_)
                            if d_27_hitEos_:
                                raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

