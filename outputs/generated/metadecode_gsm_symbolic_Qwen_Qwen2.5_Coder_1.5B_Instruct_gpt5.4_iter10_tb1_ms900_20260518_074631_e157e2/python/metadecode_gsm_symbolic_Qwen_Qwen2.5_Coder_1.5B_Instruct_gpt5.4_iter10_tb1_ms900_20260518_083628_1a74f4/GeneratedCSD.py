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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step. Write each arithmetic computation inside << >> delimiters, and ensure the final arithmetic result appears in a completed << >> span.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_forcedOpenUsed_: bool
        d_2_forcedOpenUsed_ = insideConstrained
        d_3_outsideTokensSinceSpan_: int
        d_3_outsideTokensSinceSpan_ = 0
        d_4_openTrigger_: int
        d_4_openTrigger_ = 24
        d_5_rollbackLimit_: int
        d_5_rollbackLimit_ = 24
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if (not(d_2_forcedOpenUsed_)) and ((d_3_outsideTokensSinceSpan_) >= (d_4_openTrigger_)):
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
                            d_2_forcedOpenUsed_ = True
                            d_3_outsideTokensSinceSpan_ = 0
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_9_remaining_: int
                            d_9_remaining_ = (maxSteps) - (d_1_steps_)
                            d_10_chunkBudget_: int
                            if (d_9_remaining_) < (8):
                                d_10_chunkBudget_ = d_9_remaining_
                            elif True:
                                d_10_chunkBudget_ = 8
                            d_11_chunkedGenerated_: _dafny.Seq
                            d_12_stoppedOnOpenSpan_: bool
                            d_13_stoppedOnEos_: bool
                            d_14_stepsUsed_: int
                            out3_: _dafny.Seq
                            out4_: bool
                            out5_: bool
                            out6_: int
                            out3_, out4_, out5_, out6_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_10_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_11_chunkedGenerated_ = out3_
                            d_12_stoppedOnOpenSpan_ = out4_
                            d_13_stoppedOnEos_ = out5_
                            d_14_stepsUsed_ = out6_
                            generated = d_11_chunkedGenerated_
                            d_1_steps_ = (d_1_steps_) + (d_14_stepsUsed_)
                            d_3_outsideTokensSinceSpan_ = (d_3_outsideTokensSinceSpan_) + (d_14_stepsUsed_)
                            if d_13_stoppedOnEos_:
                                raise _dafny.Break("0")
                            elif d_12_stoppedOnOpenSpan_:
                                d_15_enteredGenerated_: _dafny.Seq
                                d_16_enteredInside_: bool
                                d_17_enteredCurrent_: _dafny.Seq
                                out7_: _dafny.Seq
                                out8_: bool
                                out9_: _dafny.Seq
                                out7_, out8_, out9_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_15_enteredGenerated_ = out7_
                                d_16_enteredInside_ = out8_
                                d_17_enteredCurrent_ = out9_
                                generated = d_15_enteredGenerated_
                                insideConstrainedOut = d_16_enteredInside_
                                currentConstrainedOut = d_17_enteredCurrent_
                                d_2_forcedOpenUsed_ = True
                                d_3_outsideTokensSinceSpan_ = 0
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
                        d_3_outsideTokensSinceSpan_ = 0
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif (len(currentConstrainedOut)) >= (d_5_rollbackLimit_):
                        d_21_rolledGenerated_: _dafny.Seq
                        d_22_rolledCurrent_: _dafny.Seq
                        out13_: _dafny.Seq
                        out14_: _dafny.Seq
                        out13_, out14_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                        d_21_rolledGenerated_ = out13_
                        d_22_rolledCurrent_ = out14_
                        generated = d_21_rolledGenerated_
                        insideConstrainedOut = True
                        currentConstrainedOut = d_22_rolledCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_23_constrainedPrompt_: _dafny.Seq
                        d_23_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_24_remainingInside_: int
                        d_24_remainingInside_ = (maxSteps) - (d_1_steps_)
                        d_25_symbolBudget_: int
                        if ((stepTokenBudget) == (0)) or ((stepTokenBudget) > (d_24_remainingInside_)):
                            d_25_symbolBudget_ = d_24_remainingInside_
                        elif True:
                            d_25_symbolBudget_ = stepTokenBudget
                        d_26_symbolGenerated_: _dafny.Seq
                        d_27_symbolOut_: _dafny.Seq
                        d_28_hitEos_: bool
                        d_29_stepsUsed_: int
                        out15_: _dafny.Seq
                        out16_: _dafny.Seq
                        out17_: bool
                        out18_: int
                        out15_, out16_, out17_, out18_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_23_constrainedPrompt_, generated, currentConstrainedOut, d_25_symbolBudget_, eosToken)
                        d_26_symbolGenerated_ = out15_
                        d_27_symbolOut_ = out16_
                        d_28_hitEos_ = out17_
                        d_29_stepsUsed_ = out18_
                        generated = d_26_symbolGenerated_
                        insideConstrainedOut = True
                        currentConstrainedOut = d_27_symbolOut_
                        d_1_steps_ = (d_1_steps_) + (d_29_stepsUsed_)
                        if d_28_hitEos_:
                            raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

