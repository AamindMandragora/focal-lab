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
        d_2_rollbackLimit_: int
        d_2_rollbackLimit_ = 24
        d_3_narrowThreshold_: int
        d_3_narrowThreshold_ = 8
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_4_chunkBudget_: int
                        d_4_chunkBudget_ = (maxSteps) - (d_1_steps_)
                        d_5_chunkedGenerated_: _dafny.Seq
                        d_6_stoppedOnOpenSpan_: bool
                        d_7_stoppedOnEos_: bool
                        d_8_stepsUsed_: int
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: bool
                        out3_: int
                        out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_4_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_5_chunkedGenerated_ = out0_
                        d_6_stoppedOnOpenSpan_ = out1_
                        d_7_stoppedOnEos_ = out2_
                        d_8_stepsUsed_ = out3_
                        generated = d_5_chunkedGenerated_
                        d_1_steps_ = (d_1_steps_) + (d_8_stepsUsed_)
                        if d_7_stoppedOnEos_:
                            raise _dafny.Break("0")
                        elif d_6_stoppedOnOpenSpan_:
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
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_12_closedGenerated_: _dafny.Seq
                        d_13_closedInside_: bool
                        d_14_closedCurrent_: _dafny.Seq
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: _dafny.Seq
                        out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_12_closedGenerated_ = out7_
                        d_13_closedInside_ = out8_
                        d_14_closedCurrent_ = out9_
                        generated = d_12_closedGenerated_
                        insideConstrainedOut = d_13_closedInside_
                        currentConstrainedOut = d_14_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                        raise _dafny.Break("0")
                    elif True:
                        d_15_shouldRollback_: bool
                        d_15_shouldRollback_ = False
                        if (len(currentConstrainedOut)) >= (d_2_rollbackLimit_):
                            out10_: bool
                            out10_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                            d_15_shouldRollback_ = out10_
                        if d_15_shouldRollback_:
                            d_16_rolledGenerated_: _dafny.Seq
                            d_17_rolledCurrent_: _dafny.Seq
                            out11_: _dafny.Seq
                            out12_: _dafny.Seq
                            out11_, out12_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                            d_16_rolledGenerated_ = out11_
                            d_17_rolledCurrent_ = out12_
                            generated = d_16_rolledGenerated_
                            insideConstrainedOut = True
                            currentConstrainedOut = d_17_rolledCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_18_stablePrefix_: _dafny.Seq
                            d_18_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_19_constrainedPrompt_: _dafny.Seq
                            d_19_constrainedPrompt_ = (prompt) + (d_18_stablePrefix_)
                            d_20_validCount_: int
                            out13_: int
                            out13_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                            d_20_validCount_ = out13_
                            if ((d_20_validCount_) <= (d_3_narrowThreshold_)) or ((stepTokenBudget) <= (1)):
                                d_21_next_: _dafny.Seq
                                out14_: _dafny.Seq
                                out14_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_19_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_21_next_ = out14_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_21_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_22_appendedGenerated_: _dafny.Seq
                                    d_23_appendedInside_: bool
                                    d_24_appendedCurrent_: _dafny.Seq
                                    out15_: _dafny.Seq
                                    out16_: bool
                                    out17_: _dafny.Seq
                                    out15_, out16_, out17_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_21_next_)
                                    d_22_appendedGenerated_ = out15_
                                    d_23_appendedInside_ = out16_
                                    d_24_appendedCurrent_ = out17_
                                    generated = d_22_appendedGenerated_
                                    insideConstrainedOut = d_23_appendedInside_
                                    currentConstrainedOut = d_24_appendedCurrent_
                            elif True:
                                d_25_remaining_: int
                                d_25_remaining_ = (maxSteps) - (d_1_steps_)
                                d_26_symbolBudget_: int
                                if ((stepTokenBudget) == (0)) or ((stepTokenBudget) > (d_25_remaining_)):
                                    d_26_symbolBudget_ = d_25_remaining_
                                elif True:
                                    d_26_symbolBudget_ = stepTokenBudget
                                d_27_symbolGenerated_: _dafny.Seq
                                d_28_symbolCurrent_: _dafny.Seq
                                d_29_hitEos_: bool
                                d_30_symbolSteps_: int
                                out18_: _dafny.Seq
                                out19_: _dafny.Seq
                                out20_: bool
                                out21_: int
                                out18_, out19_, out20_, out21_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_19_constrainedPrompt_, generated, currentConstrainedOut, d_26_symbolBudget_, eosToken)
                                d_27_symbolGenerated_ = out18_
                                d_28_symbolCurrent_ = out19_
                                d_29_hitEos_ = out20_
                                d_30_symbolSteps_ = out21_
                                generated = d_27_symbolGenerated_
                                insideConstrainedOut = True
                                currentConstrainedOut = d_28_symbolCurrent_
                                d_1_steps_ = (d_1_steps_) + (d_30_symbolSteps_)
                                if d_29_hitEos_:
                                    raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

