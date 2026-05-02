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
        generated = generatedPrefix
        insideConstrainedOut = insideConstrained
        currentConstrainedOut = currentConstrained
        cost = 0
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_openTok_: _dafny.Seq
        d_2_openTok_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))
        d_3_chunkCap_: int
        d_3_chunkCap_ = 6
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_4_remaining_: int
                        d_4_remaining_ = (maxSteps) - (d_1_steps_)
                        d_5_chunkBudget_: int
                        d_5_chunkBudget_ = d_3_chunkCap_
                        if (d_4_remaining_) < (d_5_chunkBudget_):
                            d_5_chunkBudget_ = d_4_remaining_
                        d_6_chunkGenerated_: _dafny.Seq
                        d_7_stoppedOnOpenSpan_: bool
                        d_8_stoppedOnEos_: bool
                        d_9_stepsUsed_: int
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: bool
                        out3_: int
                        out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_5_chunkBudget_, d_2_openTok_, eosToken)
                        d_6_chunkGenerated_ = out0_
                        d_7_stoppedOnOpenSpan_ = out1_
                        d_8_stoppedOnEos_ = out2_
                        d_9_stepsUsed_ = out3_
                        generated = d_6_chunkGenerated_
                        d_1_steps_ = (d_1_steps_) + (d_9_stepsUsed_)
                        if d_8_stoppedOnEos_:
                            raise _dafny.Break("0")
                        elif True:
                            if d_7_stoppedOnOpenSpan_:
                                if (d_1_steps_) < (maxSteps):
                                    d_10_openedGenerated_: _dafny.Seq
                                    d_11_openedInside_: bool
                                    d_12_openedCurrent_: _dafny.Seq
                                    out4_: _dafny.Seq
                                    out5_: bool
                                    out6_: _dafny.Seq
                                    out4_, out5_, out6_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                    d_10_openedGenerated_ = out4_
                                    d_11_openedInside_ = out5_
                                    d_12_openedCurrent_ = out6_
                                    generated = d_10_openedGenerated_
                                    insideConstrainedOut = d_11_openedInside_
                                    currentConstrainedOut = d_12_openedCurrent_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                elif True:
                                    raise _dafny.Break("0")
                            elif True:
                                raise _dafny.Break("0")
                    elif True:
                        d_13_deadEnd_: bool
                        out7_: bool
                        out7_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                        d_13_deadEnd_ = out7_
                        if d_13_deadEnd_:
                            d_14_stablePrefix_: _dafny.Seq
                            d_14_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_15_repairedGenerated_: _dafny.Seq
                            d_16_repairedCurrent_: _dafny.Seq
                            out8_: _dafny.Seq
                            out9_: _dafny.Seq
                            out8_, out9_ = (d_0_helpers_).RollbackConstrainedSpan(parser, d_14_stablePrefix_, generated, currentConstrainedOut)
                            d_15_repairedGenerated_ = out8_
                            d_16_repairedCurrent_ = out9_
                            generated = d_15_repairedGenerated_
                            currentConstrainedOut = d_16_repairedCurrent_
                            insideConstrainedOut = True
                            d_17_completeAfterRepair_: bool
                            d_17_completeAfterRepair_ = (parser).IsCompletePrefix(currentConstrainedOut)
                            if d_17_completeAfterRepair_:
                                if (d_1_steps_) < (maxSteps):
                                    d_18_closedGenerated0_: _dafny.Seq
                                    d_19_closedInside0_: bool
                                    d_20_closedCurrent0_: _dafny.Seq
                                    out10_: _dafny.Seq
                                    out11_: bool
                                    out12_: _dafny.Seq
                                    out10_, out11_, out12_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_18_closedGenerated0_ = out10_
                                    d_19_closedInside0_ = out11_
                                    d_20_closedCurrent0_ = out12_
                                    generated = d_18_closedGenerated0_
                                    insideConstrainedOut = d_19_closedInside0_
                                    currentConstrainedOut = d_20_closedCurrent0_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                elif True:
                                    raise _dafny.Break("0")
                            elif True:
                                raise _dafny.Break("0")
                        elif True:
                            d_21_completeNow_: bool
                            d_21_completeNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                            if d_21_completeNow_:
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
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                d_25_remaining2_: int
                                d_25_remaining2_ = (maxSteps) - (d_1_steps_)
                                d_26_symbolBudget_: int
                                d_26_symbolBudget_ = stepTokenBudget
                                if (d_25_remaining2_) < (d_26_symbolBudget_):
                                    d_26_symbolBudget_ = d_25_remaining2_
                                if (d_26_symbolBudget_) == (0):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_27_stablePrefix2_: _dafny.Seq
                                    d_27_stablePrefix2_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                    d_28_constrainedPrompt_: _dafny.Seq
                                    d_28_constrainedPrompt_ = (prompt) + (d_27_stablePrefix2_)
                                    d_29_currentOut_: _dafny.Seq
                                    d_30_hitEos_: bool
                                    d_31_stepsUsed2_: int
                                    out16_: _dafny.Seq
                                    out17_: bool
                                    out18_: int
                                    out16_, out17_, out18_ = (d_0_helpers_).ConstrainedSymbol(lm, parser, d_28_constrainedPrompt_, currentConstrainedOut, d_26_symbolBudget_, eosToken)
                                    d_29_currentOut_ = out16_
                                    d_30_hitEos_ = out17_
                                    d_31_stepsUsed2_ = out18_
                                    currentConstrainedOut = d_29_currentOut_
                                    generated = (d_27_stablePrefix2_) + (currentConstrainedOut)
                                    insideConstrainedOut = True
                                    d_1_steps_ = (d_1_steps_) + (d_31_stepsUsed2_)
                                    if d_30_hitEos_:
                                        raise _dafny.Break("0")
                                    elif True:
                                        d_32_completeAfterGrow_: bool
                                        d_32_completeAfterGrow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                                        if d_32_completeAfterGrow_:
                                            if (d_1_steps_) < (maxSteps):
                                                d_33_closedGenerated2_: _dafny.Seq
                                                d_34_closedInside2_: bool
                                                d_35_closedCurrent2_: _dafny.Seq
                                                out19_: _dafny.Seq
                                                out20_: bool
                                                out21_: _dafny.Seq
                                                out19_, out20_, out21_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                                d_33_closedGenerated2_ = out19_
                                                d_34_closedInside2_ = out20_
                                                d_35_closedCurrent2_ = out21_
                                                generated = d_33_closedGenerated2_
                                                insideConstrainedOut = d_34_closedInside2_
                                                currentConstrainedOut = d_35_closedCurrent2_
                                                d_1_steps_ = (d_1_steps_) + (1)
                                            elif True:
                                                raise _dafny.Break("0")
                                        elif True:
                                            if (d_31_stepsUsed2_) == (0):
                                                raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

