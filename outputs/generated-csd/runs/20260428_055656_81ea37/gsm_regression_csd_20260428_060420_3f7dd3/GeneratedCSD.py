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
        d_2_openedOnce_: bool
        d_2_openedOnce_ = (insideConstrained) or ((len(generatedPrefix)) > (0))
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if insideConstrainedOut:
                        d_3_deadEnd_: bool
                        out0_: bool
                        out0_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                        d_3_deadEnd_ = out0_
                        if d_3_deadEnd_:
                            d_4_stablePrefix_: _dafny.Seq
                            d_4_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_5_repairedGenerated_: _dafny.Seq
                            d_6_repairedCurrent_: _dafny.Seq
                            out1_: _dafny.Seq
                            out2_: _dafny.Seq
                            out1_, out2_ = (d_0_helpers_).RollbackConstrainedSpan(parser, d_4_stablePrefix_, generated, currentConstrainedOut)
                            d_5_repairedGenerated_ = out1_
                            d_6_repairedCurrent_ = out2_
                            generated = d_5_repairedGenerated_
                            currentConstrainedOut = d_6_repairedCurrent_
                            insideConstrainedOut = True
                            d_7_completeAfterRepair_: bool
                            d_7_completeAfterRepair_ = (parser).IsCompletePrefix(currentConstrainedOut)
                            if d_7_completeAfterRepair_:
                                if (d_1_steps_) < (maxSteps):
                                    d_8_closedGenerated0_: _dafny.Seq
                                    d_9_closedInside0_: bool
                                    d_10_closedCurrent0_: _dafny.Seq
                                    out3_: _dafny.Seq
                                    out4_: bool
                                    out5_: _dafny.Seq
                                    out3_, out4_, out5_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_8_closedGenerated0_ = out3_
                                    d_9_closedInside0_ = out4_
                                    d_10_closedCurrent0_ = out5_
                                    generated = d_8_closedGenerated0_
                                    insideConstrainedOut = d_9_closedInside0_
                                    currentConstrainedOut = d_10_closedCurrent0_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    d_2_openedOnce_ = True
                                elif True:
                                    raise _dafny.Break("0")
                            elif True:
                                raise _dafny.Break("0")
                        elif True:
                            d_11_completeNow_: bool
                            d_11_completeNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                            if d_11_completeNow_:
                                d_12_closedGenerated1_: _dafny.Seq
                                d_13_closedInside1_: bool
                                d_14_closedCurrent1_: _dafny.Seq
                                out6_: _dafny.Seq
                                out7_: bool
                                out8_: _dafny.Seq
                                out6_, out7_, out8_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_12_closedGenerated1_ = out6_
                                d_13_closedInside1_ = out7_
                                d_14_closedCurrent1_ = out8_
                                generated = d_12_closedGenerated1_
                                insideConstrainedOut = d_13_closedInside1_
                                currentConstrainedOut = d_14_closedCurrent1_
                                d_1_steps_ = (d_1_steps_) + (1)
                                d_2_openedOnce_ = True
                            elif True:
                                d_15_remaining_: int
                                d_15_remaining_ = (maxSteps) - (d_1_steps_)
                                d_16_symbolBudget_: int
                                d_16_symbolBudget_ = stepTokenBudget
                                if (d_15_remaining_) < (d_16_symbolBudget_):
                                    d_16_symbolBudget_ = d_15_remaining_
                                if (d_16_symbolBudget_) == (0):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_17_stablePrefix2_: _dafny.Seq
                                    d_17_stablePrefix2_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                    d_18_constrainedPrompt_: _dafny.Seq
                                    d_18_constrainedPrompt_ = (prompt) + (d_17_stablePrefix2_)
                                    d_19_currentOut_: _dafny.Seq
                                    d_20_hitEos_: bool
                                    d_21_stepsUsed_: int
                                    out9_: _dafny.Seq
                                    out10_: bool
                                    out11_: int
                                    out9_, out10_, out11_ = (d_0_helpers_).ConstrainedSymbol(lm, parser, d_18_constrainedPrompt_, currentConstrainedOut, d_16_symbolBudget_, eosToken)
                                    d_19_currentOut_ = out9_
                                    d_20_hitEos_ = out10_
                                    d_21_stepsUsed_ = out11_
                                    currentConstrainedOut = d_19_currentOut_
                                    generated = (d_17_stablePrefix2_) + (currentConstrainedOut)
                                    insideConstrainedOut = True
                                    d_1_steps_ = (d_1_steps_) + (d_21_stepsUsed_)
                                    if d_20_hitEos_:
                                        raise _dafny.Break("0")
                                    elif True:
                                        d_22_completeAfterGrow_: bool
                                        d_22_completeAfterGrow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                                        if d_22_completeAfterGrow_:
                                            if (d_1_steps_) < (maxSteps):
                                                d_23_closedGenerated2_: _dafny.Seq
                                                d_24_closedInside2_: bool
                                                d_25_closedCurrent2_: _dafny.Seq
                                                out12_: _dafny.Seq
                                                out13_: bool
                                                out14_: _dafny.Seq
                                                out12_, out13_, out14_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                                d_23_closedGenerated2_ = out12_
                                                d_24_closedInside2_ = out13_
                                                d_25_closedCurrent2_ = out14_
                                                generated = d_23_closedGenerated2_
                                                insideConstrainedOut = d_24_closedInside2_
                                                currentConstrainedOut = d_25_closedCurrent2_
                                                d_1_steps_ = (d_1_steps_) + (1)
                                                d_2_openedOnce_ = True
                                            elif True:
                                                raise _dafny.Break("0")
                                        elif True:
                                            if (d_21_stepsUsed_) == (0):
                                                raise _dafny.Break("0")
                    elif True:
                        if not(d_2_openedOnce_):
                            d_26_next_: _dafny.Seq
                            out15_: _dafny.Seq
                            out15_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_26_next_ = out15_
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_26_next_]))
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_26_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_2_openedOnce_ = True
                                if (d_1_steps_) < (maxSteps):
                                    d_27_openedGenerated_: _dafny.Seq
                                    d_28_openedInside_: bool
                                    d_29_openedCurrent_: _dafny.Seq
                                    out16_: _dafny.Seq
                                    out17_: bool
                                    out18_: _dafny.Seq
                                    out16_, out17_, out18_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                    d_27_openedGenerated_ = out16_
                                    d_28_openedInside_ = out17_
                                    d_29_openedCurrent_ = out18_
                                    generated = d_27_openedGenerated_
                                    insideConstrainedOut = d_28_openedInside_
                                    currentConstrainedOut = d_29_openedCurrent_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                elif True:
                                    raise _dafny.Break("0")
                        elif True:
                            d_30_next2_: _dafny.Seq
                            out19_: _dafny.Seq
                            out19_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_30_next2_ = out19_
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_30_next2_]))
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_30_next2_) == (eosToken):
                                raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

