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
    def MyCSDStrategy(lm, parser, prompt, generatedPrefix, insideConstrained, currentConstrained, maxSteps, stepTokenBudget, eosToken):
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
        d_2_minCloseLen_: int
        d_2_minCloseLen_ = 24
        d_3_openTok_: _dafny.Seq
        d_3_openTok_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_4_remaining_: int
                        d_4_remaining_ = (maxSteps) - (d_1_steps_)
                        if (d_4_remaining_) == (0):
                            raise _dafny.Break("0")
                        elif True:
                            d_5_chunkGenerated_: _dafny.Seq
                            d_6_stoppedOnOpenSpan_: bool
                            d_7_stoppedOnEos_: bool
                            d_8_stepsUsed_: int
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: bool
                            out3_: int
                            out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_4_remaining_, d_3_openTok_, eosToken)
                            d_5_chunkGenerated_ = out0_
                            d_6_stoppedOnOpenSpan_ = out1_
                            d_7_stoppedOnEos_ = out2_
                            d_8_stepsUsed_ = out3_
                            generated = d_5_chunkGenerated_
                            d_1_steps_ = (d_1_steps_) + (d_8_stepsUsed_)
                            if d_7_stoppedOnEos_:
                                raise _dafny.Break("0")
                            elif True:
                                if (d_6_stoppedOnOpenSpan_) and ((d_1_steps_) < (maxSteps)):
                                    d_9_openedGenerated_: _dafny.Seq
                                    d_10_openedInside_: bool
                                    d_11_openedCurrent_: _dafny.Seq
                                    out4_: _dafny.Seq
                                    out5_: bool
                                    out6_: _dafny.Seq
                                    out4_, out5_, out6_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                    d_9_openedGenerated_ = out4_
                                    d_10_openedInside_ = out5_
                                    d_11_openedCurrent_ = out6_
                                    generated = d_9_openedGenerated_
                                    insideConstrainedOut = d_10_openedInside_
                                    currentConstrainedOut = d_11_openedCurrent_
                                    d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_12_completeNow_: bool
                        d_12_completeNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if (d_12_completeNow_) and ((d_2_minCloseLen_) <= (len(currentConstrainedOut))):
                            d_13_closedGenerated_: _dafny.Seq
                            d_14_closedInside_: bool
                            d_15_closedCurrent_: _dafny.Seq
                            out7_: _dafny.Seq
                            out8_: bool
                            out9_: _dafny.Seq
                            out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_13_closedGenerated_ = out7_
                            d_14_closedInside_ = out8_
                            d_15_closedCurrent_ = out9_
                            generated = d_13_closedGenerated_
                            insideConstrainedOut = d_14_closedInside_
                            currentConstrainedOut = d_15_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                            raise _dafny.Break("0")
                        elif True:
                            d_16_deadEnd_: bool
                            out10_: bool
                            out10_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 0)
                            d_16_deadEnd_ = out10_
                            if d_16_deadEnd_:
                                d_17_stablePrefix_: _dafny.Seq
                                d_17_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                d_18_repairedGenerated_: _dafny.Seq
                                d_19_repairedCurrent_: _dafny.Seq
                                out11_: _dafny.Seq
                                out12_: _dafny.Seq
                                out11_, out12_ = (d_0_helpers_).RollbackConstrainedSpan(parser, d_17_stablePrefix_, generated, currentConstrainedOut)
                                d_18_repairedGenerated_ = out11_
                                d_19_repairedCurrent_ = out12_
                                generated = d_18_repairedGenerated_
                                currentConstrainedOut = d_19_repairedCurrent_
                            elif True:
                                d_20_remaining2_: int
                                d_20_remaining2_ = (maxSteps) - (d_1_steps_)
                                if (d_20_remaining2_) == (0):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_21_constrainedPrompt_: _dafny.Seq
                                    d_21_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                    d_22_budget_: int
                                    d_22_budget_ = stepTokenBudget
                                    if (d_20_remaining2_) < (d_22_budget_):
                                        d_22_budget_ = d_20_remaining2_
                                    if (d_22_budget_) == (0):
                                        raise _dafny.Break("0")
                                    elif True:
                                        d_23_grownCurrent_: _dafny.Seq
                                        d_24_hitEos_: bool
                                        d_25_stepsUsed2_: int
                                        out13_: _dafny.Seq
                                        out14_: bool
                                        out15_: int
                                        out13_, out14_, out15_ = (d_0_helpers_).ConstrainedSymbol(lm, parser, d_21_constrainedPrompt_, currentConstrainedOut, d_22_budget_, eosToken)
                                        d_23_grownCurrent_ = out13_
                                        d_24_hitEos_ = out14_
                                        d_25_stepsUsed2_ = out15_
                                        generated = (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])) + (d_23_grownCurrent_)
                                        currentConstrainedOut = d_23_grownCurrent_
                                        d_1_steps_ = (d_1_steps_) + (d_25_stepsUsed2_)
                                        if d_24_hitEos_:
                                            raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

