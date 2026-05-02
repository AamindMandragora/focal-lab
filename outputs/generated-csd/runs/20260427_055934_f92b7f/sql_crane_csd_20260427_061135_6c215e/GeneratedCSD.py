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
        d_2_openTok_: _dafny.Seq
        d_2_openTok_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))
        d_3_minCloseLen_: int
        d_3_minCloseLen_ = 40
        d_4_nearEndChoices_: int
        d_4_nearEndChoices_ = 2
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_5_remaining_: int
                        d_5_remaining_ = (maxSteps) - (d_1_steps_)
                        if (d_5_remaining_) == (0):
                            raise _dafny.Break("0")
                        elif True:
                            d_6_chunkGenerated_: _dafny.Seq
                            d_7_stoppedOnOpenSpan_: bool
                            d_8_stoppedOnEos_: bool
                            d_9_stepsUsed_: int
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: bool
                            out3_: int
                            out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_5_remaining_, d_2_openTok_, eosToken)
                            d_6_chunkGenerated_ = out0_
                            d_7_stoppedOnOpenSpan_ = out1_
                            d_8_stoppedOnEos_ = out2_
                            d_9_stepsUsed_ = out3_
                            generated = d_6_chunkGenerated_
                            d_1_steps_ = (d_1_steps_) + (d_9_stepsUsed_)
                            if d_8_stoppedOnEos_:
                                raise _dafny.Break("0")
                            elif True:
                                if (d_7_stoppedOnOpenSpan_) and ((d_1_steps_) < (maxSteps)):
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
                        d_13_deadEnd_: bool
                        out7_: bool
                        out7_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 0)
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
                        elif True:
                            d_17_completeNow_: bool
                            d_17_completeNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                            d_18_validCount_: int
                            out10_: int
                            out10_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                            d_18_validCount_ = out10_
                            d_19_remaining2_: int
                            d_19_remaining2_ = (maxSteps) - (d_1_steps_)
                            if (d_17_completeNow_) and ((((d_3_minCloseLen_) <= (len(currentConstrainedOut))) or ((d_18_validCount_) <= (d_4_nearEndChoices_))) or ((d_19_remaining2_) == (1))):
                                d_20_closedGenerated_: _dafny.Seq
                                d_21_closedInside_: bool
                                d_22_closedCurrent_: _dafny.Seq
                                out11_: _dafny.Seq
                                out12_: bool
                                out13_: _dafny.Seq
                                out11_, out12_, out13_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_20_closedGenerated_ = out11_
                                d_21_closedInside_ = out12_
                                d_22_closedCurrent_ = out13_
                                generated = d_20_closedGenerated_
                                insideConstrainedOut = d_21_closedInside_
                                currentConstrainedOut = d_22_closedCurrent_
                                d_1_steps_ = (d_1_steps_) + (1)
                                raise _dafny.Break("0")
                            elif True:
                                if (d_19_remaining2_) == (0):
                                    raise _dafny.Break("0")
                                elif True:
                                    if d_17_completeNow_:
                                        raise _dafny.Break("0")
                                    elif True:
                                        d_23_next_: _dafny.Seq
                                        out14_: _dafny.Seq
                                        out14_ = (d_0_helpers_).ConstrainedStep(lm, parser, prompt, currentConstrainedOut, eosToken)
                                        d_23_next_ = out14_
                                        if (d_23_next_) == (eosToken):
                                            raise _dafny.Break("0")
                                        elif True:
                                            d_24_appendedGenerated_: _dafny.Seq
                                            d_25_appendedInside_: bool
                                            d_26_appendedCurrent_: _dafny.Seq
                                            out15_: _dafny.Seq
                                            out16_: bool
                                            out17_: _dafny.Seq
                                            out15_, out16_, out17_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_23_next_)
                                            d_24_appendedGenerated_ = out15_
                                            d_25_appendedInside_ = out16_
                                            d_26_appendedCurrent_ = out17_
                                            generated = d_24_appendedGenerated_
                                            insideConstrainedOut = d_25_appendedInside_
                                            currentConstrainedOut = d_26_appendedCurrent_
                                            d_1_steps_ = (d_1_steps_) + (1)
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

