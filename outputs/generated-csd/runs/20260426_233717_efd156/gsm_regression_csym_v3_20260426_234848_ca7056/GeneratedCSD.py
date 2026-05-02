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
    def MyCSDStrategy(lm, parser, prompt, generatedPrefix, insideConstrained, currentConstrained, maxSteps, eosToken):
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
        d_2_done_: bool
        d_2_done_ = False
        while ((d_1_steps_) < (maxSteps)) and (not(d_2_done_)):
            if not(insideConstrainedOut):
                d_3_remaining_: int
                d_3_remaining_ = (maxSteps) - (d_1_steps_)
                d_4_chunkGenerated_: _dafny.Seq
                d_5_stoppedOnOpenSpan_: bool
                d_6_stoppedOnEos_: bool
                d_7_stepsUsed_: int
                out0_: _dafny.Seq
                out1_: bool
                out2_: bool
                out3_: int
                out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_3_remaining_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                d_4_chunkGenerated_ = out0_
                d_5_stoppedOnOpenSpan_ = out1_
                d_6_stoppedOnEos_ = out2_
                d_7_stepsUsed_ = out3_
                generated = d_4_chunkGenerated_
                d_1_steps_ = (d_1_steps_) + (d_7_stepsUsed_)
                if d_6_stoppedOnEos_:
                    d_2_done_ = True
                elif True:
                    if d_5_stoppedOnOpenSpan_:
                        insideConstrainedOut = True
                        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
            elif True:
                d_8_remainingInside_: int
                d_8_remainingInside_ = (maxSteps) - (d_1_steps_)
                d_9_complete0_: bool
                d_9_complete0_ = (parser).IsCompletePrefix(currentConstrainedOut)
                if d_9_complete0_:
                    d_10_closedGenerated_: _dafny.Seq
                    d_11_closedInside_: bool
                    d_12_closedCurrent_: _dafny.Seq
                    out4_: _dafny.Seq
                    out5_: bool
                    out6_: _dafny.Seq
                    out4_, out5_, out6_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                    d_10_closedGenerated_ = out4_
                    d_11_closedInside_ = out5_
                    d_12_closedCurrent_ = out6_
                    generated = d_10_closedGenerated_
                    insideConstrainedOut = d_11_closedInside_
                    currentConstrainedOut = d_12_closedCurrent_
                    d_1_steps_ = (d_1_steps_) + (1)
                elif True:
                    if (d_8_remainingInside_) == (1):
                        d_2_done_ = True
                    elif True:
                        d_13_narrow_: bool
                        out7_: bool
                        out7_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                        d_13_narrow_ = out7_
                        if d_13_narrow_:
                            d_14_rolled_: _dafny.Seq
                            out8_: _dafny.Seq
                            out8_ = VerifiedDecoderAgent.CSDHelpers.RollbackToValidPrefix(parser, currentConstrainedOut)
                            d_14_rolled_ = out8_
                            d_15_rolledGenerated_: _dafny.Seq
                            d_16_rolledCurrent_: _dafny.Seq
                            out9_: _dafny.Seq
                            out10_: _dafny.Seq
                            out9_, out10_ = (d_0_helpers_).RollbackConstrainedSpan(parser, d_14_rolled_, generated, currentConstrainedOut)
                            d_15_rolledGenerated_ = out9_
                            d_16_rolledCurrent_ = out10_
                            generated = d_15_rolledGenerated_
                            currentConstrainedOut = d_16_rolledCurrent_
                            insideConstrainedOut = True
                            d_17_completeAfterRollback_: bool
                            d_17_completeAfterRollback_ = (parser).IsCompletePrefix(currentConstrainedOut)
                            if d_17_completeAfterRollback_:
                                d_18_closedGenerated2_: _dafny.Seq
                                d_19_closedInside2_: bool
                                d_20_closedCurrent2_: _dafny.Seq
                                out11_: _dafny.Seq
                                out12_: bool
                                out13_: _dafny.Seq
                                out11_, out12_, out13_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_18_closedGenerated2_ = out11_
                                d_19_closedInside2_ = out12_
                                d_20_closedCurrent2_ = out13_
                                generated = d_18_closedGenerated2_
                                insideConstrainedOut = d_19_closedInside2_
                                currentConstrainedOut = d_20_closedCurrent2_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                d_21_next_: _dafny.Seq
                                out14_: _dafny.Seq
                                out14_ = (d_0_helpers_).ConstrainedStep(lm, parser, prompt, currentConstrainedOut, eosToken)
                                d_21_next_ = out14_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_21_next_) == (eosToken):
                                    d_2_done_ = True
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
                            d_25_next2_: _dafny.Seq
                            out18_: _dafny.Seq
                            out18_ = (d_0_helpers_).ConstrainedStep(lm, parser, prompt, currentConstrainedOut, eosToken)
                            d_25_next2_ = out18_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_25_next2_) == (eosToken):
                                d_2_done_ = True
                            elif True:
                                d_26_appendedGenerated2_: _dafny.Seq
                                d_27_appendedInside2_: bool
                                d_28_appendedCurrent2_: _dafny.Seq
                                out19_: _dafny.Seq
                                out20_: bool
                                out21_: _dafny.Seq
                                out19_, out20_, out21_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_25_next2_)
                                d_26_appendedGenerated2_ = out19_
                                d_27_appendedInside2_ = out20_
                                d_28_appendedCurrent2_ = out21_
                                generated = d_26_appendedGenerated2_
                                insideConstrainedOut = d_27_appendedInside2_
                                currentConstrainedOut = d_28_appendedCurrent2_
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

