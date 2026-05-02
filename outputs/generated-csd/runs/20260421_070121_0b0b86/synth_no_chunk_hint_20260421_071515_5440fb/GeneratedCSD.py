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
            if insideConstrainedOut:
                d_3_isComplete_: bool
                d_3_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                if d_3_isComplete_:
                    d_4_closedGenerated_: _dafny.Seq
                    d_5_closedInside_: bool
                    d_6_closedCurrent_: _dafny.Seq
                    out0_: _dafny.Seq
                    out1_: bool
                    out2_: _dafny.Seq
                    out0_, out1_, out2_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                    d_4_closedGenerated_ = out0_
                    d_5_closedInside_ = out1_
                    d_6_closedCurrent_ = out2_
                    generated = d_4_closedGenerated_
                    insideConstrainedOut = d_5_closedInside_
                    currentConstrainedOut = d_6_closedCurrent_
                    d_1_steps_ = (d_1_steps_) + (1)
                elif True:
                    d_7_narrow_: bool
                    out3_: bool
                    out3_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                    d_7_narrow_ = out3_
                    if d_7_narrow_:
                        d_8_stablePrefix_: _dafny.Seq
                        d_8_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_9_rolledGenerated_: _dafny.Seq
                        d_10_rolledCurrent_: _dafny.Seq
                        out4_: _dafny.Seq
                        out5_: _dafny.Seq
                        out4_, out5_ = (d_0_helpers_).RollbackConstrainedSpan(parser, d_8_stablePrefix_, generated, currentConstrainedOut)
                        d_9_rolledGenerated_ = out4_
                        d_10_rolledCurrent_ = out5_
                        generated = d_9_rolledGenerated_
                        currentConstrainedOut = d_10_rolledCurrent_
                    elif True:
                        d_11_stablePrefix2_: _dafny.Seq
                        d_11_stablePrefix2_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_12_constrainedPrompt_: _dafny.Seq
                        d_12_constrainedPrompt_ = (prompt) + (d_11_stablePrefix2_)
                        d_13_next_: _dafny.Seq
                        out6_: _dafny.Seq
                        out6_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_12_constrainedPrompt_, currentConstrainedOut, eosToken)
                        d_13_next_ = out6_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_13_next_) == (eosToken):
                            d_2_done_ = True
                        elif True:
                            d_14_appendedGenerated_: _dafny.Seq
                            d_15_appendedInside_: bool
                            d_16_appendedCurrent_: _dafny.Seq
                            out7_: _dafny.Seq
                            out8_: bool
                            out9_: _dafny.Seq
                            out7_, out8_, out9_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_13_next_)
                            d_14_appendedGenerated_ = out7_
                            d_15_appendedInside_ = out8_
                            d_16_appendedCurrent_ = out9_
                            generated = d_14_appendedGenerated_
                            insideConstrainedOut = d_15_appendedInside_
                            currentConstrainedOut = d_16_appendedCurrent_
            elif True:
                d_17_remaining_: int
                d_17_remaining_ = (maxSteps) - (d_1_steps_)
                if (d_17_remaining_) >= (2):
                    d_18_chunkLimit_: int
                    d_18_chunkLimit_ = 1
                    if (d_17_remaining_) > (3):
                        d_18_chunkLimit_ = 2
                    d_19_chunkGenerated_: _dafny.Seq
                    d_20_stoppedOnOpenSpan_: bool
                    d_21_stoppedOnEos_: bool
                    d_22_stepsUsed_: int
                    out10_: _dafny.Seq
                    out11_: bool
                    out12_: bool
                    out13_: int
                    out10_, out11_, out12_, out13_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_18_chunkLimit_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                    d_19_chunkGenerated_ = out10_
                    d_20_stoppedOnOpenSpan_ = out11_
                    d_21_stoppedOnEos_ = out12_
                    d_22_stepsUsed_ = out13_
                    generated = d_19_chunkGenerated_
                    d_1_steps_ = (d_1_steps_) + (d_22_stepsUsed_)
                    if d_21_stoppedOnEos_:
                        d_2_done_ = True
                    elif True:
                        if d_20_stoppedOnOpenSpan_:
                            insideConstrainedOut = True
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                        elif True:
                            if (d_1_steps_) < (maxSteps):
                                d_23_openedGenerated_: _dafny.Seq
                                d_24_openedInside_: bool
                                d_25_openedCurrent_: _dafny.Seq
                                out14_: _dafny.Seq
                                out15_: bool
                                out16_: _dafny.Seq
                                out14_, out15_, out16_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                d_23_openedGenerated_ = out14_
                                d_24_openedInside_ = out15_
                                d_25_openedCurrent_ = out16_
                                generated = d_23_openedGenerated_
                                insideConstrainedOut = d_24_openedInside_
                                currentConstrainedOut = d_25_openedCurrent_
                                d_1_steps_ = (d_1_steps_) + (1)
                elif True:
                    d_26_next2_: _dafny.Seq
                    out17_: _dafny.Seq
                    out17_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                    d_26_next2_ = out17_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if (d_26_next2_) == (eosToken):
                        d_2_done_ = True
                    elif True:
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_26_next2_]))
                        if VerifiedDecoderAgent.default__.Contains(d_26_next2_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                            insideConstrainedOut = True
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

