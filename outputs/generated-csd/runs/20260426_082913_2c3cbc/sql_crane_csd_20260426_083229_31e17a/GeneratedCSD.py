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
        d_2_didPrelude_: bool
        d_2_didPrelude_ = False
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if not(d_2_didPrelude_):
                            d_3_remaining_: int
                            d_3_remaining_ = (maxSteps) - (d_1_steps_)
                            d_4_chunkLimit_: int
                            if (d_3_remaining_) < (6):
                                d_4_chunkLimit_ = d_3_remaining_
                            elif True:
                                d_4_chunkLimit_ = 6
                            if (d_4_chunkLimit_) == (0):
                                pass
                            elif True:
                                d_5_chunkGenerated_: _dafny.Seq
                                d_6_stoppedOnOpenSpan_: bool
                                d_7_stoppedOnEos_: bool
                                d_8_stepsUsed_: int
                                out0_: _dafny.Seq
                                out1_: bool
                                out2_: bool
                                out3_: int
                                out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_4_chunkLimit_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                                d_5_chunkGenerated_ = out0_
                                d_6_stoppedOnOpenSpan_ = out1_
                                d_7_stoppedOnEos_ = out2_
                                d_8_stepsUsed_ = out3_
                                generated = d_5_chunkGenerated_
                                d_1_steps_ = (d_1_steps_) + (d_8_stepsUsed_)
                                d_2_didPrelude_ = True
                                if d_7_stoppedOnEos_:
                                    raise _dafny.Break("0")
                                elif True:
                                    if d_6_stoppedOnOpenSpan_:
                                        insideConstrainedOut = True
                                        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                    elif True:
                                        if (d_1_steps_) < (maxSteps):
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
                            d_12_openedGenerated2_: _dafny.Seq
                            d_13_openedInside2_: bool
                            d_14_openedCurrent2_: _dafny.Seq
                            out7_: _dafny.Seq
                            out8_: bool
                            out9_: _dafny.Seq
                            out7_, out8_, out9_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_12_openedGenerated2_ = out7_
                            d_13_openedInside2_ = out8_
                            d_14_openedCurrent2_ = out9_
                            generated = d_12_openedGenerated2_
                            insideConstrainedOut = d_13_openedInside2_
                            currentConstrainedOut = d_14_openedCurrent2_
                            d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_15_complete_: bool
                        d_15_complete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_15_complete_:
                            d_16_closedGenerated_: _dafny.Seq
                            d_17_closedInside_: bool
                            d_18_closedCurrent_: _dafny.Seq
                            out10_: _dafny.Seq
                            out11_: bool
                            out12_: _dafny.Seq
                            out10_, out11_, out12_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_16_closedGenerated_ = out10_
                            d_17_closedInside_ = out11_
                            d_18_closedCurrent_ = out12_
                            generated = d_16_closedGenerated_
                            insideConstrainedOut = d_17_closedInside_
                            currentConstrainedOut = d_18_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                            raise _dafny.Break("0")
                        elif True:
                            d_19_constrainedPrompt_: _dafny.Seq
                            d_19_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_20_next_: _dafny.Seq
                            out13_: _dafny.Seq
                            out13_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_19_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_20_next_ = out13_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_20_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_21_appendedGenerated_: _dafny.Seq
                                d_22_appendedInside_: bool
                                d_23_appendedCurrent_: _dafny.Seq
                                out14_: _dafny.Seq
                                out15_: bool
                                out16_: _dafny.Seq
                                out14_, out15_, out16_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_20_next_)
                                d_21_appendedGenerated_ = out14_
                                d_22_appendedInside_ = out15_
                                d_23_appendedCurrent_ = out16_
                                generated = d_21_appendedGenerated_
                                insideConstrainedOut = d_22_appendedInside_
                                currentConstrainedOut = d_23_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

