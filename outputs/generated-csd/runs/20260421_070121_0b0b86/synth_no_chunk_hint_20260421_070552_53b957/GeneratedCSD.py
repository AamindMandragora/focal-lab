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
        d_2_openedAnySpan_: bool
        d_2_openedAnySpan_ = False
        if (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))) in (generatedPrefix):
            d_2_openedAnySpan_ = True
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
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
                            d_2_openedAnySpan_ = True
                        elif True:
                            d_7_stablePrefix_: _dafny.Seq
                            d_7_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_8_constrainedPrompt_: _dafny.Seq
                            d_8_constrainedPrompt_ = (prompt) + (d_7_stablePrefix_)
                            d_9_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_8_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_9_next_ = out3_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_9_next_) == (eosToken):
                                pass
                            elif True:
                                d_10_appendedGenerated_: _dafny.Seq
                                d_11_appendedInside_: bool
                                d_12_appendedCurrent_: _dafny.Seq
                                out4_: _dafny.Seq
                                out5_: bool
                                out6_: _dafny.Seq
                                out4_, out5_, out6_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_9_next_)
                                d_10_appendedGenerated_ = out4_
                                d_11_appendedInside_ = out5_
                                d_12_appendedCurrent_ = out6_
                                generated = d_10_appendedGenerated_
                                insideConstrainedOut = d_11_appendedInside_
                                currentConstrainedOut = d_12_appendedCurrent_
                            if (d_9_next_) == (eosToken):
                                raise _dafny.Break("0")
                    elif True:
                        if not(d_2_openedAnySpan_):
                            d_13_openedGenerated_: _dafny.Seq
                            d_14_openedInside_: bool
                            d_15_openedCurrent_: _dafny.Seq
                            out7_: _dafny.Seq
                            out8_: bool
                            out9_: _dafny.Seq
                            out7_, out8_, out9_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_13_openedGenerated_ = out7_
                            d_14_openedInside_ = out8_
                            d_15_openedCurrent_ = out9_
                            generated = d_13_openedGenerated_
                            insideConstrainedOut = d_14_openedInside_
                            currentConstrainedOut = d_15_openedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_2_openedAnySpan_ = True
                        elif True:
                            d_16_remaining_: int
                            d_16_remaining_ = (maxSteps) - (d_1_steps_)
                            d_17_chunkLimit_: int
                            d_17_chunkLimit_ = 1
                            if (1) < (d_16_remaining_):
                                d_17_chunkLimit_ = 1
                            elif True:
                                d_17_chunkLimit_ = d_16_remaining_
                            d_18_chunkGenerated_: _dafny.Seq
                            d_19_stoppedOnOpenSpan_: bool
                            d_20_stoppedOnEos_: bool
                            d_21_stepsUsed_: int
                            out10_: _dafny.Seq
                            out11_: bool
                            out12_: bool
                            out13_: int
                            out10_, out11_, out12_, out13_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_17_chunkLimit_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_18_chunkGenerated_ = out10_
                            d_19_stoppedOnOpenSpan_ = out11_
                            d_20_stoppedOnEos_ = out12_
                            d_21_stepsUsed_ = out13_
                            generated = d_18_chunkGenerated_
                            d_1_steps_ = (d_1_steps_) + (d_21_stepsUsed_)
                            if d_20_stoppedOnEos_:
                                raise _dafny.Break("0")
                            elif True:
                                if d_19_stoppedOnOpenSpan_:
                                    insideConstrainedOut = True
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

