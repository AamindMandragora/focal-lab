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
        while (d_1_steps_) < (maxSteps):
            if not(insideConstrainedOut):
                d_2_chunkLimit_: int
                d_2_chunkLimit_ = (maxSteps) - (d_1_steps_)
                if (d_2_chunkLimit_) > (8):
                    d_2_chunkLimit_ = 8
                d_3_chunkGenerated_: _dafny.Seq
                d_4_stoppedOnOpenSpan_: bool
                d_5_stoppedOnEos_: bool
                d_6_stepsUsed_: int
                out0_: _dafny.Seq
                out1_: bool
                out2_: bool
                out3_: int
                out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_2_chunkLimit_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                d_3_chunkGenerated_ = out0_
                d_4_stoppedOnOpenSpan_ = out1_
                d_5_stoppedOnEos_ = out2_
                d_6_stepsUsed_ = out3_
                generated = d_3_chunkGenerated_
                d_1_steps_ = (d_1_steps_) + (d_6_stepsUsed_)
                if d_5_stoppedOnEos_:
                    pass
                elif True:
                    if d_4_stoppedOnOpenSpan_:
                        insideConstrainedOut = True
                        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                if d_5_stoppedOnEos_:
                    d_1_steps_ = maxSteps
            elif True:
                d_7_isComplete_: bool
                d_7_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                if d_7_isComplete_:
                    d_8_closedGenerated_: _dafny.Seq
                    d_9_closedInside_: bool
                    d_10_closedCurrent_: _dafny.Seq
                    out4_: _dafny.Seq
                    out5_: bool
                    out6_: _dafny.Seq
                    out4_, out5_, out6_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                    d_8_closedGenerated_ = out4_
                    d_9_closedInside_ = out5_
                    d_10_closedCurrent_ = out6_
                    generated = d_8_closedGenerated_
                    insideConstrainedOut = d_9_closedInside_
                    currentConstrainedOut = d_10_closedCurrent_
                    d_1_steps_ = (d_1_steps_) + (1)
                elif True:
                    d_11_stablePrefix_: _dafny.Seq
                    d_11_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                    d_12_constrainedPrompt_: _dafny.Seq
                    d_12_constrainedPrompt_ = (prompt) + (d_11_stablePrefix_)
                    d_13_next_: _dafny.Seq
                    out7_: _dafny.Seq
                    out7_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_12_constrainedPrompt_, currentConstrainedOut, eosToken)
                    d_13_next_ = out7_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if (d_13_next_) == (eosToken):
                        d_1_steps_ = maxSteps
                    elif True:
                        d_14_appendedGenerated_: _dafny.Seq
                        d_15_appendedInside_: bool
                        d_16_appendedCurrent_: _dafny.Seq
                        out8_: _dafny.Seq
                        out9_: bool
                        out10_: _dafny.Seq
                        out8_, out9_, out10_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_13_next_)
                        d_14_appendedGenerated_ = out8_
                        d_15_appendedInside_ = out9_
                        d_16_appendedCurrent_ = out10_
                        generated = d_14_appendedGenerated_
                        insideConstrainedOut = d_15_appendedInside_
                        currentConstrainedOut = d_16_appendedCurrent_
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

