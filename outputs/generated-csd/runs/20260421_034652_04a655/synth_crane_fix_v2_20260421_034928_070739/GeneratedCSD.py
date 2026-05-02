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
        insideConstrainedOut = False
        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
        cost = 0
        d_1_steps_: int
        d_1_steps_ = 0
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
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
                        insideConstrainedOut = False
                        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                        raise _dafny.Break("0")
                    elif True:
                        if d_4_stoppedOnOpenSpan_:
                            insideConstrainedOut = False
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

