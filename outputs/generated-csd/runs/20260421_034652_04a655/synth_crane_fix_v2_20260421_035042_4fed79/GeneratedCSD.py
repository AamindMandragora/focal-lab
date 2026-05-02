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
        d_2_done_: bool
        d_2_done_ = False
        d_3_chunkLimit_: int
        d_3_chunkLimit_ = 6
        while ((d_1_steps_) < (maxSteps)) and (not(d_2_done_)):
            d_4_remaining_: int
            d_4_remaining_ = (maxSteps) - (d_1_steps_)
            d_5_useChunk_: int
            d_5_useChunk_ = d_3_chunkLimit_
            if (d_4_remaining_) < (d_5_useChunk_):
                d_5_useChunk_ = d_4_remaining_
            d_6_gen2_: _dafny.Seq
            d_7_stoppedOnOpenSpan_: bool
            d_8_stoppedOnEos_: bool
            d_9_stepsUsed_: int
            out0_: _dafny.Seq
            out1_: bool
            out2_: bool
            out3_: int
            out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_5_useChunk_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
            d_6_gen2_ = out0_
            d_7_stoppedOnOpenSpan_ = out1_
            d_8_stoppedOnEos_ = out2_
            d_9_stepsUsed_ = out3_
            generated = d_6_gen2_
            d_1_steps_ = (d_1_steps_) + (d_9_stepsUsed_)
            if d_8_stoppedOnEos_:
                d_2_done_ = True
            elif True:
                if d_7_stoppedOnOpenSpan_:
                    d_2_done_ = False
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

