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
        d_2_normalized_: bool
        d_2_normalized_ = False
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(d_2_normalized_):
                        if insideConstrainedOut:
                            d_3_stablePrefix_: _dafny.Seq
                            d_3_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            generated = d_3_stablePrefix_
                            insideConstrainedOut = False
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                            d_2_normalized_ = True
                        elif True:
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                            insideConstrainedOut = False
                            d_2_normalized_ = True
                    elif True:
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
                            out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_4_remaining_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_5_chunkGenerated_ = out0_
                            d_6_stoppedOnOpenSpan_ = out1_
                            d_7_stoppedOnEos_ = out2_
                            d_8_stepsUsed_ = out3_
                            generated = d_5_chunkGenerated_
                            insideConstrainedOut = False
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                            d_1_steps_ = (d_1_steps_) + (d_8_stepsUsed_)
                            if d_7_stoppedOnEos_:
                                raise _dafny.Break("0")
                            elif True:
                                if d_6_stoppedOnOpenSpan_:
                                    raise _dafny.Break("0")
                                elif True:
                                    if (d_8_stepsUsed_) == (0):
                                        raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

