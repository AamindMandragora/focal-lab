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
            if insideConstrainedOut:
                if (parser).IsCompletePrefix(currentConstrainedOut):
                    d_2_closedGenerated_: _dafny.Seq
                    d_3_closedInside_: bool
                    d_4_closedCurrent_: _dafny.Seq
                    out0_: _dafny.Seq
                    out1_: bool
                    out2_: _dafny.Seq
                    out0_, out1_, out2_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                    d_2_closedGenerated_ = out0_
                    d_3_closedInside_ = out1_
                    d_4_closedCurrent_ = out2_
                    generated = d_2_closedGenerated_
                    insideConstrainedOut = d_3_closedInside_
                    currentConstrainedOut = d_4_closedCurrent_
                    d_1_steps_ = (d_1_steps_) + (1)
                elif True:
                    d_5_nextConstrained_: _dafny.Seq
                    out3_: _dafny.Seq
                    out3_ = (d_0_helpers_).ConstrainedStep(lm, parser, prompt, currentConstrainedOut, eosToken)
                    d_5_nextConstrained_ = out3_
                    if (d_5_nextConstrained_) == (eosToken):
                        d_1_steps_ = maxSteps
                    elif True:
                        d_6_appendedGenerated_: _dafny.Seq
                        d_7_appendedInside_: bool
                        d_8_appendedCurrent_: _dafny.Seq
                        out4_: _dafny.Seq
                        out5_: bool
                        out6_: _dafny.Seq
                        out4_, out5_, out6_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_5_nextConstrained_)
                        d_6_appendedGenerated_ = out4_
                        d_7_appendedInside_ = out5_
                        d_8_appendedCurrent_ = out6_
                        generated = d_6_appendedGenerated_
                        insideConstrainedOut = d_7_appendedInside_
                        currentConstrainedOut = d_8_appendedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
            elif True:
                d_9_next_: _dafny.Seq
                out7_: _dafny.Seq
                out7_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                d_9_next_ = out7_
                if (d_9_next_) == (eosToken):
                    d_1_steps_ = maxSteps
                elif True:
                    if VerifiedDecoderAgent.default__.Contains(d_9_next_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                        d_10_openedGenerated_: _dafny.Seq
                        d_11_openedInside_: bool
                        d_12_openedCurrent_: _dafny.Seq
                        out8_: _dafny.Seq
                        out9_: bool
                        out10_: _dafny.Seq
                        out8_, out9_, out10_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                        d_10_openedGenerated_ = out8_
                        d_11_openedInside_ = out9_
                        d_12_openedCurrent_ = out10_
                        generated = d_10_openedGenerated_
                        insideConstrainedOut = d_11_openedInside_
                        currentConstrainedOut = d_12_openedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_9_next_]))
                        d_1_steps_ = (d_1_steps_) + (1)
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

