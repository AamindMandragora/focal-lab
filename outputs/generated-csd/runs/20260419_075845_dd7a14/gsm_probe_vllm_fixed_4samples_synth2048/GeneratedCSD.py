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
                d_2_next_: _dafny.Seq
                out0_: _dafny.Seq
                out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                d_2_next_ = out0_
                if (d_2_next_) == (eosToken):
                    d_1_steps_ = (d_1_steps_) + (1)
                elif True:
                    if (d_2_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                        d_3_generated2_: _dafny.Seq
                        d_4_inside2_: bool
                        d_5_current2_: _dafny.Seq
                        out1_: _dafny.Seq
                        out2_: bool
                        out3_: _dafny.Seq
                        out1_, out2_, out3_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                        d_3_generated2_ = out1_
                        d_4_inside2_ = out2_
                        d_5_current2_ = out3_
                        generated = d_3_generated2_
                        insideConstrainedOut = d_4_inside2_
                        currentConstrainedOut = d_5_current2_
                    elif True:
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_2_next_]))
                    d_1_steps_ = (d_1_steps_) + (1)
            elif True:
                d_6_isComplete_: bool
                d_6_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                if d_6_isComplete_:
                    d_7_generated3_: _dafny.Seq
                    d_8_inside3_: bool
                    d_9_current3_: _dafny.Seq
                    out4_: _dafny.Seq
                    out5_: bool
                    out6_: _dafny.Seq
                    out4_, out5_, out6_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                    d_7_generated3_ = out4_
                    d_8_inside3_ = out5_
                    d_9_current3_ = out6_
                    generated = d_7_generated3_
                    insideConstrainedOut = d_8_inside3_
                    currentConstrainedOut = d_9_current3_
                    d_1_steps_ = (d_1_steps_) + (1)
                elif True:
                    d_10_next2_: _dafny.Seq
                    out7_: _dafny.Seq
                    out7_ = (d_0_helpers_).ConstrainedStep(lm, parser, prompt, currentConstrainedOut, eosToken)
                    d_10_next2_ = out7_
                    if (d_10_next2_) == (eosToken):
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_11_generated4_: _dafny.Seq
                        d_12_inside4_: bool
                        d_13_current4_: _dafny.Seq
                        out8_: _dafny.Seq
                        out9_: bool
                        out10_: _dafny.Seq
                        out8_, out9_, out10_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_10_next2_)
                        d_11_generated4_ = out8_
                        d_12_inside4_ = out9_
                        d_13_current4_ = out10_
                        generated = d_11_generated4_
                        insideConstrainedOut = d_12_inside4_
                        currentConstrainedOut = d_13_current4_
                        d_1_steps_ = (d_1_steps_) + (1)
            if (d_1_steps_) < (maxSteps):
                if (((generated) == (generatedPrefix)) and ((insideConstrainedOut) == (insideConstrained))) and ((currentConstrainedOut) == (currentConstrained)):
                    d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

