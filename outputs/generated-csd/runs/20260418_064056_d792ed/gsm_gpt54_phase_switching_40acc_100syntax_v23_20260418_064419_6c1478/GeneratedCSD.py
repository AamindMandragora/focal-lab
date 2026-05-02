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
        d_1_done_: bool
        d_1_done_ = False
        hi0_ = maxSteps
        for d_2_step_ in range(0, hi0_):
            if not(d_1_done_):
                if insideConstrainedOut:
                    d_3_next_: _dafny.Seq
                    out0_: _dafny.Seq
                    out0_ = (d_0_helpers_).ConstrainedStep(lm, parser, prompt, currentConstrainedOut, eosToken)
                    d_3_next_ = out0_
                    if (d_3_next_) == (eosToken):
                        d_1_done_ = True
                    elif True:
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_3_next_]))
                        currentConstrainedOut = (currentConstrainedOut) + (_dafny.SeqWithoutIsStrInference([d_3_next_]))
                        cost = (cost) + (1)
                elif True:
                    d_4_next2_: _dafny.Seq
                    out1_: _dafny.Seq
                    out1_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                    d_4_next2_ = out1_
                    if (d_4_next2_) == (eosToken):
                        d_1_done_ = True
                    elif True:
                        if VerifiedDecoderAgent.default__.Contains(d_4_next2_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                            d_5_g1_: _dafny.Seq
                            d_6_i1_: bool
                            d_7_c1_: _dafny.Seq
                            out2_: _dafny.Seq
                            out3_: bool
                            out4_: _dafny.Seq
                            out2_, out3_, out4_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_5_g1_ = out2_
                            d_6_i1_ = out3_
                            d_7_c1_ = out4_
                            generated = d_5_g1_
                            insideConstrainedOut = d_6_i1_
                            currentConstrainedOut = d_7_c1_
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_4_next2_]))
                        cost = (cost) + (1)
        return generated, insideConstrainedOut, currentConstrainedOut, cost

