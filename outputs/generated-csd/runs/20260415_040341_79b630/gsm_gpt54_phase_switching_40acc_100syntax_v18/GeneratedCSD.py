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
                d_1_steps_ = (d_1_steps_) + (1)
                if (d_2_next_) == (eosToken):
                    pass
                elif True:
                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_2_next_]))
                    if (d_2_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                        insideConstrainedOut = True
                        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
            elif True:
                if (parser).IsCompletePrefix(currentConstrainedOut):
                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]))
                    insideConstrainedOut = False
                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    d_1_steps_ = (d_1_steps_) + (1)
                elif True:
                    d_3_next_: _dafny.Seq
                    out1_: _dafny.Seq
                    out1_ = (d_0_helpers_).ConstrainedStep(lm, parser, prompt, currentConstrainedOut, eosToken)
                    d_3_next_ = out1_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if (d_3_next_) == (eosToken):
                        pass
                    elif True:
                        d_4_appendedGenerated_: _dafny.Seq
                        d_5_appendedInside_: bool
                        d_6_appendedCurrent_: _dafny.Seq
                        out2_: _dafny.Seq
                        out3_: bool
                        out4_: _dafny.Seq
                        out2_, out3_, out4_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_3_next_)
                        d_4_appendedGenerated_ = out2_
                        d_5_appendedInside_ = out3_
                        d_6_appendedCurrent_ = out4_
                        generated = d_4_appendedGenerated_
                        insideConstrainedOut = d_5_appendedInside_
                        currentConstrainedOut = d_6_appendedCurrent_
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

