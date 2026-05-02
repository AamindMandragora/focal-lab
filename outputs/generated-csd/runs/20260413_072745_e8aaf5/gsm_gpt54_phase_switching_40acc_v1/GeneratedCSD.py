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
        d_1_suffix_: _dafny.Seq
        d_1_suffix_ = _dafny.SeqWithoutIsStrInference([])
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_done_: bool
        d_3_done_ = False
        while ((d_2_steps_) < (maxSteps)) and (not(d_3_done_)):
            if not(insideConstrainedOut):
                d_4_next_: _dafny.Seq
                out0_: _dafny.Seq
                out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                d_4_next_ = out0_
                d_2_steps_ = (d_2_steps_) + (1)
                if (d_4_next_) == (eosToken):
                    d_3_done_ = True
                elif True:
                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_4_next_]))
                    d_1_suffix_ = (d_1_suffix_) + (_dafny.SeqWithoutIsStrInference([d_4_next_]))
                    if (d_4_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                        insideConstrainedOut = True
                        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
            elif True:
                d_5_isComplete_: bool
                d_5_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                if d_5_isComplete_:
                    d_2_steps_ = (d_2_steps_) + (1)
                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]))
                    d_1_suffix_ = (d_1_suffix_) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]))
                    insideConstrainedOut = False
                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                elif True:
                    d_6_next_: _dafny.Seq
                    out1_: _dafny.Seq
                    out1_ = (d_0_helpers_).ConstrainedStep(lm, parser, prompt, currentConstrainedOut, eosToken)
                    d_6_next_ = out1_
                    d_2_steps_ = (d_2_steps_) + (1)
                    if (d_6_next_) == (eosToken):
                        d_3_done_ = True
                    elif True:
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_6_next_]))
                        d_1_suffix_ = (d_1_suffix_) + (_dafny.SeqWithoutIsStrInference([d_6_next_]))
                        if (d_6_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))):
                            insideConstrainedOut = False
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                        elif True:
                            currentConstrainedOut = (currentConstrainedOut) + (_dafny.SeqWithoutIsStrInference([d_6_next_]))
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

