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
        d_2_sawOpen_: bool
        d_2_sawOpen_ = False
        d_3_sawClose_: bool
        d_3_sawClose_ = False
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    (lm).GenerateLogits((prompt) + (generated))
                    if not(d_2_sawOpen_):
                        (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]), _dafny.BigRational('8e0'))
                    elif True:
                        if not(d_3_sawClose_):
                            (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]), _dafny.BigRational('8e0'))
                    d_4_next_: _dafny.Seq
                    out0_: _dafny.Seq
                    out0_ = (lm).ChooseNextTokenUnconstrained()
                    d_4_next_ = out0_
                    (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                    d_1_steps_ = (d_1_steps_) + (1)
                    if (d_4_next_) == (eosToken):
                        raise _dafny.Break("0")
                    elif True:
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_4_next_]))
                        if (d_4_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                            d_2_sawOpen_ = True
                        elif True:
                            if (d_4_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))):
                                d_3_sawClose_ = True
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

