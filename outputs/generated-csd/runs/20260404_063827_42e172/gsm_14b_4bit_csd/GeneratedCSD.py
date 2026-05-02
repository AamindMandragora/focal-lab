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
    def MyCSDStrategy(lm, parser, prompt, maxSteps, eosToken):
        generated: _dafny.Seq = _dafny.Seq({})
        cost: int = int(0)
        d_0_helpers_: VerifiedDecoderAgent.CSDHelpers
        nw0_ = VerifiedDecoderAgent.CSDHelpers()
        nw0_.ctor__()
        d_0_helpers_ = nw0_
        generated = _dafny.SeqWithoutIsStrInference([])
        d_1_steps_: int
        d_1_steps_ = 0
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if (parser).IsCompletePrefix(generated):
                        raise _dafny.Break("0")
                    d_2_count_: int
                    out0_: int
                    out0_ = (d_0_helpers_).ValidTokenCount(parser, generated)
                    d_2_count_ = out0_
                    d_3_next_: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                    if (d_2_count_) > (8):
                        out1_: _dafny.Seq
                        out1_ = (d_0_helpers_).TemperatureConstrainedStep(lm, parser, prompt, generated, _dafny.BigRational('15e-1'), eosToken)
                        d_3_next_ = out1_
                    elif True:
                        out2_: _dafny.Seq
                        out2_ = (d_0_helpers_).TemperatureConstrainedStep(lm, parser, prompt, generated, _dafny.BigRational('3e-1'), eosToken)
                        d_3_next_ = out2_
                    if (d_3_next_) == (eosToken):
                        raise _dafny.Break("0")
                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_3_next_]))
                    d_1_steps_ = (d_1_steps_) + (1)
                    pass
            pass
        cost = d_0_helpers_.cost
        cost = d_0_helpers_.cost
        return generated, cost

