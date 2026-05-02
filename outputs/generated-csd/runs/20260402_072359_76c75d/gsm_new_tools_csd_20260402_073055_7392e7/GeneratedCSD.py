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
        d_2_currentConstrained_: _dafny.Seq
        d_2_currentConstrained_ = _dafny.SeqWithoutIsStrInference([])
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    d_3_next_: _dafny.Seq
                    out0_: _dafny.Seq
                    out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                    d_3_next_ = out0_
                    if (d_3_next_) == (eosToken):
                        raise _dafny.Break("0")
                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_3_next_]))
                    d_1_steps_ = (d_1_steps_) + (1)
                    pass
            pass
        cost = d_0_helpers_.cost
        cost = d_0_helpers_.cost
        return generated, cost

