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
        d_1_currentConstrained_: _dafny.Seq
        d_1_currentConstrained_ = _dafny.SeqWithoutIsStrInference([])
        d_2_steps_: int
        d_2_steps_ = 0
        generated = _dafny.SeqWithoutIsStrInference([])
        with _dafny.label("0"):
            while (d_2_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if (parser).IsCompletePrefix(d_1_currentConstrained_):
                        raise _dafny.Break("0")
                    elif True:
                        d_3_oldCost_: int
                        d_3_oldCost_ = d_0_helpers_.cost
                        d_4_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).ConstrainedStep(lm, parser, prompt, d_1_currentConstrained_, eosToken)
                        d_4_next_ = out0_
                        if (d_4_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_1_currentConstrained_ = (d_1_currentConstrained_) + (_dafny.SeqWithoutIsStrInference([d_4_next_]))
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_4_next_]))
                            d_2_steps_ = (d_2_steps_) + (1)
                    pass
            pass
        cost = d_0_helpers_.cost
        cost = d_0_helpers_.cost
        return generated, cost

