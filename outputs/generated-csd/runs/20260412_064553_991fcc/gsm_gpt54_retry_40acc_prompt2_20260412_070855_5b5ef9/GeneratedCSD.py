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
        if True:
            d_1_helpers_: VerifiedDecoderAgent.CSDHelpers
            nw1_ = VerifiedDecoderAgent.CSDHelpers()
            nw1_.ctor__()
            d_1_helpers_ = nw1_
            generated = _dafny.SeqWithoutIsStrInference([])
            cost = d_1_helpers_.cost
        cost = d_0_helpers_.cost
        return generated, cost

