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
        d_2_insideConstrained_: bool
        d_2_insideConstrained_ = False
        d_3_currentConstrained_: _dafny.Seq
        d_3_currentConstrained_ = _dafny.SeqWithoutIsStrInference([])
        d_4_previousToken_: _dafny.Seq
        d_4_previousToken_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(d_2_insideConstrained_):
                        d_5_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_5_next_ = out0_
                        if (d_5_next_) == (eosToken):
                            raise _dafny.Break("0")
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_5_next_]))
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_5_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                            d_2_insideConstrained_ = True
                            d_3_currentConstrained_ = _dafny.SeqWithoutIsStrInference([])
                        d_4_previousToken_ = d_5_next_
                    elif True:
                        if (parser).IsCompletePrefix(d_3_currentConstrained_):
                            d_2_insideConstrained_ = False
                        elif True:
                            d_6_next_: _dafny.Seq
                            d_7_wasConstrained_: bool
                            out1_: _dafny.Seq
                            out2_: bool
                            out1_, out2_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, prompt, d_3_currentConstrained_)
                            d_6_next_ = out1_
                            d_7_wasConstrained_ = out2_
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_6_next_]))
                            d_1_steps_ = (d_1_steps_) + (1)
                            if d_7_wasConstrained_:
                                d_3_currentConstrained_ = (d_3_currentConstrained_) + (_dafny.SeqWithoutIsStrInference([d_6_next_]))
                                if (d_6_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))):
                                    d_2_insideConstrained_ = False
                                d_4_previousToken_ = d_6_next_
                            elif True:
                                d_2_insideConstrained_ = False
                    pass
            pass
        cost = d_0_helpers_.cost
        cost = d_0_helpers_.cost
        return generated, cost

