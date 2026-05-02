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
        d_2_inWindow_: bool
        d_2_inWindow_ = False
        d_3_currentConstrained_: _dafny.Seq
        d_3_currentConstrained_ = _dafny.SeqWithoutIsStrInference([])
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(d_2_inWindow_):
                        d_4_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_4_next_ = out0_
                        if (d_4_next_) == (eosToken):
                            raise _dafny.Break("0")
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_4_next_]))
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_4_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                            d_2_inWindow_ = True
                            d_3_currentConstrained_ = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        if (parser).IsCompletePrefix(d_3_currentConstrained_):
                            d_2_inWindow_ = False
                        elif True:
                            d_5_next_: _dafny.Seq
                            d_6_wasConstrained_: bool
                            out1_: _dafny.Seq
                            out2_: bool
                            out1_, out2_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, prompt, d_3_currentConstrained_)
                            d_5_next_ = out1_
                            d_6_wasConstrained_ = out2_
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_5_next_]))
                            d_1_steps_ = (d_1_steps_) + (1)
                            if d_6_wasConstrained_:
                                d_3_currentConstrained_ = (d_3_currentConstrained_) + (_dafny.SeqWithoutIsStrInference([d_5_next_]))
                                if (d_5_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))):
                                    d_2_inWindow_ = False
                            elif True:
                                d_2_inWindow_ = False
                    pass
            pass
        cost = d_0_helpers_.cost
        cost = d_0_helpers_.cost
        return generated, cost

