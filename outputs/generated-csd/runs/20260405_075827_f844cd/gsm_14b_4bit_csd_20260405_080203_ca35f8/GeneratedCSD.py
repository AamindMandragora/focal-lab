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
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(d_2_insideConstrained_):
                        d_4_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_4_next_ = out0_
                        if (d_4_next_) == (eosToken):
                            raise _dafny.Break("0")
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_4_next_]))
                        d_1_steps_ = (d_1_steps_) + (1)
                        if VerifiedDecoderAgent.default__.Contains(d_4_next_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                            d_2_insideConstrained_ = True
                            d_3_currentConstrained_ = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        if (parser).IsCompletePrefix(d_3_currentConstrained_):
                            d_2_insideConstrained_ = False
                        elif True:
                            d_5_count_: int
                            out1_: int
                            out1_ = (d_0_helpers_).ValidTokenCount(parser, d_3_currentConstrained_)
                            d_5_count_ = out1_
                            d_6_next_: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                            if (d_5_count_) > (10):
                                out2_: _dafny.Seq
                                out2_ = (d_0_helpers_).TemperatureConstrainedStep(lm, parser, prompt, d_3_currentConstrained_, _dafny.BigRational('7e-1'), eosToken)
                                d_6_next_ = out2_
                            elif True:
                                out3_: _dafny.Seq
                                out3_ = (d_0_helpers_).TemperatureConstrainedStep(lm, parser, prompt, d_3_currentConstrained_, _dafny.BigRational('3e-1'), eosToken)
                                d_6_next_ = out3_
                            if (d_6_next_) == (eosToken):
                                raise _dafny.Break("0")
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_6_next_]))
                            d_3_currentConstrained_ = (d_3_currentConstrained_) + (_dafny.SeqWithoutIsStrInference([d_6_next_]))
                            d_1_steps_ = (d_1_steps_) + (1)
                            if VerifiedDecoderAgent.default__.Contains(d_6_next_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))):
                                d_2_insideConstrained_ = False
                    pass
            pass
        cost = d_0_helpers_.cost
        cost = d_0_helpers_.cost
        return generated, cost

