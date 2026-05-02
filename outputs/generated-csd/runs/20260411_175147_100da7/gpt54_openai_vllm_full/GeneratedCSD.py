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
        d_3_inConstrained_: bool
        d_3_inConstrained_ = False
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if d_3_inConstrained_:
                        d_4_isComplete_: bool
                        d_4_isComplete_ = (parser).IsCompletePrefix(d_2_currentConstrained_)
                        if d_4_isComplete_:
                            d_5_next_: _dafny.Seq
                            out0_: _dafny.Seq
                            out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_5_next_ = out0_
                            if (d_5_next_) == (eosToken):
                                raise _dafny.Break("0")
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_5_next_]))
                            if (d_5_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))):
                                d_2_currentConstrained_ = _dafny.SeqWithoutIsStrInference([])
                                d_3_inConstrained_ = False
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_6_next_: _dafny.Seq
                            out1_: _dafny.Seq
                            out1_ = (d_0_helpers_).ConstrainedStep(lm, parser, prompt, d_2_currentConstrained_, eosToken)
                            d_6_next_ = out1_
                            if (d_6_next_) == (eosToken):
                                raise _dafny.Break("0")
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_6_next_]))
                            if (d_6_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))):
                                d_2_currentConstrained_ = _dafny.SeqWithoutIsStrInference([])
                                d_3_inConstrained_ = False
                            elif True:
                                d_2_currentConstrained_ = (d_2_currentConstrained_) + (_dafny.SeqWithoutIsStrInference([d_6_next_]))
                            d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_7_next_: _dafny.Seq
                        out2_: _dafny.Seq
                        out2_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_7_next_ = out2_
                        if (d_7_next_) == (eosToken):
                            raise _dafny.Break("0")
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_7_next_]))
                        if (d_7_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                            d_2_currentConstrained_ = _dafny.SeqWithoutIsStrInference([])
                            d_3_inConstrained_ = True
                        d_1_steps_ = (d_1_steps_) + (1)
                    pass
            pass
        cost = d_0_helpers_.cost
        cost = d_0_helpers_.cost
        return generated, cost

