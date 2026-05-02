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
        d_3_seeded_: bool
        d_3_seeded_ = False
        d_4_seedIndex_: int
        d_4_seedIndex_ = 0
        generated = _dafny.SeqWithoutIsStrInference([])
        with _dafny.label("0"):
            while (d_2_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(d_3_seeded_):
                        if (d_4_seedIndex_) == (0):
                            if (parser).IsValidPrefix((d_1_currentConstrained_) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]))):
                                d_1_currentConstrained_ = (d_1_currentConstrained_) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]))
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]))
                                d_2_steps_ = (d_2_steps_) + (1)
                            d_4_seedIndex_ = 1
                        elif (d_4_seedIndex_) == (1):
                            if (parser).IsValidPrefix((d_1_currentConstrained_) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "1"))]))):
                                d_1_currentConstrained_ = (d_1_currentConstrained_) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "1"))]))
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "1"))]))
                                d_2_steps_ = (d_2_steps_) + (1)
                            d_4_seedIndex_ = 2
                        elif (d_4_seedIndex_) == (2):
                            if (parser).IsValidPrefix((d_1_currentConstrained_) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "+"))]))):
                                d_1_currentConstrained_ = (d_1_currentConstrained_) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "+"))]))
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "+"))]))
                                d_2_steps_ = (d_2_steps_) + (1)
                            d_4_seedIndex_ = 3
                        elif (d_4_seedIndex_) == (3):
                            if (parser).IsValidPrefix((d_1_currentConstrained_) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "1"))]))):
                                d_1_currentConstrained_ = (d_1_currentConstrained_) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "1"))]))
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "1"))]))
                                d_2_steps_ = (d_2_steps_) + (1)
                            d_4_seedIndex_ = 4
                        elif (d_4_seedIndex_) == (4):
                            if (parser).IsValidPrefix((d_1_currentConstrained_) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "="))]))):
                                d_1_currentConstrained_ = (d_1_currentConstrained_) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "="))]))
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "="))]))
                                d_2_steps_ = (d_2_steps_) + (1)
                            d_4_seedIndex_ = 5
                        elif (d_4_seedIndex_) == (5):
                            if (parser).IsValidPrefix((d_1_currentConstrained_) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "2"))]))):
                                d_1_currentConstrained_ = (d_1_currentConstrained_) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "2"))]))
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "2"))]))
                                d_2_steps_ = (d_2_steps_) + (1)
                            d_4_seedIndex_ = 6
                        elif True:
                            if (parser).IsValidPrefix((d_1_currentConstrained_) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]))):
                                d_1_currentConstrained_ = (d_1_currentConstrained_) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]))
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]))
                                d_2_steps_ = (d_2_steps_) + (1)
                            d_4_seedIndex_ = 7
                            d_3_seeded_ = True
                    elif True:
                        if (parser).IsCompletePrefix(d_1_currentConstrained_):
                            raise _dafny.Break("0")
                        elif True:
                            d_5_oldCost_: int
                            d_5_oldCost_ = d_0_helpers_.cost
                            d_6_next_: _dafny.Seq
                            out0_: _dafny.Seq
                            out0_ = (d_0_helpers_).ConstrainedStep(lm, parser, prompt, d_1_currentConstrained_, eosToken)
                            d_6_next_ = out0_
                            if (d_6_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_1_currentConstrained_ = (d_1_currentConstrained_) + (_dafny.SeqWithoutIsStrInference([d_6_next_]))
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_6_next_]))
                                d_2_steps_ = (d_2_steps_) + (1)
                    pass
            pass
        cost = d_0_helpers_.cost
        cost = d_0_helpers_.cost
        return generated, cost

