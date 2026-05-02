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
    def MyCSDStrategy(lm, parser, prompt, currentPrefix, maxSteps, eosToken):
        generated: _dafny.Seq = _dafny.Seq({})
        cost: int = int(0)
        d_0_helpers_: VerifiedDecoderAgent.CSDHelpers
        nw0_ = VerifiedDecoderAgent.CSDHelpers()
        nw0_.ctor__()
        d_0_helpers_ = nw0_
        generated = currentPrefix
        (d_0_helpers_).cost = 0
        cost = 0
        d_1_suffix_: _dafny.Seq
        d_1_suffix_ = _dafny.SeqWithoutIsStrInference([])
        d_2_steps_: int
        d_2_steps_ = 0
        with _dafny.label("0"):
            while (d_2_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if (parser).IsCompletePrefix(generated):
                        raise _dafny.Break("0")
                    elif True:
                        d_3_validCount_: int
                        out0_: int
                        out0_ = (d_0_helpers_).ValidTokenCount(parser, generated)
                        d_3_validCount_ = out0_
                        if (d_3_validCount_) <= (6):
                            d_4_next_: _dafny.Seq
                            out1_: _dafny.Seq
                            out1_ = (d_0_helpers_).ConstrainedStep(lm, parser, prompt, generated, eosToken)
                            d_4_next_ = out1_
                            if (d_4_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_4_next_]))
                                d_1_suffix_ = (d_1_suffix_) + (_dafny.SeqWithoutIsStrInference([d_4_next_]))
                                d_2_steps_ = (d_2_steps_) + (1)
                        elif True:
                            d_5_next_: _dafny.Seq
                            d_6_isValid_: bool
                            out2_: _dafny.Seq
                            out3_: bool
                            out2_, out3_ = (d_0_helpers_).SoftConstrainedStep(lm, parser, prompt, generated, _dafny.BigRational('4e0'), eosToken)
                            d_5_next_ = out2_
                            d_6_isValid_ = out3_
                            if (d_5_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                if d_6_isValid_:
                                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_5_next_]))
                                    d_1_suffix_ = (d_1_suffix_) + (_dafny.SeqWithoutIsStrInference([d_5_next_]))
                                    d_2_steps_ = (d_2_steps_) + (1)
                                elif True:
                                    raise _dafny.Break("0")
                    pass
            pass
        cost = d_0_helpers_.cost
        return generated, cost

