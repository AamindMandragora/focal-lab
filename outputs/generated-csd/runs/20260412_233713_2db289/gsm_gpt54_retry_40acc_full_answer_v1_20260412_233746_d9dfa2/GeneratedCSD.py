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
        if ((d_2_steps_) < (maxSteps)) and (not((parser).IsCompletePrefix(generated))):
            d_3_first_: _dafny.Seq
            out0_: _dafny.Seq
            out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
            d_3_first_ = out0_
            d_2_steps_ = (d_2_steps_) + (1)
            if (d_3_first_) == (eosToken):
                pass
            elif True:
                d_4_firstValid_: bool
                out1_: bool
                out1_ = (d_0_helpers_).IsTokenValidNext(parser, generated, d_3_first_)
                d_4_firstValid_ = out1_
                if d_4_firstValid_:
                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_3_first_]))
                    d_1_suffix_ = (d_1_suffix_) + (_dafny.SeqWithoutIsStrInference([d_3_first_]))
                elif True:
                    if ((d_2_steps_) < (maxSteps)) and (not((parser).IsCompletePrefix(generated))):
                        d_5_forced_: _dafny.Seq
                        out2_: _dafny.Seq
                        out2_ = (d_0_helpers_).ConstrainedStep(lm, parser, prompt, generated, eosToken)
                        d_5_forced_ = out2_
                        d_2_steps_ = (d_2_steps_) + (1)
                        if (d_5_forced_) == (eosToken):
                            pass
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_5_forced_]))
                            d_1_suffix_ = (d_1_suffix_) + (_dafny.SeqWithoutIsStrInference([d_5_forced_]))
        with _dafny.label("0"):
            while (d_2_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if (parser).IsCompletePrefix(generated):
                        raise _dafny.Break("0")
                    elif True:
                        d_6_next_: _dafny.Seq
                        out3_: _dafny.Seq
                        out3_ = (d_0_helpers_).ConstrainedStep(lm, parser, prompt, generated, eosToken)
                        d_6_next_ = out3_
                        d_2_steps_ = (d_2_steps_) + (1)
                        if (d_6_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_6_next_]))
                            d_1_suffix_ = (d_1_suffix_) + (_dafny.SeqWithoutIsStrInference([d_6_next_]))
                    pass
            pass
        cost = d_0_helpers_.cost
        return generated, cost

