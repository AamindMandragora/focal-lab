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
        d_3_delimOpen_: _dafny.Seq
        d_3_delimOpen_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))])
        d_4_delimClose_: _dafny.Seq
        d_4_delimClose_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))])
        d_5_balance_: int
        d_5_balance_ = 0
        d_6_i_: int
        d_6_i_ = 0
        while (d_6_i_) < (len(generated)):
            if ((generated)[d_6_i_]) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                d_5_balance_ = (d_5_balance_) + (1)
            elif True:
                if ((generated)[d_6_i_]) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))):
                    if (d_5_balance_) > (0):
                        d_5_balance_ = (d_5_balance_) - (1)
            d_6_i_ = (d_6_i_) + (1)
        with _dafny.label("0"):
            while (d_2_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if (parser).IsCompletePrefix(generated):
                        raise _dafny.Break("0")
                    elif True:
                        d_7_validCount_: int
                        out0_: int
                        out0_ = (d_0_helpers_).ValidTokenCount(parser, generated)
                        d_7_validCount_ = out0_
                        if (d_7_validCount_) <= (2):
                            d_8_next_: _dafny.Seq
                            out1_: _dafny.Seq
                            out1_ = (d_0_helpers_).ConstrainedStep(lm, parser, prompt, generated, eosToken)
                            d_8_next_ = out1_
                            if (d_8_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_8_next_]))
                                d_1_suffix_ = (d_1_suffix_) + (_dafny.SeqWithoutIsStrInference([d_8_next_]))
                                if (d_8_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    d_5_balance_ = (d_5_balance_) + (1)
                                elif True:
                                    if ((d_8_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")))) and ((d_5_balance_) > (0)):
                                        d_5_balance_ = (d_5_balance_) - (1)
                                d_2_steps_ = (d_2_steps_) + (1)
                        elif True:
                            if (d_5_balance_) > (0):
                                d_9_next_: _dafny.Seq
                                out2_: _dafny.Seq
                                out2_ = (d_0_helpers_).BoostedConstrainedStep(lm, parser, prompt, generated, d_4_delimClose_, _dafny.BigRational('1e2'), eosToken)
                                d_9_next_ = out2_
                                if (d_9_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_9_next_]))
                                    d_1_suffix_ = (d_1_suffix_) + (_dafny.SeqWithoutIsStrInference([d_9_next_]))
                                    if (d_9_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                        d_5_balance_ = (d_5_balance_) + (1)
                                    elif True:
                                        if ((d_9_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")))) and ((d_5_balance_) > (0)):
                                            d_5_balance_ = (d_5_balance_) - (1)
                                    d_2_steps_ = (d_2_steps_) + (1)
                            elif True:
                                d_10_next_: _dafny.Seq
                                out3_: _dafny.Seq
                                out3_ = (d_0_helpers_).BoostedConstrainedStep(lm, parser, prompt, generated, d_3_delimOpen_, _dafny.BigRational('1e2'), eosToken)
                                d_10_next_ = out3_
                                if (d_10_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_10_next_]))
                                    d_1_suffix_ = (d_1_suffix_) + (_dafny.SeqWithoutIsStrInference([d_10_next_]))
                                    if (d_10_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                        d_5_balance_ = (d_5_balance_) + (1)
                                    elif True:
                                        if ((d_10_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")))) and ((d_5_balance_) > (0)):
                                            d_5_balance_ = (d_5_balance_) - (1)
                                    d_2_steps_ = (d_2_steps_) + (1)
                    pass
            pass
        cost = d_0_helpers_.cost
        return generated, cost

