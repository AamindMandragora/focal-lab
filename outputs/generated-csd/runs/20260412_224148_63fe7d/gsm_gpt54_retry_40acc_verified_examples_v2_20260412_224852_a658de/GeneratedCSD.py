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
        if True:
            generated = currentPrefix
            (d_0_helpers_).cost = 0
            cost = 0
            d_1_suffix_: _dafny.Seq
            d_1_suffix_ = _dafny.SeqWithoutIsStrInference([])
            d_2_steps_: int
            d_2_steps_ = 0
            d_3_inDelim_: bool
            d_3_inDelim_ = False
            d_4_contentInDelim_: bool
            d_4_contentInDelim_ = False
            with _dafny.label("0"):
                while (d_2_steps_) < (maxSteps):
                    with _dafny.c_label("0"):
                        if (parser).IsCompletePrefix(generated):
                            raise _dafny.Break("0")
                        elif True:
                            d_5_canOpen_: bool
                            out0_: bool
                            out0_ = (d_0_helpers_).IsTokenValidNext(parser, generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))
                            d_5_canOpen_ = out0_
                            d_6_canClose_: bool
                            out1_: bool
                            out1_ = (d_0_helpers_).IsTokenValidNext(parser, generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")))
                            d_6_canClose_ = out1_
                            if (not(d_3_inDelim_)) and (d_5_canOpen_):
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]))
                                d_1_suffix_ = (d_1_suffix_) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]))
                                d_3_inDelim_ = True
                                d_4_contentInDelim_ = False
                                d_2_steps_ = (d_2_steps_) + (1)
                            elif True:
                                if ((d_3_inDelim_) and (d_6_canClose_)) and (d_4_contentInDelim_):
                                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]))
                                    d_1_suffix_ = (d_1_suffix_) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]))
                                    d_3_inDelim_ = False
                                    d_4_contentInDelim_ = False
                                    d_2_steps_ = (d_2_steps_) + (1)
                                elif True:
                                    if d_3_inDelim_:
                                        if (((d_2_steps_) + (1)) == (maxSteps)) and (d_6_canClose_):
                                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]))
                                            d_1_suffix_ = (d_1_suffix_) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]))
                                            d_3_inDelim_ = False
                                            d_4_contentInDelim_ = False
                                            d_2_steps_ = (d_2_steps_) + (1)
                                        elif True:
                                            d_7_next_: _dafny.Seq
                                            out2_: _dafny.Seq
                                            out2_ = (d_0_helpers_).ConstrainedStep(lm, parser, prompt, generated, eosToken)
                                            d_7_next_ = out2_
                                            if (d_7_next_) == (eosToken):
                                                raise _dafny.Break("0")
                                            elif True:
                                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_7_next_]))
                                                d_1_suffix_ = (d_1_suffix_) + (_dafny.SeqWithoutIsStrInference([d_7_next_]))
                                                if (d_7_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                                    d_3_inDelim_ = True
                                                    d_4_contentInDelim_ = False
                                                elif True:
                                                    if (d_7_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))):
                                                        d_3_inDelim_ = False
                                                        d_4_contentInDelim_ = False
                                                    elif True:
                                                        d_4_contentInDelim_ = True
                                                d_2_steps_ = (d_2_steps_) + (1)
                                    elif True:
                                        d_8_next2_: _dafny.Seq
                                        out3_: _dafny.Seq
                                        out3_ = (d_0_helpers_).ConstrainedStep(lm, parser, prompt, generated, eosToken)
                                        d_8_next2_ = out3_
                                        if (d_8_next2_) == (eosToken):
                                            raise _dafny.Break("0")
                                        elif True:
                                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_8_next2_]))
                                            d_1_suffix_ = (d_1_suffix_) + (_dafny.SeqWithoutIsStrInference([d_8_next2_]))
                                            if (d_8_next2_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                                d_3_inDelim_ = True
                                                d_4_contentInDelim_ = False
                                            elif True:
                                                if (d_8_next2_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))):
                                                    d_3_inDelim_ = False
                                                    d_4_contentInDelim_ = False
                                            d_2_steps_ = (d_2_steps_) + (1)
                        pass
                pass
            cost = d_0_helpers_.cost
        return generated, cost

