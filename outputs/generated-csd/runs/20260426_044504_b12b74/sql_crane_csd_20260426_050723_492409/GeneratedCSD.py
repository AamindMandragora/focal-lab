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
    def MyCSDStrategy(lm, parser, prompt, generatedPrefix, insideConstrained, currentConstrained, maxSteps, eosToken):
        generated: _dafny.Seq = _dafny.Seq({})
        insideConstrainedOut: bool = False
        currentConstrainedOut: _dafny.Seq = _dafny.Seq({})
        cost: int = int(0)
        d_0_helpers_: VerifiedDecoderAgent.CSDHelpers
        nw0_ = VerifiedDecoderAgent.CSDHelpers()
        nw0_.ctor__()
        d_0_helpers_ = nw0_
        generated = generatedPrefix
        insideConstrainedOut = insideConstrained
        currentConstrainedOut = currentConstrained
        cost = 0
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_openedOnce_: bool
        d_2_openedOnce_ = insideConstrained
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if insideConstrainedOut:
                        d_3_completeNow_: bool
                        d_3_completeNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_3_completeNow_:
                            d_4_g0_: _dafny.Seq
                            d_5_i0_: bool
                            d_6_c0_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_4_g0_ = out0_
                            d_5_i0_ = out1_
                            d_6_c0_ = out2_
                            generated = d_4_g0_
                            insideConstrainedOut = d_5_i0_
                            currentConstrainedOut = d_6_c0_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_7_nextInside_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).ConstrainedStep(lm, parser, prompt, currentConstrainedOut, eosToken)
                            d_7_nextInside_ = out3_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_7_nextInside_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_8_g1_: _dafny.Seq
                                d_9_i1_: bool
                                d_10_c1_: _dafny.Seq
                                out4_: _dafny.Seq
                                out5_: bool
                                out6_: _dafny.Seq
                                out4_, out5_, out6_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_7_nextInside_)
                                d_8_g1_ = out4_
                                d_9_i1_ = out5_
                                d_10_c1_ = out6_
                                generated = d_8_g1_
                                insideConstrainedOut = d_9_i1_
                                currentConstrainedOut = d_10_c1_
                    elif True:
                        if not(d_2_openedOnce_):
                            d_11_g2_: _dafny.Seq
                            d_12_i2_: bool
                            d_13_c2_: _dafny.Seq
                            out7_: _dafny.Seq
                            out8_: bool
                            out9_: _dafny.Seq
                            out7_, out8_, out9_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_11_g2_ = out7_
                            d_12_i2_ = out8_
                            d_13_c2_ = out9_
                            generated = d_11_g2_
                            insideConstrainedOut = d_12_i2_
                            currentConstrainedOut = d_13_c2_
                            d_2_openedOnce_ = True
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            (lm).GenerateLogits((prompt) + (generated))
                            (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([eosToken]), _dafny.BigRational('1e2'))
                            d_14_nextOutside_: _dafny.Seq
                            out10_: _dafny.Seq
                            out10_ = (lm).ChooseNextToken()
                            d_14_nextOutside_ = out10_
                            (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_14_nextOutside_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_14_nextOutside_]))
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

