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
                    if not(insideConstrainedOut):
                        if not(d_2_openedOnce_):
                            d_3_g0_: _dafny.Seq
                            d_4_i0_: bool
                            d_5_c0_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_3_g0_ = out0_
                            d_4_i0_ = out1_
                            d_5_c0_ = out2_
                            generated = d_3_g0_
                            insideConstrainedOut = d_4_i0_
                            currentConstrainedOut = d_5_c0_
                            d_2_openedOnce_ = True
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            (lm).GenerateLogits((prompt) + (generated))
                            (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([eosToken]), _dafny.BigRational('1e2'))
                            d_6_nextOutside_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (lm).ChooseNextToken()
                            d_6_nextOutside_ = out3_
                            (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_6_nextOutside_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_6_nextOutside_]))
                    elif True:
                        d_7_completeNow_: bool
                        d_7_completeNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_7_completeNow_:
                            (lm).GenerateLogits((prompt) + (generated))
                            (d_0_helpers_).PenalizeTokenLogits(lm, _dafny.SeqWithoutIsStrInference([eosToken]), _dafny.BigRational('1e2'))
                            (d_0_helpers_).PenalizeTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]), _dafny.BigRational('8e0'))
                            d_8_topToken_: _dafny.Seq
                            out4_: _dafny.Seq
                            out4_ = (d_0_helpers_).GetHighestLogitToken(lm)
                            d_8_topToken_ = out4_
                            d_9_topValid_: bool
                            out5_: bool
                            out5_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_8_topToken_)
                            d_9_topValid_ = out5_
                            if d_9_topValid_:
                                d_10_nextInside0_: _dafny.Seq
                                out6_: _dafny.Seq
                                out6_ = (d_0_helpers_).ConstrainedStep(lm, parser, prompt, currentConstrainedOut, eosToken)
                                d_10_nextInside0_ = out6_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_10_nextInside0_) == (eosToken):
                                    d_11_g1_: _dafny.Seq
                                    d_12_i1_: bool
                                    d_13_c1_: _dafny.Seq
                                    out7_: _dafny.Seq
                                    out8_: bool
                                    out9_: _dafny.Seq
                                    out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_11_g1_ = out7_
                                    d_12_i1_ = out8_
                                    d_13_c1_ = out9_
                                    generated = d_11_g1_
                                    insideConstrainedOut = d_12_i1_
                                    currentConstrainedOut = d_13_c1_
                                elif True:
                                    d_14_g2_: _dafny.Seq
                                    d_15_i2_: bool
                                    d_16_c2_: _dafny.Seq
                                    out10_: _dafny.Seq
                                    out11_: bool
                                    out12_: _dafny.Seq
                                    out10_, out11_, out12_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_14_g2_ = out10_
                                    d_15_i2_ = out11_
                                    d_16_c2_ = out12_
                                    generated = d_14_g2_
                                    insideConstrainedOut = d_15_i2_
                                    currentConstrainedOut = d_16_c2_
                            elif True:
                                d_17_g3_: _dafny.Seq
                                d_18_i3_: bool
                                d_19_c3_: _dafny.Seq
                                out13_: _dafny.Seq
                                out14_: bool
                                out15_: _dafny.Seq
                                out13_, out14_, out15_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_17_g3_ = out13_
                                d_18_i3_ = out14_
                                d_19_c3_ = out15_
                                generated = d_17_g3_
                                insideConstrainedOut = d_18_i3_
                                currentConstrainedOut = d_19_c3_
                                d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_20_nextInside_: _dafny.Seq
                            out16_: _dafny.Seq
                            out16_ = (d_0_helpers_).ConstrainedStep(lm, parser, prompt, currentConstrainedOut, eosToken)
                            d_20_nextInside_ = out16_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_20_nextInside_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_21_g4_: _dafny.Seq
                                d_22_i4_: bool
                                d_23_c4_: _dafny.Seq
                                out17_: _dafny.Seq
                                out18_: bool
                                out19_: _dafny.Seq
                                out17_, out18_, out19_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_20_nextInside_)
                                d_21_g4_ = out17_
                                d_22_i4_ = out18_
                                d_23_c4_ = out19_
                                generated = d_21_g4_
                                insideConstrainedOut = d_22_i4_
                                currentConstrainedOut = d_23_c4_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

