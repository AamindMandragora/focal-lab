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
        (d_0_helpers_).cost = 0
        d_1_steps_: int
        d_1_steps_ = 0
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_2_remaining_: int
                        d_2_remaining_ = (maxSteps) - (d_1_steps_)
                        if (d_2_remaining_) >= (3):
                            (lm).GenerateLogits((prompt) + (generated))
                            (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]), _dafny.BigRational('8e0'))
                            (d_0_helpers_).PenalizeTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]), _dafny.BigRational('1e2'))
                            d_3_topOutside_: _dafny.Seq
                            out0_: _dafny.Seq
                            out0_ = (d_0_helpers_).GetHighestLogitToken(lm)
                            d_3_topOutside_ = out0_
                            if (d_3_topOutside_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                d_4_gOpen_: _dafny.Seq
                                d_5_iOpen_: bool
                                d_6_cOpen_: _dafny.Seq
                                out1_: _dafny.Seq
                                out2_: bool
                                out3_: _dafny.Seq
                                out1_, out2_, out3_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                d_4_gOpen_ = out1_
                                d_5_iOpen_ = out2_
                                d_6_cOpen_ = out3_
                                generated = d_4_gOpen_
                                insideConstrainedOut = d_5_iOpen_
                                currentConstrainedOut = d_6_cOpen_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                d_7_nextOutside_: _dafny.Seq
                                out4_: _dafny.Seq
                                out4_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                                d_7_nextOutside_ = out4_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_7_nextOutside_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_7_nextOutside_]))
                        elif True:
                            d_8_nextOutside2_: _dafny.Seq
                            out5_: _dafny.Seq
                            out5_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_8_nextOutside2_ = out5_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_8_nextOutside2_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_8_nextOutside2_]))
                    elif True:
                        d_9_complete_: bool
                        d_9_complete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        d_10_clen_: int
                        d_10_clen_ = len(currentConstrainedOut)
                        if (d_9_complete_) and ((d_10_clen_) >= (3)):
                            d_11_gClose_: _dafny.Seq
                            d_12_iClose_: bool
                            d_13_cClose_: _dafny.Seq
                            out6_: _dafny.Seq
                            out7_: bool
                            out8_: _dafny.Seq
                            out6_, out7_, out8_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_11_gClose_ = out6_
                            d_12_iClose_ = out7_
                            d_13_cClose_ = out8_
                            generated = d_11_gClose_
                            insideConstrainedOut = d_12_iClose_
                            currentConstrainedOut = d_13_cClose_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            if ((d_1_steps_) + (1)) >= (maxSteps):
                                if d_9_complete_:
                                    d_14_gClose2_: _dafny.Seq
                                    d_15_iClose2_: bool
                                    d_16_cClose2_: _dafny.Seq
                                    out9_: _dafny.Seq
                                    out10_: bool
                                    out11_: _dafny.Seq
                                    out9_, out10_, out11_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_14_gClose2_ = out9_
                                    d_15_iClose2_ = out10_
                                    d_16_cClose2_ = out11_
                                    generated = d_14_gClose2_
                                    insideConstrainedOut = d_15_iClose2_
                                    currentConstrainedOut = d_16_cClose2_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                elif True:
                                    raise _dafny.Break("0")
                            elif True:
                                (lm).GenerateLogits((prompt) + (generated))
                                if d_9_complete_:
                                    (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]), _dafny.BigRational('12e0'))
                                    d_17_nextInsideComplete_: _dafny.Seq
                                    out12_: _dafny.Seq
                                    out12_ = (d_0_helpers_).ConstrainedStep(lm, parser, prompt, currentConstrainedOut, eosToken)
                                    d_17_nextInsideComplete_ = out12_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    if (d_17_nextInsideComplete_) == (eosToken):
                                        raise _dafny.Break("0")
                                    elif True:
                                        raise _dafny.Break("0")
                                elif True:
                                    (d_0_helpers_).PenalizeTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]), _dafny.BigRational('1e2'))
                                    d_18_nextInside_: _dafny.Seq
                                    out13_: _dafny.Seq
                                    out13_ = (d_0_helpers_).ConstrainedStep(lm, parser, prompt, currentConstrainedOut, eosToken)
                                    d_18_nextInside_ = out13_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    if (d_18_nextInside_) == (eosToken):
                                        raise _dafny.Break("0")
                                    elif True:
                                        d_19_gApp_: _dafny.Seq
                                        d_20_iApp_: bool
                                        d_21_cApp_: _dafny.Seq
                                        out14_: _dafny.Seq
                                        out15_: bool
                                        out16_: _dafny.Seq
                                        out14_, out15_, out16_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_18_nextInside_)
                                        d_19_gApp_ = out14_
                                        d_20_iApp_ = out15_
                                        d_21_cApp_ = out16_
                                        generated = d_19_gApp_
                                        insideConstrainedOut = d_20_iApp_
                                        currentConstrainedOut = d_21_cApp_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

