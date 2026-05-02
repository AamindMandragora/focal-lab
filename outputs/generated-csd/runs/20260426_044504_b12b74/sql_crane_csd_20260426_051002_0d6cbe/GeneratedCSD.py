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
                        d_4_validCountNow_: int
                        out0_: int
                        out0_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                        d_4_validCountNow_ = out0_
                        if d_3_completeNow_:
                            d_5_gClose_: _dafny.Seq
                            d_6_iClose_: bool
                            d_7_cClose_: _dafny.Seq
                            out1_: _dafny.Seq
                            out2_: bool
                            out3_: _dafny.Seq
                            out1_, out2_, out3_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_5_gClose_ = out1_
                            d_6_iClose_ = out2_
                            d_7_cClose_ = out3_
                            generated = d_5_gClose_
                            insideConstrainedOut = d_6_iClose_
                            currentConstrainedOut = d_7_cClose_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            if (d_4_validCountNow_) <= (4):
                                (lm).GenerateLogits((prompt) + (generated))
                                d_8_candTight_: _dafny.Seq
                                out4_: _dafny.Seq
                                out4_ = (d_0_helpers_).TopValidCandidates(lm, parser, prompt, currentConstrainedOut, 4, eosToken)
                                d_8_candTight_ = out4_
                                (d_0_helpers_).BoostTokenLogits(lm, d_8_candTight_, _dafny.BigRational('2e1'))
                            elif True:
                                d_9_narrow_: bool
                                out5_: bool
                                out5_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 6)
                                d_9_narrow_ = out5_
                                if d_9_narrow_:
                                    (lm).GenerateLogits((prompt) + (generated))
                                    d_10_candNarrow_: _dafny.Seq
                                    out6_: _dafny.Seq
                                    out6_ = (d_0_helpers_).TopValidCandidates(lm, parser, prompt, currentConstrainedOut, 8, eosToken)
                                    d_10_candNarrow_ = out6_
                                    (d_0_helpers_).BoostTokenLogits(lm, d_10_candNarrow_, _dafny.BigRational('1e1'))
                                elif True:
                                    (lm).GenerateLogits((prompt) + (generated))
                                    d_11_candWide_: _dafny.Seq
                                    out7_: _dafny.Seq
                                    out7_ = (d_0_helpers_).TopValidCandidates(lm, parser, prompt, currentConstrainedOut, 12, eosToken)
                                    d_11_candWide_ = out7_
                                    (d_0_helpers_).BoostTokenLogits(lm, d_11_candWide_, _dafny.BigRational('6e0'))
                            d_12_nextInside_: _dafny.Seq
                            out8_: _dafny.Seq
                            out8_ = (d_0_helpers_).ConstrainedStep(lm, parser, prompt, currentConstrainedOut, eosToken)
                            d_12_nextInside_ = out8_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_12_nextInside_) == (eosToken):
                                d_13_fallback_: _dafny.Seq
                                out9_: _dafny.Seq
                                out9_ = (d_0_helpers_).TopValidCandidates(lm, parser, prompt, currentConstrainedOut, 1, eosToken)
                                d_13_fallback_ = out9_
                                if (len(d_13_fallback_)) > (0):
                                    d_14_chosen_: _dafny.Seq
                                    d_14_chosen_ = (d_13_fallback_)[0]
                                    d_15_chosenValid_: bool
                                    out10_: bool
                                    out10_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_14_chosen_)
                                    d_15_chosenValid_ = out10_
                                    if d_15_chosenValid_:
                                        d_16_gAppendFallback_: _dafny.Seq
                                        d_17_iAppendFallback_: bool
                                        d_18_cAppendFallback_: _dafny.Seq
                                        out11_: _dafny.Seq
                                        out12_: bool
                                        out13_: _dafny.Seq
                                        out11_, out12_, out13_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_14_chosen_)
                                        d_16_gAppendFallback_ = out11_
                                        d_17_iAppendFallback_ = out12_
                                        d_18_cAppendFallback_ = out13_
                                        generated = d_16_gAppendFallback_
                                        insideConstrainedOut = d_17_iAppendFallback_
                                        currentConstrainedOut = d_18_cAppendFallback_
                                    elif True:
                                        raise _dafny.Break("0")
                                elif True:
                                    raise _dafny.Break("0")
                            elif True:
                                d_19_nextValid_: bool
                                out14_: bool
                                out14_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_12_nextInside_)
                                d_19_nextValid_ = out14_
                                if d_19_nextValid_:
                                    d_20_gAppend_: _dafny.Seq
                                    d_21_iAppend_: bool
                                    d_22_cAppend_: _dafny.Seq
                                    out15_: _dafny.Seq
                                    out16_: bool
                                    out17_: _dafny.Seq
                                    out15_, out16_, out17_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_12_nextInside_)
                                    d_20_gAppend_ = out15_
                                    d_21_iAppend_ = out16_
                                    d_22_cAppend_ = out17_
                                    generated = d_20_gAppend_
                                    insideConstrainedOut = d_21_iAppend_
                                    currentConstrainedOut = d_22_cAppend_
                                elif True:
                                    d_23_fallback2_: _dafny.Seq
                                    out18_: _dafny.Seq
                                    out18_ = (d_0_helpers_).TopValidCandidates(lm, parser, prompt, currentConstrainedOut, 1, eosToken)
                                    d_23_fallback2_ = out18_
                                    if (len(d_23_fallback2_)) > (0):
                                        d_24_chosen2_: _dafny.Seq
                                        d_24_chosen2_ = (d_23_fallback2_)[0]
                                        d_25_chosen2Valid_: bool
                                        out19_: bool
                                        out19_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_24_chosen2_)
                                        d_25_chosen2Valid_ = out19_
                                        if d_25_chosen2Valid_:
                                            d_26_gAppend2_: _dafny.Seq
                                            d_27_iAppend2_: bool
                                            d_28_cAppend2_: _dafny.Seq
                                            out20_: _dafny.Seq
                                            out21_: bool
                                            out22_: _dafny.Seq
                                            out20_, out21_, out22_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_24_chosen2_)
                                            d_26_gAppend2_ = out20_
                                            d_27_iAppend2_ = out21_
                                            d_28_cAppend2_ = out22_
                                            generated = d_26_gAppend2_
                                            insideConstrainedOut = d_27_iAppend2_
                                            currentConstrainedOut = d_28_cAppend2_
                                        elif True:
                                            raise _dafny.Break("0")
                                    elif True:
                                        raise _dafny.Break("0")
                    elif True:
                        if not(d_2_openedOnce_):
                            d_29_gOpen_: _dafny.Seq
                            d_30_iOpen_: bool
                            d_31_cOpen_: _dafny.Seq
                            out23_: _dafny.Seq
                            out24_: bool
                            out25_: _dafny.Seq
                            out23_, out24_, out25_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_29_gOpen_ = out23_
                            d_30_iOpen_ = out24_
                            d_31_cOpen_ = out25_
                            generated = d_29_gOpen_
                            insideConstrainedOut = d_30_iOpen_
                            currentConstrainedOut = d_31_cOpen_
                            d_2_openedOnce_ = True
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            (lm).GenerateLogits((prompt) + (generated))
                            (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([eosToken]), _dafny.BigRational('1e2'))
                            d_32_nextOutside_: _dafny.Seq
                            out26_: _dafny.Seq
                            out26_ = (lm).ChooseNextToken()
                            d_32_nextOutside_ = out26_
                            (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_32_nextOutside_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_32_nextOutside_]))
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

