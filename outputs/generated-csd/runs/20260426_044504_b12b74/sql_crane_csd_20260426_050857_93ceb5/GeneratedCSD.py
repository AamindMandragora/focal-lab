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
                            d_4_gClose_: _dafny.Seq
                            d_5_iClose_: bool
                            d_6_cClose_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_4_gClose_ = out0_
                            d_5_iClose_ = out1_
                            d_6_cClose_ = out2_
                            generated = d_4_gClose_
                            insideConstrainedOut = d_5_iClose_
                            currentConstrainedOut = d_6_cClose_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_7_validCount_: int
                            out3_: int
                            out3_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                            d_7_validCount_ = out3_
                            if (d_7_validCount_) <= (6):
                                (lm).GenerateLogits((prompt) + (generated))
                                d_8_candSmall_: _dafny.Seq
                                out4_: _dafny.Seq
                                out4_ = (d_0_helpers_).TopValidCandidates(lm, parser, prompt, currentConstrainedOut, 6, eosToken)
                                d_8_candSmall_ = out4_
                                (d_0_helpers_).BoostTokenLogits(lm, d_8_candSmall_, _dafny.BigRational('12e0'))
                            elif True:
                                d_9_narrow_: bool
                                out5_: bool
                                out5_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 4)
                                d_9_narrow_ = out5_
                                if d_9_narrow_:
                                    (lm).GenerateLogits((prompt) + (generated))
                                    d_10_candNarrow_: _dafny.Seq
                                    out6_: _dafny.Seq
                                    out6_ = (d_0_helpers_).TopValidCandidates(lm, parser, prompt, currentConstrainedOut, 8, eosToken)
                                    d_10_candNarrow_ = out6_
                                    (d_0_helpers_).BoostTokenLogits(lm, d_10_candNarrow_, _dafny.BigRational('8e0'))
                            d_11_nextInside_: _dafny.Seq
                            out7_: _dafny.Seq
                            out7_ = (d_0_helpers_).ConstrainedStep(lm, parser, prompt, currentConstrainedOut, eosToken)
                            d_11_nextInside_ = out7_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_11_nextInside_) == (eosToken):
                                d_12_fallback_: _dafny.Seq
                                out8_: _dafny.Seq
                                out8_ = (d_0_helpers_).TopValidCandidates(lm, parser, prompt, currentConstrainedOut, 1, eosToken)
                                d_12_fallback_ = out8_
                                if (len(d_12_fallback_)) > (0):
                                    d_13_chosen_: _dafny.Seq
                                    d_13_chosen_ = (d_12_fallback_)[0]
                                    d_14_chosenValid_: bool
                                    out9_: bool
                                    out9_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_13_chosen_)
                                    d_14_chosenValid_ = out9_
                                    if d_14_chosenValid_:
                                        d_15_gAppendFallback_: _dafny.Seq
                                        d_16_iAppendFallback_: bool
                                        d_17_cAppendFallback_: _dafny.Seq
                                        out10_: _dafny.Seq
                                        out11_: bool
                                        out12_: _dafny.Seq
                                        out10_, out11_, out12_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_13_chosen_)
                                        d_15_gAppendFallback_ = out10_
                                        d_16_iAppendFallback_ = out11_
                                        d_17_cAppendFallback_ = out12_
                                        generated = d_15_gAppendFallback_
                                        insideConstrainedOut = d_16_iAppendFallback_
                                        currentConstrainedOut = d_17_cAppendFallback_
                                    elif True:
                                        raise _dafny.Break("0")
                                elif True:
                                    raise _dafny.Break("0")
                            elif True:
                                d_18_gAppend_: _dafny.Seq
                                d_19_iAppend_: bool
                                d_20_cAppend_: _dafny.Seq
                                out13_: _dafny.Seq
                                out14_: bool
                                out15_: _dafny.Seq
                                out13_, out14_, out15_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_11_nextInside_)
                                d_18_gAppend_ = out13_
                                d_19_iAppend_ = out14_
                                d_20_cAppend_ = out15_
                                generated = d_18_gAppend_
                                insideConstrainedOut = d_19_iAppend_
                                currentConstrainedOut = d_20_cAppend_
                    elif True:
                        if not(d_2_openedOnce_):
                            d_21_gOpen_: _dafny.Seq
                            d_22_iOpen_: bool
                            d_23_cOpen_: _dafny.Seq
                            out16_: _dafny.Seq
                            out17_: bool
                            out18_: _dafny.Seq
                            out16_, out17_, out18_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_21_gOpen_ = out16_
                            d_22_iOpen_ = out17_
                            d_23_cOpen_ = out18_
                            generated = d_21_gOpen_
                            insideConstrainedOut = d_22_iOpen_
                            currentConstrainedOut = d_23_cOpen_
                            d_2_openedOnce_ = True
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            (lm).GenerateLogits((prompt) + (generated))
                            (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([eosToken]), _dafny.BigRational('1e2'))
                            d_24_nextOutside_: _dafny.Seq
                            out19_: _dafny.Seq
                            out19_ = (lm).ChooseNextToken()
                            d_24_nextOutside_ = out19_
                            (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_24_nextOutside_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_24_nextOutside_]))
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

