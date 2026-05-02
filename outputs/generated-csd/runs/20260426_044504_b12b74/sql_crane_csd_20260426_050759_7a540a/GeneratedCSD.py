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
                            d_8_narrow_: bool
                            out4_: bool
                            out4_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 4)
                            d_8_narrow_ = out4_
                            if d_8_narrow_:
                                (lm).GenerateLogits((prompt) + (generated))
                                (d_0_helpers_).ScaleAllLogits(lm, _dafny.BigRational('5e-1'))
                                d_9_candNarrow_: _dafny.Seq
                                out5_: _dafny.Seq
                                out5_ = (d_0_helpers_).TopValidCandidates(lm, parser, prompt, currentConstrainedOut, 6, eosToken)
                                d_9_candNarrow_ = out5_
                                (d_0_helpers_).BoostTokenLogits(lm, d_9_candNarrow_, _dafny.BigRational('8e0'))
                            elif True:
                                if (d_7_validCount_) <= (8):
                                    (lm).GenerateLogits((prompt) + (generated))
                                    d_10_candSmall_: _dafny.Seq
                                    out6_: _dafny.Seq
                                    out6_ = (d_0_helpers_).TopValidCandidates(lm, parser, prompt, currentConstrainedOut, 8, eosToken)
                                    d_10_candSmall_ = out6_
                                    (d_0_helpers_).BoostTokenLogits(lm, d_10_candSmall_, _dafny.BigRational('3e0'))
                            d_11_nextInside_: _dafny.Seq
                            out7_: _dafny.Seq
                            out7_ = (d_0_helpers_).ConstrainedStep(lm, parser, prompt, currentConstrainedOut, eosToken)
                            d_11_nextInside_ = out7_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_11_nextInside_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_12_gAppend_: _dafny.Seq
                                d_13_iAppend_: bool
                                d_14_cAppend_: _dafny.Seq
                                out8_: _dafny.Seq
                                out9_: bool
                                out10_: _dafny.Seq
                                out8_, out9_, out10_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_11_nextInside_)
                                d_12_gAppend_ = out8_
                                d_13_iAppend_ = out9_
                                d_14_cAppend_ = out10_
                                generated = d_12_gAppend_
                                insideConstrainedOut = d_13_iAppend_
                                currentConstrainedOut = d_14_cAppend_
                    elif True:
                        if not(d_2_openedOnce_):
                            d_15_gOpen_: _dafny.Seq
                            d_16_iOpen_: bool
                            d_17_cOpen_: _dafny.Seq
                            out11_: _dafny.Seq
                            out12_: bool
                            out13_: _dafny.Seq
                            out11_, out12_, out13_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_15_gOpen_ = out11_
                            d_16_iOpen_ = out12_
                            d_17_cOpen_ = out13_
                            generated = d_15_gOpen_
                            insideConstrainedOut = d_16_iOpen_
                            currentConstrainedOut = d_17_cOpen_
                            d_2_openedOnce_ = True
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            (lm).GenerateLogits((prompt) + (generated))
                            (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([eosToken]), _dafny.BigRational('1e2'))
                            d_18_nextOutside_: _dafny.Seq
                            out14_: _dafny.Seq
                            out14_ = (lm).ChooseNextToken()
                            d_18_nextOutside_ = out14_
                            (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_18_nextOutside_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_18_nextOutside_]))
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

