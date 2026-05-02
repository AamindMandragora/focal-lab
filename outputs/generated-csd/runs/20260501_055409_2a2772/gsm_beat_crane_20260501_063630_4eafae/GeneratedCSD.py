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
    def MyCSDStrategy(lm, parser, prompt, generatedPrefix, insideConstrained, currentConstrained, maxSteps, stepTokenBudget, validTokenGroups, eosToken):
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
        d_2_reserveTail_: int
        d_2_reserveTail_ = 6
        d_3_chunkSize_: int
        d_3_chunkSize_ = 12
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if insideConstrainedOut:
                        d_4_completeNow_: bool
                        d_4_completeNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_4_completeNow_:
                            if ((d_1_steps_) + (1)) <= (maxSteps):
                                d_5_gClose_: _dafny.Seq
                                d_6_iClose_: bool
                                d_7_cClose_: _dafny.Seq
                                out0_: _dafny.Seq
                                out1_: bool
                                out2_: _dafny.Seq
                                out0_, out1_, out2_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_5_gClose_ = out0_
                                d_6_iClose_ = out1_
                                d_7_cClose_ = out2_
                                generated = d_5_gClose_
                                insideConstrainedOut = d_6_iClose_
                                currentConstrainedOut = d_7_cClose_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                raise _dafny.Break("0")
                        elif True:
                            if ((d_1_steps_) + (1)) <= (maxSteps):
                                d_8_next_: _dafny.Seq
                                out3_: _dafny.Seq
                                out3_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, (prompt) + (generated), currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                                d_8_next_ = out3_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_8_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_9_gApp_: _dafny.Seq
                                    d_10_iApp_: bool
                                    d_11_cApp_: _dafny.Seq
                                    out4_: _dafny.Seq
                                    out5_: bool
                                    out6_: _dafny.Seq
                                    out4_, out5_, out6_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_8_next_)
                                    d_9_gApp_ = out4_
                                    d_10_iApp_ = out5_
                                    d_11_cApp_ = out6_
                                    generated = d_9_gApp_
                                    insideConstrainedOut = d_10_iApp_
                                    currentConstrainedOut = d_11_cApp_
                            elif True:
                                raise _dafny.Break("0")
                    elif True:
                        d_12_remaining_: int
                        d_12_remaining_ = (maxSteps) - (d_1_steps_)
                        d_13_canOpen_: bool
                        d_13_canOpen_ = False
                        if (((_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))) in ((lm).Tokens)) and ((d_12_remaining_) <= (d_2_reserveTail_))) and ((d_12_remaining_) >= (2)):
                            d_13_canOpen_ = True
                        if d_13_canOpen_:
                            d_14_gOpen_: _dafny.Seq
                            d_15_iOpen_: bool
                            d_16_cOpen_: _dafny.Seq
                            out7_: _dafny.Seq
                            out8_: bool
                            out9_: _dafny.Seq
                            out7_, out8_, out9_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_14_gOpen_ = out7_
                            d_15_iOpen_ = out8_
                            d_16_cOpen_ = out9_
                            generated = d_14_gOpen_
                            insideConstrainedOut = d_15_iOpen_
                            currentConstrainedOut = d_16_cOpen_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_17_maxChunk_: int
                            d_17_maxChunk_ = d_3_chunkSize_
                            if (d_12_remaining_) > (d_2_reserveTail_):
                                if ((d_12_remaining_) - (d_2_reserveTail_)) < (d_17_maxChunk_):
                                    d_17_maxChunk_ = (d_12_remaining_) - (d_2_reserveTail_)
                            elif True:
                                if (d_12_remaining_) < (d_17_maxChunk_):
                                    d_17_maxChunk_ = d_12_remaining_
                            if (d_17_maxChunk_) == (0):
                                raise _dafny.Break("0")
                            elif True:
                                d_18_gChunk_: _dafny.Seq
                                d_19_stoppedOnOpenSpan_: bool
                                d_20_stoppedOnEos_: bool
                                d_21_stepsUsed_: int
                                out10_: _dafny.Seq
                                out11_: bool
                                out12_: bool
                                out13_: int
                                out10_, out11_, out12_, out13_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_17_maxChunk_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                                d_18_gChunk_ = out10_
                                d_19_stoppedOnOpenSpan_ = out11_
                                d_20_stoppedOnEos_ = out12_
                                d_21_stepsUsed_ = out13_
                                generated = d_18_gChunk_
                                d_1_steps_ = (d_1_steps_) + (d_21_stepsUsed_)
                                if d_20_stoppedOnEos_:
                                    if (((_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))) in ((lm).Tokens)) and ((d_1_steps_) < (maxSteps))) and (((maxSteps) - (d_1_steps_)) >= (2)):
                                        d_22_gOpen2_: _dafny.Seq
                                        d_23_iOpen2_: bool
                                        d_24_cOpen2_: _dafny.Seq
                                        out14_: _dafny.Seq
                                        out15_: bool
                                        out16_: _dafny.Seq
                                        out14_, out15_, out16_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                        d_22_gOpen2_ = out14_
                                        d_23_iOpen2_ = out15_
                                        d_24_cOpen2_ = out16_
                                        generated = d_22_gOpen2_
                                        insideConstrainedOut = d_23_iOpen2_
                                        currentConstrainedOut = d_24_cOpen2_
                                        d_1_steps_ = (d_1_steps_) + (1)
                                    elif True:
                                        raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

