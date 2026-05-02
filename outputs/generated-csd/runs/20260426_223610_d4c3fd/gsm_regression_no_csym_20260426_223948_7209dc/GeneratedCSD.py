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
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if insideConstrainedOut:
                        d_2_completeNow_: bool
                        d_2_completeNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_2_completeNow_:
                            d_3_closedGenerated_: _dafny.Seq
                            d_4_closedInside_: bool
                            d_5_closedCurrent_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_3_closedGenerated_ = out0_
                            d_4_closedInside_ = out1_
                            d_5_closedCurrent_ = out2_
                            generated = d_3_closedGenerated_
                            insideConstrainedOut = d_4_closedInside_
                            currentConstrainedOut = d_5_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_6_remaining_: int
                            d_6_remaining_ = (maxSteps) - (d_1_steps_)
                            if (d_6_remaining_) <= (1):
                                raise _dafny.Break("0")
                            elif True:
                                d_7_nextc_: _dafny.Seq
                                out3_: _dafny.Seq
                                out3_ = (d_0_helpers_).ConstrainedStep(lm, parser, prompt, currentConstrainedOut, eosToken)
                                d_7_nextc_ = out3_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_7_nextc_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_8_appendedGenerated_: _dafny.Seq
                                    d_9_appendedInside_: bool
                                    d_10_appendedCurrent_: _dafny.Seq
                                    out4_: _dafny.Seq
                                    out5_: bool
                                    out6_: _dafny.Seq
                                    out4_, out5_, out6_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_7_nextc_)
                                    d_8_appendedGenerated_ = out4_
                                    d_9_appendedInside_ = out5_
                                    d_10_appendedCurrent_ = out6_
                                    generated = d_8_appendedGenerated_
                                    insideConstrainedOut = d_9_appendedInside_
                                    currentConstrainedOut = d_10_appendedCurrent_
                    elif True:
                        d_11_remainingOut_: int
                        d_11_remainingOut_ = (maxSteps) - (d_1_steps_)
                        if (d_11_remainingOut_) <= (1):
                            d_12_next_: _dafny.Seq
                            out7_: _dafny.Seq
                            out7_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_12_next_ = out7_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_12_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_12_next_]))
                        elif True:
                            (lm).GenerateLogits((prompt) + (generated))
                            d_13_argmax_: _dafny.Seq
                            out8_: _dafny.Seq
                            out8_ = (d_0_helpers_).GetHighestLogitToken(lm)
                            d_13_argmax_ = out8_
                            d_14_openLogit_: _dafny.BigRational
                            out9_: _dafny.BigRational
                            out9_ = (d_0_helpers_).GetTokenLogit(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))
                            d_14_openLogit_ = out9_
                            d_15_argmaxLogit_: _dafny.BigRational
                            out10_: _dafny.BigRational
                            out10_ = (d_0_helpers_).GetTokenLogit(lm, d_13_argmax_)
                            d_15_argmaxLogit_ = out10_
                            if ((d_13_argmax_) != (eosToken)) and ((d_14_openLogit_) >= ((d_15_argmaxLogit_) - (_dafny.BigRational('1e0')))):
                                d_16_openedGenerated_: _dafny.Seq
                                d_17_openedInside_: bool
                                d_18_openedCurrent_: _dafny.Seq
                                out11_: _dafny.Seq
                                out12_: bool
                                out13_: _dafny.Seq
                                out11_, out12_, out13_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                d_16_openedGenerated_ = out11_
                                d_17_openedInside_ = out12_
                                d_18_openedCurrent_ = out13_
                                generated = d_16_openedGenerated_
                                insideConstrainedOut = d_17_openedInside_
                                currentConstrainedOut = d_18_openedCurrent_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                d_19_next2_: _dafny.Seq
                                out14_: _dafny.Seq
                                out14_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                                d_19_next2_ = out14_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_19_next2_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif (d_19_next2_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    d_20_openedGenerated2_: _dafny.Seq
                                    d_21_openedInside2_: bool
                                    d_22_openedCurrent2_: _dafny.Seq
                                    out15_: _dafny.Seq
                                    out16_: bool
                                    out17_: _dafny.Seq
                                    out15_, out16_, out17_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                    d_20_openedGenerated2_ = out15_
                                    d_21_openedInside2_ = out16_
                                    d_22_openedCurrent2_ = out17_
                                    generated = d_20_openedGenerated2_
                                    insideConstrainedOut = d_21_openedInside2_
                                    currentConstrainedOut = d_22_openedCurrent2_
                                elif True:
                                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_19_next2_]))
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

