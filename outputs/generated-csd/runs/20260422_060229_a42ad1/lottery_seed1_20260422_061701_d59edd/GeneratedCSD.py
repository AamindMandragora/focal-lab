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
                        (lm).GenerateLogits((prompt) + (generated))
                        (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]), _dafny.BigRational('3e0'))
                        (d_0_helpers_).PenalizeTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]), _dafny.BigRational('1e2'))
                        d_2_topOutside_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).GetHighestLogitToken(lm)
                        d_2_topOutside_ = out0_
                        if (d_2_topOutside_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                            d_3_openedGenerated_: _dafny.Seq
                            d_4_openedInside_: bool
                            d_5_openedCurrent_: _dafny.Seq
                            out1_: _dafny.Seq
                            out2_: bool
                            out3_: _dafny.Seq
                            out1_, out2_, out3_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_3_openedGenerated_ = out1_
                            d_4_openedInside_ = out2_
                            d_5_openedCurrent_ = out3_
                            generated = d_3_openedGenerated_
                            insideConstrainedOut = d_4_openedInside_
                            currentConstrainedOut = d_5_openedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_6_nextOutside_: _dafny.Seq
                            out4_: _dafny.Seq
                            out4_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_6_nextOutside_ = out4_
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_6_nextOutside_]))
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_6_nextOutside_) == (eosToken):
                                raise _dafny.Break("0")
                    elif True:
                        d_7_completeNow_: bool
                        d_7_completeNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_7_completeNow_:
                            d_8_closedGenerated_: _dafny.Seq
                            d_9_closedInside_: bool
                            d_10_closedCurrent_: _dafny.Seq
                            out5_: _dafny.Seq
                            out6_: bool
                            out7_: _dafny.Seq
                            out5_, out6_, out7_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_8_closedGenerated_ = out5_
                            d_9_closedInside_ = out6_
                            d_10_closedCurrent_ = out7_
                            generated = d_8_closedGenerated_
                            insideConstrainedOut = d_9_closedInside_
                            currentConstrainedOut = d_10_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_11_narrow_: bool
                            out8_: bool
                            out8_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 2)
                            d_11_narrow_ = out8_
                            d_12_nextInside_: _dafny.Seq
                            out9_: _dafny.Seq
                            out9_ = (d_0_helpers_).ConstrainedStep(lm, parser, prompt, currentConstrainedOut, eosToken)
                            d_12_nextInside_ = out9_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_12_nextInside_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_13_appendedGenerated_: _dafny.Seq
                                d_14_appendedInside_: bool
                                d_15_appendedCurrent_: _dafny.Seq
                                out10_: _dafny.Seq
                                out11_: bool
                                out12_: _dafny.Seq
                                out10_, out11_, out12_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_12_nextInside_)
                                d_13_appendedGenerated_ = out10_
                                d_14_appendedInside_ = out11_
                                d_15_appendedCurrent_ = out12_
                                generated = d_13_appendedGenerated_
                                insideConstrainedOut = d_14_appendedInside_
                                currentConstrainedOut = d_15_appendedCurrent_
                                d_16_completeAfter_: bool
                                d_16_completeAfter_ = (parser).IsCompletePrefix(currentConstrainedOut)
                                if ((d_11_narrow_) and (d_16_completeAfter_)) and ((d_1_steps_) < (maxSteps)):
                                    d_17_closedGenerated2_: _dafny.Seq
                                    d_18_closedInside2_: bool
                                    d_19_closedCurrent2_: _dafny.Seq
                                    out13_: _dafny.Seq
                                    out14_: bool
                                    out15_: _dafny.Seq
                                    out13_, out14_, out15_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_17_closedGenerated2_ = out13_
                                    d_18_closedInside2_ = out14_
                                    d_19_closedCurrent2_ = out15_
                                    generated = d_17_closedGenerated2_
                                    insideConstrainedOut = d_18_closedInside2_
                                    currentConstrainedOut = d_19_closedCurrent2_
                                    d_1_steps_ = (d_1_steps_) + (1)
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_20_completeEnd_: bool
            d_20_completeEnd_ = (parser).IsCompletePrefix(currentConstrainedOut)
            if d_20_completeEnd_:
                d_21_closedGenerated3_: _dafny.Seq
                d_22_closedInside3_: bool
                d_23_closedCurrent3_: _dafny.Seq
                out16_: _dafny.Seq
                out17_: bool
                out18_: _dafny.Seq
                out16_, out17_, out18_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_21_closedGenerated3_ = out16_
                d_22_closedInside3_ = out17_
                d_23_closedCurrent3_ = out18_
                generated = d_21_closedGenerated3_
                insideConstrainedOut = d_22_closedInside3_
                currentConstrainedOut = d_23_closedCurrent3_
                d_1_steps_ = (d_1_steps_) + (1)
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

