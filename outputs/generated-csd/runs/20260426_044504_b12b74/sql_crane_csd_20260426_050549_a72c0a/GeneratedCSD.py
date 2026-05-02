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
                            d_3_openedGenerated_: _dafny.Seq
                            d_4_openedInside_: bool
                            d_5_openedCurrent_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_3_openedGenerated_ = out0_
                            d_4_openedInside_ = out1_
                            d_5_openedCurrent_ = out2_
                            generated = d_3_openedGenerated_
                            insideConstrainedOut = d_4_openedInside_
                            currentConstrainedOut = d_5_openedCurrent_
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
                        d_7_complete_: bool
                        d_7_complete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_7_complete_:
                            (lm).GenerateLogits((prompt) + (generated))
                            (d_0_helpers_).PenalizeTokenLogits(lm, _dafny.SeqWithoutIsStrInference([eosToken]), _dafny.BigRational('1e2'))
                            d_8_topToken_: _dafny.Seq
                            out4_: _dafny.Seq
                            out4_ = (d_0_helpers_).GetHighestLogitToken(lm)
                            d_8_topToken_ = out4_
                            d_9_topContinues_: bool
                            out5_: bool
                            out5_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_8_topToken_)
                            d_9_topContinues_ = out5_
                            if (d_9_topContinues_) or (VerifiedDecoderAgent.default__.Contains(d_8_topToken_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))):
                                d_10_nextInsideComplete_: _dafny.Seq
                                out6_: _dafny.Seq
                                out6_ = (d_0_helpers_).ConstrainedStep(lm, parser, (prompt) + (generated), currentConstrainedOut, eosToken)
                                d_10_nextInsideComplete_ = out6_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_10_nextInsideComplete_) == (eosToken):
                                    d_11_closedGenerated0_: _dafny.Seq
                                    d_12_closedInside0_: bool
                                    d_13_closedCurrent0_: _dafny.Seq
                                    out7_: _dafny.Seq
                                    out8_: bool
                                    out9_: _dafny.Seq
                                    out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_11_closedGenerated0_ = out7_
                                    d_12_closedInside0_ = out8_
                                    d_13_closedCurrent0_ = out9_
                                    generated = d_11_closedGenerated0_
                                    insideConstrainedOut = d_12_closedInside0_
                                    currentConstrainedOut = d_13_closedCurrent0_
                                elif True:
                                    d_14_stillComplete_: bool
                                    d_14_stillComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                                    if d_14_stillComplete_:
                                        d_15_closedGenerated1_: _dafny.Seq
                                        d_16_closedInside1_: bool
                                        d_17_closedCurrent1_: _dafny.Seq
                                        out10_: _dafny.Seq
                                        out11_: bool
                                        out12_: _dafny.Seq
                                        out10_, out11_, out12_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                        d_15_closedGenerated1_ = out10_
                                        d_16_closedInside1_ = out11_
                                        d_17_closedCurrent1_ = out12_
                                        generated = d_15_closedGenerated1_
                                        insideConstrainedOut = d_16_closedInside1_
                                        currentConstrainedOut = d_17_closedCurrent1_
                                    elif True:
                                        d_18_appendedGenerated0_: _dafny.Seq
                                        d_19_appendedInside0_: bool
                                        d_20_appendedCurrent0_: _dafny.Seq
                                        out13_: _dafny.Seq
                                        out14_: bool
                                        out15_: _dafny.Seq
                                        out13_, out14_, out15_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_10_nextInsideComplete_)
                                        d_18_appendedGenerated0_ = out13_
                                        d_19_appendedInside0_ = out14_
                                        d_20_appendedCurrent0_ = out15_
                                        generated = d_18_appendedGenerated0_
                                        insideConstrainedOut = d_19_appendedInside0_
                                        currentConstrainedOut = d_20_appendedCurrent0_
                            elif True:
                                d_21_closedGenerated_: _dafny.Seq
                                d_22_closedInside_: bool
                                d_23_closedCurrent_: _dafny.Seq
                                out16_: _dafny.Seq
                                out17_: bool
                                out18_: _dafny.Seq
                                out16_, out17_, out18_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_21_closedGenerated_ = out16_
                                d_22_closedInside_ = out17_
                                d_23_closedCurrent_ = out18_
                                generated = d_21_closedGenerated_
                                insideConstrainedOut = d_22_closedInside_
                                currentConstrainedOut = d_23_closedCurrent_
                                d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_24_nextInside_: _dafny.Seq
                            out19_: _dafny.Seq
                            out19_ = (d_0_helpers_).ConstrainedStep(lm, parser, (prompt) + (generated), currentConstrainedOut, eosToken)
                            d_24_nextInside_ = out19_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_24_nextInside_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_25_appendedGenerated_: _dafny.Seq
                                d_26_appendedInside_: bool
                                d_27_appendedCurrent_: _dafny.Seq
                                out20_: _dafny.Seq
                                out21_: bool
                                out22_: _dafny.Seq
                                out20_, out21_, out22_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_24_nextInside_)
                                d_25_appendedGenerated_ = out20_
                                d_26_appendedInside_ = out21_
                                d_27_appendedCurrent_ = out22_
                                generated = d_25_appendedGenerated_
                                insideConstrainedOut = d_26_appendedInside_
                                currentConstrainedOut = d_27_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

