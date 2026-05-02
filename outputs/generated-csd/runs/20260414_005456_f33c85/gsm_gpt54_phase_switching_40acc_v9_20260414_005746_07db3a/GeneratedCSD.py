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
        d_2_suffix_: _dafny.Seq
        d_2_suffix_ = _dafny.SeqWithoutIsStrInference([])
        d_3_haveNonEmptySegment_: bool
        d_3_haveNonEmptySegment_ = (insideConstrained) and ((len(currentConstrained)) > (0))
        d_4_openedSpan_: bool
        d_4_openedSpan_ = insideConstrained
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if ((not(d_3_haveNonEmptySegment_)) and (not(d_4_openedSpan_))) and ((d_1_steps_) < (maxSteps)):
                            d_5_openedGenerated_: _dafny.Seq
                            d_6_openedInside_: bool
                            d_7_openedCurrent_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_5_openedGenerated_ = out0_
                            d_6_openedInside_ = out1_
                            d_7_openedCurrent_ = out2_
                            generated = d_5_openedGenerated_
                            insideConstrainedOut = d_6_openedInside_
                            currentConstrainedOut = d_7_openedCurrent_
                            d_4_openedSpan_ = True
                            d_2_suffix_ = _dafny.SeqWithoutIsStrInference((generated)[len(generatedPrefix)::])
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_8_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_8_next_ = out3_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_8_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_8_next_]))
                                d_2_suffix_ = _dafny.SeqWithoutIsStrInference((generated)[len(generatedPrefix)::])
                    elif True:
                        if (len(currentConstrainedOut)) == (0):
                            d_9_isComplete0_: bool
                            d_9_isComplete0_ = (parser).IsCompletePrefix(currentConstrainedOut)
                            if d_9_isComplete0_:
                                d_10_closedGenerated0_: _dafny.Seq
                                d_11_closedInside0_: bool
                                d_12_closedCurrent0_: _dafny.Seq
                                out4_: _dafny.Seq
                                out5_: bool
                                out6_: _dafny.Seq
                                out4_, out5_, out6_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_10_closedGenerated0_ = out4_
                                d_11_closedInside0_ = out5_
                                d_12_closedCurrent0_ = out6_
                                generated = d_10_closedGenerated0_
                                insideConstrainedOut = d_11_closedInside0_
                                currentConstrainedOut = d_12_closedCurrent0_
                                d_2_suffix_ = _dafny.SeqWithoutIsStrInference((generated)[len(generatedPrefix)::])
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                d_13_next0_: _dafny.Seq
                                out7_: _dafny.Seq
                                out7_ = (d_0_helpers_).ConstrainedStep(lm, parser, prompt, currentConstrainedOut, eosToken)
                                d_13_next0_ = out7_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_13_next0_) == (eosToken):
                                    pass
                                elif True:
                                    d_14_valid0_: bool
                                    out8_: bool
                                    out8_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_13_next0_)
                                    d_14_valid0_ = out8_
                                    if d_14_valid0_:
                                        d_15_appendedGenerated0_: _dafny.Seq
                                        d_16_appendedInside0_: bool
                                        d_17_appendedCurrent0_: _dafny.Seq
                                        out9_: _dafny.Seq
                                        out10_: bool
                                        out11_: _dafny.Seq
                                        out9_, out10_, out11_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_13_next0_)
                                        d_15_appendedGenerated0_ = out9_
                                        d_16_appendedInside0_ = out10_
                                        d_17_appendedCurrent0_ = out11_
                                        generated = d_15_appendedGenerated0_
                                        insideConstrainedOut = d_16_appendedInside0_
                                        currentConstrainedOut = d_17_appendedCurrent0_
                                        d_3_haveNonEmptySegment_ = True
                                        d_2_suffix_ = _dafny.SeqWithoutIsStrInference((generated)[len(generatedPrefix)::])
                        elif True:
                            d_18_isComplete_: bool
                            d_18_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                            if d_18_isComplete_:
                                d_19_closedGenerated_: _dafny.Seq
                                d_20_closedInside_: bool
                                d_21_closedCurrent_: _dafny.Seq
                                out12_: _dafny.Seq
                                out13_: bool
                                out14_: _dafny.Seq
                                out12_, out13_, out14_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_19_closedGenerated_ = out12_
                                d_20_closedInside_ = out13_
                                d_21_closedCurrent_ = out14_
                                generated = d_19_closedGenerated_
                                insideConstrainedOut = d_20_closedInside_
                                currentConstrainedOut = d_21_closedCurrent_
                                d_2_suffix_ = _dafny.SeqWithoutIsStrInference((generated)[len(generatedPrefix)::])
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                d_22_narrow_: bool
                                out15_: bool
                                out15_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                                d_22_narrow_ = out15_
                                if d_22_narrow_:
                                    d_23_repairedGenerated_: _dafny.Seq
                                    d_24_repairedCurrent_: _dafny.Seq
                                    out16_: _dafny.Seq
                                    out17_: _dafny.Seq
                                    out16_, out17_ = (d_0_helpers_).RollbackConstrainedSpan(parser, generated, currentConstrainedOut)
                                    d_23_repairedGenerated_ = out16_
                                    d_24_repairedCurrent_ = out17_
                                    generated = d_23_repairedGenerated_
                                    currentConstrainedOut = d_24_repairedCurrent_
                                    insideConstrainedOut = True
                                    d_2_suffix_ = _dafny.SeqWithoutIsStrInference((generated)[len(generatedPrefix)::])
                                    d_1_steps_ = (d_1_steps_) + (1)
                                elif True:
                                    d_25_next2_: _dafny.Seq
                                    out18_: _dafny.Seq
                                    out18_ = (d_0_helpers_).ConstrainedStep(lm, parser, prompt, currentConstrainedOut, eosToken)
                                    d_25_next2_ = out18_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    if (d_25_next2_) == (eosToken):
                                        pass
                                    elif True:
                                        d_26_valid2_: bool
                                        out19_: bool
                                        out19_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_25_next2_)
                                        d_26_valid2_ = out19_
                                        if d_26_valid2_:
                                            d_27_appendedGenerated_: _dafny.Seq
                                            d_28_appendedInside_: bool
                                            d_29_appendedCurrent_: _dafny.Seq
                                            out20_: _dafny.Seq
                                            out21_: bool
                                            out22_: _dafny.Seq
                                            out20_, out21_, out22_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_25_next2_)
                                            d_27_appendedGenerated_ = out20_
                                            d_28_appendedInside_ = out21_
                                            d_29_appendedCurrent_ = out22_
                                            generated = d_27_appendedGenerated_
                                            insideConstrainedOut = d_28_appendedInside_
                                            currentConstrainedOut = d_29_appendedCurrent_
                                            d_3_haveNonEmptySegment_ = True
                                            d_2_suffix_ = _dafny.SeqWithoutIsStrInference((generated)[len(generatedPrefix)::])
                                        elif True:
                                            d_30_repairedGenerated2_: _dafny.Seq
                                            d_31_repairedCurrent2_: _dafny.Seq
                                            out23_: _dafny.Seq
                                            out24_: _dafny.Seq
                                            out23_, out24_ = (d_0_helpers_).RollbackConstrainedSpan(parser, generated, currentConstrainedOut)
                                            d_30_repairedGenerated2_ = out23_
                                            d_31_repairedCurrent2_ = out24_
                                            generated = d_30_repairedGenerated2_
                                            currentConstrainedOut = d_31_repairedCurrent2_
                                            insideConstrainedOut = True
                                            d_2_suffix_ = _dafny.SeqWithoutIsStrInference((generated)[len(generatedPrefix)::])
                                            d_1_steps_ = (d_1_steps_) + (1)
                    pass
            pass
        cost = d_0_helpers_.cost
        return generated, insideConstrainedOut, currentConstrainedOut, cost

