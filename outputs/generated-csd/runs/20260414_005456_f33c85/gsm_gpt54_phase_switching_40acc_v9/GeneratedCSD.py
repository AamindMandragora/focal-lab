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
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if ((len(generated)) > (len(generatedPrefix))) and (((d_1_steps_) + (1)) < (maxSteps)):
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
                            d_2_suffix_ = _dafny.SeqWithoutIsStrInference((generated)[len(generatedPrefix)::])
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_6_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_6_next_ = out3_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_6_next_) == (eosToken):
                                pass
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_6_next_]))
                                d_2_suffix_ = _dafny.SeqWithoutIsStrInference((generated)[len(generatedPrefix)::])
                            if (d_6_next_) == (eosToken):
                                raise _dafny.Break("0")
                    elif True:
                        d_7_isComplete_: bool
                        d_7_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_7_isComplete_:
                            d_8_closedGenerated_: _dafny.Seq
                            d_9_closedInside_: bool
                            d_10_closedCurrent_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_8_closedGenerated_ = out4_
                            d_9_closedInside_ = out5_
                            d_10_closedCurrent_ = out6_
                            generated = d_8_closedGenerated_
                            insideConstrainedOut = d_9_closedInside_
                            currentConstrainedOut = d_10_closedCurrent_
                            d_2_suffix_ = _dafny.SeqWithoutIsStrInference((generated)[len(generatedPrefix)::])
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_11_narrow_: bool
                            out7_: bool
                            out7_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                            d_11_narrow_ = out7_
                            if d_11_narrow_:
                                d_12_repairedGenerated_: _dafny.Seq
                                d_13_repairedCurrent_: _dafny.Seq
                                out8_: _dafny.Seq
                                out9_: _dafny.Seq
                                out8_, out9_ = (d_0_helpers_).RollbackConstrainedSpan(parser, generated, currentConstrainedOut)
                                d_12_repairedGenerated_ = out8_
                                d_13_repairedCurrent_ = out9_
                                generated = d_12_repairedGenerated_
                                currentConstrainedOut = d_13_repairedCurrent_
                                insideConstrainedOut = True
                                d_2_suffix_ = _dafny.SeqWithoutIsStrInference((generated)[len(generatedPrefix)::])
                                d_1_steps_ = (d_1_steps_) + (1)
                                d_14_repairedComplete_: bool
                                d_14_repairedComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                                if d_14_repairedComplete_:
                                    d_15_closedGenerated2_: _dafny.Seq
                                    d_16_closedInside2_: bool
                                    d_17_closedCurrent2_: _dafny.Seq
                                    out10_: _dafny.Seq
                                    out11_: bool
                                    out12_: _dafny.Seq
                                    out10_, out11_, out12_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_15_closedGenerated2_ = out10_
                                    d_16_closedInside2_ = out11_
                                    d_17_closedCurrent2_ = out12_
                                    generated = d_15_closedGenerated2_
                                    insideConstrainedOut = d_16_closedInside2_
                                    currentConstrainedOut = d_17_closedCurrent2_
                                    d_2_suffix_ = _dafny.SeqWithoutIsStrInference((generated)[len(generatedPrefix)::])
                            elif True:
                                d_18_next_: _dafny.Seq
                                out13_: _dafny.Seq
                                out13_ = (d_0_helpers_).ConstrainedStep(lm, parser, prompt, currentConstrainedOut, eosToken)
                                d_18_next_ = out13_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_18_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                d_19_valid_: bool
                                out14_: bool
                                out14_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_18_next_)
                                d_19_valid_ = out14_
                                if d_19_valid_:
                                    d_20_appendedGenerated_: _dafny.Seq
                                    d_21_appendedInside_: bool
                                    d_22_appendedCurrent_: _dafny.Seq
                                    out15_: _dafny.Seq
                                    out16_: bool
                                    out17_: _dafny.Seq
                                    out15_, out16_, out17_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_18_next_)
                                    d_20_appendedGenerated_ = out15_
                                    d_21_appendedInside_ = out16_
                                    d_22_appendedCurrent_ = out17_
                                    generated = d_20_appendedGenerated_
                                    insideConstrainedOut = d_21_appendedInside_
                                    currentConstrainedOut = d_22_appendedCurrent_
                                    d_2_suffix_ = _dafny.SeqWithoutIsStrInference((generated)[len(generatedPrefix)::])
                                elif True:
                                    d_23_repairedGenerated2_: _dafny.Seq
                                    d_24_repairedCurrent2_: _dafny.Seq
                                    out18_: _dafny.Seq
                                    out19_: _dafny.Seq
                                    out18_, out19_ = (d_0_helpers_).RollbackConstrainedSpan(parser, generated, currentConstrainedOut)
                                    d_23_repairedGenerated2_ = out18_
                                    d_24_repairedCurrent2_ = out19_
                                    generated = d_23_repairedGenerated2_
                                    currentConstrainedOut = d_24_repairedCurrent2_
                                    insideConstrainedOut = True
                                    d_2_suffix_ = _dafny.SeqWithoutIsStrInference((generated)[len(generatedPrefix)::])
                                    d_25_repairedComplete2_: bool
                                    d_25_repairedComplete2_ = (parser).IsCompletePrefix(currentConstrainedOut)
                                    if d_25_repairedComplete2_:
                                        d_26_closedGenerated3_: _dafny.Seq
                                        d_27_closedInside3_: bool
                                        d_28_closedCurrent3_: _dafny.Seq
                                        out20_: _dafny.Seq
                                        out21_: bool
                                        out22_: _dafny.Seq
                                        out20_, out21_, out22_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                        d_26_closedGenerated3_ = out20_
                                        d_27_closedInside3_ = out21_
                                        d_28_closedCurrent3_ = out22_
                                        generated = d_26_closedGenerated3_
                                        insideConstrainedOut = d_27_closedInside3_
                                        currentConstrainedOut = d_28_closedCurrent3_
                                        d_2_suffix_ = _dafny.SeqWithoutIsStrInference((generated)[len(generatedPrefix)::])
                    pass
            pass
        cost = d_0_helpers_.cost
        return generated, insideConstrainedOut, currentConstrainedOut, cost

