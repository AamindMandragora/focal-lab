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
        d_3_openedAnyConstrained_: bool
        d_3_openedAnyConstrained_ = insideConstrained
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if (not(d_3_openedAnyConstrained_)) and (((d_1_steps_) + (2)) <= (maxSteps)):
                            d_4_openedGenerated_: _dafny.Seq
                            d_5_openedInside_: bool
                            d_6_openedCurrent_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_4_openedGenerated_ = out0_
                            d_5_openedInside_ = out1_
                            d_6_openedCurrent_ = out2_
                            generated = d_4_openedGenerated_
                            insideConstrainedOut = d_5_openedInside_
                            currentConstrainedOut = d_6_openedCurrent_
                            d_3_openedAnyConstrained_ = True
                            d_2_suffix_ = _dafny.SeqWithoutIsStrInference((generated)[len(generatedPrefix)::])
                            d_1_steps_ = (d_1_steps_) + (1)
                            if ((d_1_steps_) < (maxSteps)) and (insideConstrainedOut):
                                d_7_completeNow_: bool
                                d_7_completeNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                                if d_7_completeNow_:
                                    d_8_closedGenerated00_: _dafny.Seq
                                    d_9_closedInside00_: bool
                                    d_10_closedCurrent00_: _dafny.Seq
                                    out3_: _dafny.Seq
                                    out4_: bool
                                    out5_: _dafny.Seq
                                    out3_, out4_, out5_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_8_closedGenerated00_ = out3_
                                    d_9_closedInside00_ = out4_
                                    d_10_closedCurrent00_ = out5_
                                    generated = d_8_closedGenerated00_
                                    insideConstrainedOut = d_9_closedInside00_
                                    currentConstrainedOut = d_10_closedCurrent00_
                                    d_2_suffix_ = _dafny.SeqWithoutIsStrInference((generated)[len(generatedPrefix)::])
                                    d_1_steps_ = (d_1_steps_) + (1)
                                elif True:
                                    d_11_nextForced_: _dafny.Seq
                                    out6_: _dafny.Seq
                                    out6_ = (d_0_helpers_).ConstrainedStep(lm, parser, prompt, currentConstrainedOut, eosToken)
                                    d_11_nextForced_ = out6_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    if (d_11_nextForced_) == (eosToken):
                                        pass
                                    elif True:
                                        d_12_validForced_: bool
                                        out7_: bool
                                        out7_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_11_nextForced_)
                                        d_12_validForced_ = out7_
                                        if d_12_validForced_:
                                            d_13_appendedGenerated0_: _dafny.Seq
                                            d_14_appendedInside0_: bool
                                            d_15_appendedCurrent0_: _dafny.Seq
                                            out8_: _dafny.Seq
                                            out9_: bool
                                            out10_: _dafny.Seq
                                            out8_, out9_, out10_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_11_nextForced_)
                                            d_13_appendedGenerated0_ = out8_
                                            d_14_appendedInside0_ = out9_
                                            d_15_appendedCurrent0_ = out10_
                                            generated = d_13_appendedGenerated0_
                                            insideConstrainedOut = d_14_appendedInside0_
                                            currentConstrainedOut = d_15_appendedCurrent0_
                                            d_2_suffix_ = _dafny.SeqWithoutIsStrInference((generated)[len(generatedPrefix)::])
                                            d_16_completeAfterFirst_: bool
                                            d_16_completeAfterFirst_ = (parser).IsCompletePrefix(currentConstrainedOut)
                                            if (d_16_completeAfterFirst_) and ((d_1_steps_) < (maxSteps)):
                                                d_17_closedGenerated0_: _dafny.Seq
                                                d_18_closedInside0_: bool
                                                d_19_closedCurrent0_: _dafny.Seq
                                                out11_: _dafny.Seq
                                                out12_: bool
                                                out13_: _dafny.Seq
                                                out11_, out12_, out13_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                                d_17_closedGenerated0_ = out11_
                                                d_18_closedInside0_ = out12_
                                                d_19_closedCurrent0_ = out13_
                                                generated = d_17_closedGenerated0_
                                                insideConstrainedOut = d_18_closedInside0_
                                                currentConstrainedOut = d_19_closedCurrent0_
                                                d_2_suffix_ = _dafny.SeqWithoutIsStrInference((generated)[len(generatedPrefix)::])
                                                d_1_steps_ = (d_1_steps_) + (1)
                                        elif True:
                                            d_20_repairedGenerated0_: _dafny.Seq
                                            d_21_repairedCurrent0_: _dafny.Seq
                                            out14_: _dafny.Seq
                                            out15_: _dafny.Seq
                                            out14_, out15_ = (d_0_helpers_).RollbackConstrainedSpan(parser, generated, currentConstrainedOut)
                                            d_20_repairedGenerated0_ = out14_
                                            d_21_repairedCurrent0_ = out15_
                                            generated = d_20_repairedGenerated0_
                                            currentConstrainedOut = d_21_repairedCurrent0_
                                            insideConstrainedOut = True
                                            d_2_suffix_ = _dafny.SeqWithoutIsStrInference((generated)[len(generatedPrefix)::])
                        elif True:
                            d_22_next_: _dafny.Seq
                            out16_: _dafny.Seq
                            out16_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_22_next_ = out16_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_22_next_) == (eosToken):
                                pass
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_22_next_]))
                                d_2_suffix_ = _dafny.SeqWithoutIsStrInference((generated)[len(generatedPrefix)::])
                            if (d_22_next_) == (eosToken):
                                raise _dafny.Break("0")
                    elif True:
                        d_23_isComplete_: bool
                        d_23_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if (d_23_isComplete_) and ((len(currentConstrainedOut)) > (0)):
                            d_24_closedGenerated_: _dafny.Seq
                            d_25_closedInside_: bool
                            d_26_closedCurrent_: _dafny.Seq
                            out17_: _dafny.Seq
                            out18_: bool
                            out19_: _dafny.Seq
                            out17_, out18_, out19_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_24_closedGenerated_ = out17_
                            d_25_closedInside_ = out18_
                            d_26_closedCurrent_ = out19_
                            generated = d_24_closedGenerated_
                            insideConstrainedOut = d_25_closedInside_
                            currentConstrainedOut = d_26_closedCurrent_
                            d_2_suffix_ = _dafny.SeqWithoutIsStrInference((generated)[len(generatedPrefix)::])
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_27_narrow_: bool
                            out20_: bool
                            out20_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                            d_27_narrow_ = out20_
                            if (d_27_narrow_) and ((len(currentConstrainedOut)) > (0)):
                                d_28_repairedGenerated_: _dafny.Seq
                                d_29_repairedCurrent_: _dafny.Seq
                                out21_: _dafny.Seq
                                out22_: _dafny.Seq
                                out21_, out22_ = (d_0_helpers_).RollbackConstrainedSpan(parser, generated, currentConstrainedOut)
                                d_28_repairedGenerated_ = out21_
                                d_29_repairedCurrent_ = out22_
                                generated = d_28_repairedGenerated_
                                currentConstrainedOut = d_29_repairedCurrent_
                                insideConstrainedOut = True
                                d_2_suffix_ = _dafny.SeqWithoutIsStrInference((generated)[len(generatedPrefix)::])
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                d_30_incompleteNow_: bool
                                d_30_incompleteNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                                if d_30_incompleteNow_:
                                    d_31_closedGenerated2_: _dafny.Seq
                                    d_32_closedInside2_: bool
                                    d_33_closedCurrent2_: _dafny.Seq
                                    out23_: _dafny.Seq
                                    out24_: bool
                                    out25_: _dafny.Seq
                                    out23_, out24_, out25_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_31_closedGenerated2_ = out23_
                                    d_32_closedInside2_ = out24_
                                    d_33_closedCurrent2_ = out25_
                                    generated = d_31_closedGenerated2_
                                    insideConstrainedOut = d_32_closedInside2_
                                    currentConstrainedOut = d_33_closedCurrent2_
                                    d_2_suffix_ = _dafny.SeqWithoutIsStrInference((generated)[len(generatedPrefix)::])
                                    d_1_steps_ = (d_1_steps_) + (1)
                                elif True:
                                    d_34_next2_: _dafny.Seq
                                    out26_: _dafny.Seq
                                    out26_ = (d_0_helpers_).ConstrainedStep(lm, parser, prompt, currentConstrainedOut, eosToken)
                                    d_34_next2_ = out26_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    if (d_34_next2_) == (eosToken):
                                        pass
                                    elif True:
                                        d_35_valid2_: bool
                                        out27_: bool
                                        out27_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_34_next2_)
                                        d_35_valid2_ = out27_
                                        if d_35_valid2_:
                                            d_36_appendedGenerated_: _dafny.Seq
                                            d_37_appendedInside_: bool
                                            d_38_appendedCurrent_: _dafny.Seq
                                            out28_: _dafny.Seq
                                            out29_: bool
                                            out30_: _dafny.Seq
                                            out28_, out29_, out30_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_34_next2_)
                                            d_36_appendedGenerated_ = out28_
                                            d_37_appendedInside_ = out29_
                                            d_38_appendedCurrent_ = out30_
                                            generated = d_36_appendedGenerated_
                                            insideConstrainedOut = d_37_appendedInside_
                                            currentConstrainedOut = d_38_appendedCurrent_
                                            d_2_suffix_ = _dafny.SeqWithoutIsStrInference((generated)[len(generatedPrefix)::])
                                        elif True:
                                            if (len(currentConstrainedOut)) > (0):
                                                d_39_repairedGenerated2_: _dafny.Seq
                                                d_40_repairedCurrent2_: _dafny.Seq
                                                out31_: _dafny.Seq
                                                out32_: _dafny.Seq
                                                out31_, out32_ = (d_0_helpers_).RollbackConstrainedSpan(parser, generated, currentConstrainedOut)
                                                d_39_repairedGenerated2_ = out31_
                                                d_40_repairedCurrent2_ = out32_
                                                generated = d_39_repairedGenerated2_
                                                currentConstrainedOut = d_40_repairedCurrent2_
                                                insideConstrainedOut = True
                                                d_2_suffix_ = _dafny.SeqWithoutIsStrInference((generated)[len(generatedPrefix)::])
                                    if (d_34_next2_) == (eosToken):
                                        raise _dafny.Break("0")
                    pass
            pass
        cost = d_0_helpers_.cost
        return generated, insideConstrainedOut, currentConstrainedOut, cost

