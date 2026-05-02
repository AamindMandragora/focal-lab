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
                        d_2_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_2_next_ = out0_
                        if (d_2_next_) == (eosToken):
                            d_1_steps_ = (d_1_steps_) + (1)
                            (d_0_helpers_).cost = d_1_steps_
                            raise _dafny.Break("0")
                        elif True:
                            d_3_shouldOpen_: bool
                            d_3_shouldOpen_ = ((((((((((((((((VerifiedDecoderAgent.default__.Contains(d_2_next_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))) or (VerifiedDecoderAgent.default__.Contains(d_2_next_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<"))))) or (VerifiedDecoderAgent.default__.Contains(d_2_next_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "0"))))) or (VerifiedDecoderAgent.default__.Contains(d_2_next_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "1"))))) or (VerifiedDecoderAgent.default__.Contains(d_2_next_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "2"))))) or (VerifiedDecoderAgent.default__.Contains(d_2_next_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "3"))))) or (VerifiedDecoderAgent.default__.Contains(d_2_next_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "4"))))) or (VerifiedDecoderAgent.default__.Contains(d_2_next_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "5"))))) or (VerifiedDecoderAgent.default__.Contains(d_2_next_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "6"))))) or (VerifiedDecoderAgent.default__.Contains(d_2_next_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "7"))))) or (VerifiedDecoderAgent.default__.Contains(d_2_next_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "8"))))) or (VerifiedDecoderAgent.default__.Contains(d_2_next_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "9"))))) or (VerifiedDecoderAgent.default__.Contains(d_2_next_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "+"))))) or (VerifiedDecoderAgent.default__.Contains(d_2_next_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "-"))))) or (VerifiedDecoderAgent.default__.Contains(d_2_next_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "*"))))) or (VerifiedDecoderAgent.default__.Contains(d_2_next_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "/"))))) or (VerifiedDecoderAgent.default__.Contains(d_2_next_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "="))))
                            if d_3_shouldOpen_:
                                d_4_openedGenerated_: _dafny.Seq
                                d_5_openedInside_: bool
                                d_6_openedCurrent_: _dafny.Seq
                                out1_: _dafny.Seq
                                out2_: bool
                                out3_: _dafny.Seq
                                out1_, out2_, out3_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                d_4_openedGenerated_ = out1_
                                d_5_openedInside_ = out2_
                                d_6_openedCurrent_ = out3_
                                generated = d_4_openedGenerated_
                                insideConstrainedOut = d_5_openedInside_
                                currentConstrainedOut = d_6_openedCurrent_
                                d_1_steps_ = (d_1_steps_) + (1)
                                (d_0_helpers_).cost = d_1_steps_
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_2_next_]))
                                insideConstrainedOut = False
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                d_1_steps_ = (d_1_steps_) + (1)
                                (d_0_helpers_).cost = d_1_steps_
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
                            d_1_steps_ = (d_1_steps_) + (1)
                            (d_0_helpers_).cost = d_1_steps_
                        elif True:
                            d_11_validCount_: int
                            out7_: int
                            out7_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                            d_11_validCount_ = out7_
                            d_12_narrow_: bool
                            out8_: bool
                            out8_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                            d_12_narrow_ = out8_
                            if ((d_11_validCount_) == (0)) or (d_12_narrow_):
                                d_13_rolled_: _dafny.Seq
                                out9_: _dafny.Seq
                                out9_ = VerifiedDecoderAgent.CSDHelpers.RollbackToValidPrefix(parser, currentConstrainedOut)
                                d_13_rolled_ = out9_
                                generated = (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])) + (d_13_rolled_)
                                currentConstrainedOut = d_13_rolled_
                                d_14_rolledComplete_: bool
                                d_14_rolledComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                                if d_14_rolledComplete_:
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
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    (d_0_helpers_).cost = d_1_steps_
                                elif True:
                                    d_18_constrainedPrompt_: _dafny.Seq
                                    d_18_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                    d_19_next2_: _dafny.Seq
                                    out13_: _dafny.Seq
                                    out13_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_18_constrainedPrompt_, currentConstrainedOut, eosToken)
                                    d_19_next2_ = out13_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    (d_0_helpers_).cost = d_1_steps_
                                    if (d_19_next2_) == (eosToken):
                                        raise _dafny.Break("0")
                                    elif True:
                                        d_20_appendedGenerated2_: _dafny.Seq
                                        d_21_appendedInside2_: bool
                                        d_22_appendedCurrent2_: _dafny.Seq
                                        out14_: _dafny.Seq
                                        out15_: bool
                                        out16_: _dafny.Seq
                                        out14_, out15_, out16_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_19_next2_)
                                        d_20_appendedGenerated2_ = out14_
                                        d_21_appendedInside2_ = out15_
                                        d_22_appendedCurrent2_ = out16_
                                        generated = d_20_appendedGenerated2_
                                        insideConstrainedOut = d_21_appendedInside2_
                                        currentConstrainedOut = d_22_appendedCurrent2_
                            elif True:
                                d_23_constrainedPrompt2_: _dafny.Seq
                                d_23_constrainedPrompt2_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_24_next3_: _dafny.Seq
                                out17_: _dafny.Seq
                                out17_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_23_constrainedPrompt2_, currentConstrainedOut, eosToken)
                                d_24_next3_ = out17_
                                d_1_steps_ = (d_1_steps_) + (1)
                                (d_0_helpers_).cost = d_1_steps_
                                if (d_24_next3_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_25_appendedGenerated3_: _dafny.Seq
                                    d_26_appendedInside3_: bool
                                    d_27_appendedCurrent3_: _dafny.Seq
                                    out18_: _dafny.Seq
                                    out19_: bool
                                    out20_: _dafny.Seq
                                    out18_, out19_, out20_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_24_next3_)
                                    d_25_appendedGenerated3_ = out18_
                                    d_26_appendedInside3_ = out19_
                                    d_27_appendedCurrent3_ = out20_
                                    generated = d_25_appendedGenerated3_
                                    insideConstrainedOut = d_26_appendedInside3_
                                    currentConstrainedOut = d_27_appendedCurrent3_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

