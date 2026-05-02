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
        d_2_shouldOpen_: bool
        d_2_shouldOpen_ = False
        while (d_1_steps_) < (maxSteps):
            if not(insideConstrainedOut):
                if (d_2_shouldOpen_) and (((d_1_steps_) + (1)) < (maxSteps)):
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
                    d_2_shouldOpen_ = False
                    d_1_steps_ = (d_1_steps_) + (1)
                elif True:
                    d_6_next_: _dafny.Seq
                    out3_: _dafny.Seq
                    out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                    d_6_next_ = out3_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if (d_6_next_) == (eosToken):
                        d_2_shouldOpen_ = False
                    elif True:
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_6_next_]))
                        if ((((((((((d_6_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")))) or ((d_6_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ":"))))) or (VerifiedDecoderAgent.default__.Contains(d_6_next_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "calculate"))))) or (VerifiedDecoderAgent.default__.Contains(d_6_next_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "compute"))))) or (VerifiedDecoderAgent.default__.Contains(d_6_next_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "total"))))) or (VerifiedDecoderAgent.default__.Contains(d_6_next_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "sum"))))) or (VerifiedDecoderAgent.default__.Contains(d_6_next_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "plus"))))) or (VerifiedDecoderAgent.default__.Contains(d_6_next_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "minus"))))) or (VerifiedDecoderAgent.default__.Contains(d_6_next_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "times"))))) or (VerifiedDecoderAgent.default__.Contains(d_6_next_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "divide")))):
                            d_2_shouldOpen_ = True
                        elif (d_6_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))):
                            d_2_shouldOpen_ = False
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
                    d_2_shouldOpen_ = False
                    d_1_steps_ = (d_1_steps_) + (1)
                elif True:
                    d_11_constrainedPrompt_: _dafny.Seq
                    d_11_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                    d_12_narrow_: bool
                    out7_: bool
                    out7_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                    d_12_narrow_ = out7_
                    if d_12_narrow_:
                        d_13_stablePrefix_: _dafny.Seq
                        d_13_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_14_repairedGenerated_: _dafny.Seq
                        d_15_repairedCurrent_: _dafny.Seq
                        out8_: _dafny.Seq
                        out9_: _dafny.Seq
                        out8_, out9_ = (d_0_helpers_).RollbackConstrainedSpan(parser, d_13_stablePrefix_, generated, currentConstrainedOut)
                        d_14_repairedGenerated_ = out8_
                        d_15_repairedCurrent_ = out9_
                        generated = d_14_repairedGenerated_
                        currentConstrainedOut = d_15_repairedCurrent_
                        if (parser).IsValidPrefix(currentConstrainedOut):
                            insideConstrainedOut = True
                        elif True:
                            insideConstrainedOut = False
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        d_16_next_: _dafny.Seq
                        d_17_isValid_: bool
                        out10_: _dafny.Seq
                        out11_: bool
                        out10_, out11_ = (d_0_helpers_).SoftConstrainedStep(lm, parser, d_11_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('2e0'), eosToken)
                        d_16_next_ = out10_
                        d_17_isValid_ = out11_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_16_next_) == (eosToken):
                            pass
                        elif True:
                            if d_17_isValid_:
                                d_18_validNext_: bool
                                out12_: bool
                                out12_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_16_next_)
                                d_18_validNext_ = out12_
                                if d_18_validNext_:
                                    d_19_appendedGenerated_: _dafny.Seq
                                    d_20_appendedInside_: bool
                                    d_21_appendedCurrent_: _dafny.Seq
                                    out13_: _dafny.Seq
                                    out14_: bool
                                    out15_: _dafny.Seq
                                    out13_, out14_, out15_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_16_next_)
                                    d_19_appendedGenerated_ = out13_
                                    d_20_appendedInside_ = out14_
                                    d_21_appendedCurrent_ = out15_
                                    generated = d_19_appendedGenerated_
                                    insideConstrainedOut = d_20_appendedInside_
                                    currentConstrainedOut = d_21_appendedCurrent_
                                elif True:
                                    d_22_stablePrefix2_: _dafny.Seq
                                    d_22_stablePrefix2_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                    d_23_repairedGenerated2_: _dafny.Seq
                                    d_24_repairedCurrent2_: _dafny.Seq
                                    out16_: _dafny.Seq
                                    out17_: _dafny.Seq
                                    out16_, out17_ = (d_0_helpers_).RollbackConstrainedSpan(parser, d_22_stablePrefix2_, generated, currentConstrainedOut)
                                    d_23_repairedGenerated2_ = out16_
                                    d_24_repairedCurrent2_ = out17_
                                    generated = d_23_repairedGenerated2_
                                    currentConstrainedOut = d_24_repairedCurrent2_
                                    if (parser).IsValidPrefix(currentConstrainedOut):
                                        insideConstrainedOut = True
                                    elif True:
                                        insideConstrainedOut = False
                                        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                            elif True:
                                d_25_stablePrefix3_: _dafny.Seq
                                d_25_stablePrefix3_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                d_26_repairedGenerated3_: _dafny.Seq
                                d_27_repairedCurrent3_: _dafny.Seq
                                out18_: _dafny.Seq
                                out19_: _dafny.Seq
                                out18_, out19_ = (d_0_helpers_).RollbackConstrainedSpan(parser, d_25_stablePrefix3_, generated, currentConstrainedOut)
                                d_26_repairedGenerated3_ = out18_
                                d_27_repairedCurrent3_ = out19_
                                generated = d_26_repairedGenerated3_
                                currentConstrainedOut = d_27_repairedCurrent3_
                                if (parser).IsValidPrefix(currentConstrainedOut):
                                    insideConstrainedOut = True
                                elif True:
                                    insideConstrainedOut = False
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
        cost = d_0_helpers_.cost
        return generated, insideConstrainedOut, currentConstrainedOut, cost

