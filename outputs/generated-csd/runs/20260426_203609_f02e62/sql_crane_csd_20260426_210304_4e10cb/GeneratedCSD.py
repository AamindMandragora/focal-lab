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
                    if not(insideConstrainedOut):
                        d_2_openedGenerated_: _dafny.Seq
                        d_3_openedInside_: bool
                        d_4_openedCurrent_: _dafny.Seq
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: _dafny.Seq
                        out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                        d_2_openedGenerated_ = out0_
                        d_3_openedInside_ = out1_
                        d_4_openedCurrent_ = out2_
                        generated = d_2_openedGenerated_
                        insideConstrainedOut = d_3_openedInside_
                        currentConstrainedOut = d_4_openedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        if ((d_1_steps_) + (1)) == (maxSteps):
                            d_5_completeLast_: bool
                            d_5_completeLast_ = (parser).IsCompletePrefix(currentConstrainedOut)
                            if d_5_completeLast_:
                                d_6_closedGeneratedLast_: _dafny.Seq
                                d_7_closedInsideLast_: bool
                                d_8_closedCurrentLast_: _dafny.Seq
                                out3_: _dafny.Seq
                                out4_: bool
                                out5_: _dafny.Seq
                                out3_, out4_, out5_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_6_closedGeneratedLast_ = out3_
                                d_7_closedInsideLast_ = out4_
                                d_8_closedCurrentLast_ = out5_
                                generated = d_6_closedGeneratedLast_
                                insideConstrainedOut = d_7_closedInsideLast_
                                currentConstrainedOut = d_8_closedCurrentLast_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                raise _dafny.Break("0")
                        elif True:
                            d_9_narrow_: bool
                            out6_: bool
                            out6_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                            d_9_narrow_ = out6_
                            if not(d_9_narrow_):
                                d_10_rolled_: _dafny.Seq
                                out7_: _dafny.Seq
                                out7_ = VerifiedDecoderAgent.CSDHelpers.RollbackToValidPrefix(parser, currentConstrainedOut)
                                d_10_rolled_ = out7_
                                d_11_stablePrefix_: _dafny.Seq
                                d_11_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                d_12_rolledGenerated_: _dafny.Seq
                                d_13_rolledCurrent_: _dafny.Seq
                                out8_: _dafny.Seq
                                out9_: _dafny.Seq
                                out8_, out9_ = (d_0_helpers_).RollbackConstrainedSpan(parser, d_11_stablePrefix_, generated, currentConstrainedOut)
                                d_12_rolledGenerated_ = out8_
                                d_13_rolledCurrent_ = out9_
                                generated = d_12_rolledGenerated_
                                currentConstrainedOut = d_13_rolledCurrent_
                                if (parser).IsCompletePrefix(currentConstrainedOut):
                                    d_14_closedGeneratedRb_: _dafny.Seq
                                    d_15_closedInsideRb_: bool
                                    d_16_closedCurrentRb_: _dafny.Seq
                                    out10_: _dafny.Seq
                                    out11_: bool
                                    out12_: _dafny.Seq
                                    out10_, out11_, out12_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_14_closedGeneratedRb_ = out10_
                                    d_15_closedInsideRb_ = out11_
                                    d_16_closedCurrentRb_ = out12_
                                    generated = d_14_closedGeneratedRb_
                                    insideConstrainedOut = d_15_closedInsideRb_
                                    currentConstrainedOut = d_16_closedCurrentRb_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                d_17_isComplete_: bool
                                d_17_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                                if d_17_isComplete_:
                                    d_18_validCount_: int
                                    out13_: int
                                    out13_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                                    d_18_validCount_ = out13_
                                    if (d_18_validCount_) == (0):
                                        d_19_closedGenerated_: _dafny.Seq
                                        d_20_closedInside_: bool
                                        d_21_closedCurrent_: _dafny.Seq
                                        out14_: _dafny.Seq
                                        out15_: bool
                                        out16_: _dafny.Seq
                                        out14_, out15_, out16_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                        d_19_closedGenerated_ = out14_
                                        d_20_closedInside_ = out15_
                                        d_21_closedCurrent_ = out16_
                                        generated = d_19_closedGenerated_
                                        insideConstrainedOut = d_20_closedInside_
                                        currentConstrainedOut = d_21_closedCurrent_
                                        d_1_steps_ = (d_1_steps_) + (1)
                                    elif True:
                                        d_22_constrainedPrompt_: _dafny.Seq
                                        d_22_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                        d_23_next_: _dafny.Seq
                                        out17_: _dafny.Seq
                                        out17_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_22_constrainedPrompt_, currentConstrainedOut, eosToken)
                                        d_23_next_ = out17_
                                        d_1_steps_ = (d_1_steps_) + (1)
                                        if (d_23_next_) == (eosToken):
                                            raise _dafny.Break("0")
                                        elif True:
                                            d_24_appendedGenerated_: _dafny.Seq
                                            d_25_appendedInside_: bool
                                            d_26_appendedCurrent_: _dafny.Seq
                                            out18_: _dafny.Seq
                                            out19_: bool
                                            out20_: _dafny.Seq
                                            out18_, out19_, out20_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_23_next_)
                                            d_24_appendedGenerated_ = out18_
                                            d_25_appendedInside_ = out19_
                                            d_26_appendedCurrent_ = out20_
                                            generated = d_24_appendedGenerated_
                                            insideConstrainedOut = d_25_appendedInside_
                                            currentConstrainedOut = d_26_appendedCurrent_
                                elif True:
                                    d_27_constrainedPrompt2_: _dafny.Seq
                                    d_27_constrainedPrompt2_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                    d_28_next2_: _dafny.Seq
                                    out21_: _dafny.Seq
                                    out21_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_27_constrainedPrompt2_, currentConstrainedOut, eosToken)
                                    d_28_next2_ = out21_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    if (d_28_next2_) == (eosToken):
                                        raise _dafny.Break("0")
                                    elif True:
                                        d_29_appendedGenerated2_: _dafny.Seq
                                        d_30_appendedInside2_: bool
                                        d_31_appendedCurrent2_: _dafny.Seq
                                        out22_: _dafny.Seq
                                        out23_: bool
                                        out24_: _dafny.Seq
                                        out22_, out23_, out24_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_28_next2_)
                                        d_29_appendedGenerated2_ = out22_
                                        d_30_appendedInside2_ = out23_
                                        d_31_appendedCurrent2_ = out24_
                                        generated = d_29_appendedGenerated2_
                                        insideConstrainedOut = d_30_appendedInside2_
                                        currentConstrainedOut = d_31_appendedCurrent2_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

