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
        if True:
            generated = generatedPrefix
            insideConstrainedOut = insideConstrained
            currentConstrainedOut = currentConstrained
            cost = 0
            d_1_steps_: int
            d_1_steps_ = 0
            d_2_openedOnce_: bool
            d_2_openedOnce_ = insideConstrained
            d_3_didInitialUnconstrained_: bool
            d_3_didInitialUnconstrained_ = False
            with _dafny.label("0"):
                while (d_1_steps_) < (maxSteps):
                    with _dafny.c_label("0"):
                        if (not(d_3_didInitialUnconstrained_)) and (not(insideConstrainedOut)):
                            d_4_next0_: _dafny.Seq
                            out0_: _dafny.Seq
                            out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_4_next0_ = out0_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_3_didInitialUnconstrained_ = True
                            if (d_4_next0_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_4_next0_]))
                        elif True:
                            if (not(insideConstrainedOut)) and (not(d_2_openedOnce_)):
                                d_5_openedGenerated_: _dafny.Seq
                                d_6_openedInside_: bool
                                d_7_openedCurrent_: _dafny.Seq
                                out1_: _dafny.Seq
                                out2_: bool
                                out3_: _dafny.Seq
                                out1_, out2_, out3_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                d_5_openedGenerated_ = out1_
                                d_6_openedInside_ = out2_
                                d_7_openedCurrent_ = out3_
                                generated = d_5_openedGenerated_
                                insideConstrainedOut = d_6_openedInside_
                                currentConstrainedOut = d_7_openedCurrent_
                                d_2_openedOnce_ = True
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                if insideConstrainedOut:
                                    d_8_completeNow_: bool
                                    d_8_completeNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                                    if d_8_completeNow_:
                                        d_9_mustUseLastStep_: bool
                                        d_9_mustUseLastStep_ = ((d_1_steps_) + (1)) == (maxSteps)
                                        d_10_validCountNow_: int
                                        out4_: int
                                        out4_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                                        d_10_validCountNow_ = out4_
                                        if (d_9_mustUseLastStep_) or ((d_10_validCountNow_) == (0)):
                                            d_11_closedGenerated_: _dafny.Seq
                                            d_12_closedInside_: bool
                                            d_13_closedCurrent_: _dafny.Seq
                                            out5_: _dafny.Seq
                                            out6_: bool
                                            out7_: _dafny.Seq
                                            out5_, out6_, out7_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                            d_11_closedGenerated_ = out5_
                                            d_12_closedInside_ = out6_
                                            d_13_closedCurrent_ = out7_
                                            generated = d_11_closedGenerated_
                                            insideConstrainedOut = d_12_closedInside_
                                            currentConstrainedOut = d_13_closedCurrent_
                                            d_1_steps_ = (d_1_steps_) + (1)
                                        elif True:
                                            d_14_constrainedPrompt0_: _dafny.Seq
                                            d_14_constrainedPrompt0_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                            d_15_next1_: _dafny.Seq
                                            out8_: _dafny.Seq
                                            out8_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_14_constrainedPrompt0_, currentConstrainedOut, eosToken)
                                            d_15_next1_ = out8_
                                            d_1_steps_ = (d_1_steps_) + (1)
                                            if (d_15_next1_) == (eosToken):
                                                d_16_closedGeneratedE_: _dafny.Seq
                                                d_17_closedInsideE_: bool
                                                d_18_closedCurrentE_: _dafny.Seq
                                                out9_: _dafny.Seq
                                                out10_: bool
                                                out11_: _dafny.Seq
                                                out9_, out10_, out11_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                                d_16_closedGeneratedE_ = out9_
                                                d_17_closedInsideE_ = out10_
                                                d_18_closedCurrentE_ = out11_
                                                generated = d_16_closedGeneratedE_
                                                insideConstrainedOut = d_17_closedInsideE_
                                                currentConstrainedOut = d_18_closedCurrentE_
                                            elif True:
                                                d_19_stillIncomplete_: bool
                                                d_19_stillIncomplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                                                if d_19_stillIncomplete_:
                                                    d_20_closedGeneratedE2_: _dafny.Seq
                                                    d_21_closedInsideE2_: bool
                                                    d_22_closedCurrentE2_: _dafny.Seq
                                                    out12_: _dafny.Seq
                                                    out13_: bool
                                                    out14_: _dafny.Seq
                                                    out12_, out13_, out14_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                                    d_20_closedGeneratedE2_ = out12_
                                                    d_21_closedInsideE2_ = out13_
                                                    d_22_closedCurrentE2_ = out14_
                                                    generated = d_20_closedGeneratedE2_
                                                    insideConstrainedOut = d_21_closedInsideE2_
                                                    currentConstrainedOut = d_22_closedCurrentE2_
                                                elif True:
                                                    d_23_appendedGenerated1_: _dafny.Seq
                                                    d_24_appendedInside1_: bool
                                                    d_25_appendedCurrent1_: _dafny.Seq
                                                    out15_: _dafny.Seq
                                                    out16_: bool
                                                    out17_: _dafny.Seq
                                                    out15_, out16_, out17_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_15_next1_)
                                                    d_23_appendedGenerated1_ = out15_
                                                    d_24_appendedInside1_ = out16_
                                                    d_25_appendedCurrent1_ = out17_
                                                    generated = d_23_appendedGenerated1_
                                                    insideConstrainedOut = d_24_appendedInside1_
                                                    currentConstrainedOut = d_25_appendedCurrent1_
                                    elif True:
                                        d_26_narrow_: bool
                                        out18_: bool
                                        out18_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                                        d_26_narrow_ = out18_
                                        if not(d_26_narrow_):
                                            d_27_rolled_: _dafny.Seq
                                            out19_: _dafny.Seq
                                            out19_ = VerifiedDecoderAgent.CSDHelpers.RollbackToValidPrefix(parser, currentConstrainedOut)
                                            d_27_rolled_ = out19_
                                            d_28_stablePrefix_: _dafny.Seq
                                            d_28_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                            d_29_rolledGenerated_: _dafny.Seq
                                            d_30_rolledCurrent_: _dafny.Seq
                                            out20_: _dafny.Seq
                                            out21_: _dafny.Seq
                                            out20_, out21_ = (d_0_helpers_).RollbackConstrainedSpan(parser, d_28_stablePrefix_, generated, currentConstrainedOut)
                                            d_29_rolledGenerated_ = out20_
                                            d_30_rolledCurrent_ = out21_
                                            generated = d_29_rolledGenerated_
                                            currentConstrainedOut = d_30_rolledCurrent_
                                            d_1_steps_ = (d_1_steps_) + (1)
                                            d_31_completeAfterRollback_: bool
                                            d_31_completeAfterRollback_ = (parser).IsCompletePrefix(currentConstrainedOut)
                                            if (d_31_completeAfterRollback_) and ((d_1_steps_) < (maxSteps)):
                                                d_32_closedGeneratedRb_: _dafny.Seq
                                                d_33_closedInsideRb_: bool
                                                d_34_closedCurrentRb_: _dafny.Seq
                                                out22_: _dafny.Seq
                                                out23_: bool
                                                out24_: _dafny.Seq
                                                out22_, out23_, out24_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                                d_32_closedGeneratedRb_ = out22_
                                                d_33_closedInsideRb_ = out23_
                                                d_34_closedCurrentRb_ = out24_
                                                generated = d_32_closedGeneratedRb_
                                                insideConstrainedOut = d_33_closedInsideRb_
                                                currentConstrainedOut = d_34_closedCurrentRb_
                                                d_1_steps_ = (d_1_steps_) + (1)
                                        elif True:
                                            d_35_constrainedPrompt_: _dafny.Seq
                                            d_35_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                            d_36_next_: _dafny.Seq
                                            out25_: _dafny.Seq
                                            out25_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_35_constrainedPrompt_, currentConstrainedOut, eosToken)
                                            d_36_next_ = out25_
                                            d_1_steps_ = (d_1_steps_) + (1)
                                            if (d_36_next_) == (eosToken):
                                                raise _dafny.Break("0")
                                            elif True:
                                                d_37_appendedGenerated_: _dafny.Seq
                                                d_38_appendedInside_: bool
                                                d_39_appendedCurrent_: _dafny.Seq
                                                out26_: _dafny.Seq
                                                out27_: bool
                                                out28_: _dafny.Seq
                                                out26_, out27_, out28_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_36_next_)
                                                d_37_appendedGenerated_ = out26_
                                                d_38_appendedInside_ = out27_
                                                d_39_appendedCurrent_ = out28_
                                                generated = d_37_appendedGenerated_
                                                insideConstrainedOut = d_38_appendedInside_
                                                currentConstrainedOut = d_39_appendedCurrent_
                                elif True:
                                    raise _dafny.Break("0")
                        pass
                pass
            cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

