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
    def MyCSDStrategy(lm, parser, prompt, generatedPrefix, insideConstrained, currentConstrained, maxSteps, stepTokenBudget, eosToken):
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
        generated = generatedPrefix
        insideConstrainedOut = insideConstrained
        currentConstrainedOut = currentConstrained
        cost = 0
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_openTok_: _dafny.Seq
        d_2_openTok_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))
        d_3_minCloseLen_: int
        d_3_minCloseLen_ = 24
        d_4_nearEndChoices_: int
        d_4_nearEndChoices_ = 1
        d_5_candidateCap_: int
        d_5_candidateCap_ = 6
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_6_remaining_: int
                        d_6_remaining_ = (maxSteps) - (d_1_steps_)
                        if (d_6_remaining_) == (0):
                            raise _dafny.Break("0")
                        elif True:
                            d_7_chunkGenerated_: _dafny.Seq
                            d_8_stoppedOnOpenSpan_: bool
                            d_9_stoppedOnEos_: bool
                            d_10_stepsUsed_: int
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: bool
                            out3_: int
                            out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_6_remaining_, d_2_openTok_, eosToken)
                            d_7_chunkGenerated_ = out0_
                            d_8_stoppedOnOpenSpan_ = out1_
                            d_9_stoppedOnEos_ = out2_
                            d_10_stepsUsed_ = out3_
                            generated = d_7_chunkGenerated_
                            d_1_steps_ = (d_1_steps_) + (d_10_stepsUsed_)
                            if d_9_stoppedOnEos_:
                                raise _dafny.Break("0")
                            elif True:
                                if (d_8_stoppedOnOpenSpan_) and ((d_1_steps_) < (maxSteps)):
                                    d_11_openedGenerated_: _dafny.Seq
                                    d_12_openedInside_: bool
                                    d_13_openedCurrent_: _dafny.Seq
                                    out4_: _dafny.Seq
                                    out5_: bool
                                    out6_: _dafny.Seq
                                    out4_, out5_, out6_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                    d_11_openedGenerated_ = out4_
                                    d_12_openedInside_ = out5_
                                    d_13_openedCurrent_ = out6_
                                    generated = d_11_openedGenerated_
                                    insideConstrainedOut = d_12_openedInside_
                                    currentConstrainedOut = d_13_openedCurrent_
                                    d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_14_deadEnd_: bool
                        out7_: bool
                        out7_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 0)
                        d_14_deadEnd_ = out7_
                        if d_14_deadEnd_:
                            d_15_stablePrefix_: _dafny.Seq
                            d_15_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_16_repairedGenerated_: _dafny.Seq
                            d_17_repairedCurrent_: _dafny.Seq
                            out8_: _dafny.Seq
                            out9_: _dafny.Seq
                            out8_, out9_ = (d_0_helpers_).RollbackConstrainedSpan(parser, d_15_stablePrefix_, generated, currentConstrainedOut)
                            d_16_repairedGenerated_ = out8_
                            d_17_repairedCurrent_ = out9_
                            generated = d_16_repairedGenerated_
                            currentConstrainedOut = d_17_repairedCurrent_
                        elif True:
                            d_18_remaining2_: int
                            d_18_remaining2_ = (maxSteps) - (d_1_steps_)
                            if (d_18_remaining2_) == (0):
                                raise _dafny.Break("0")
                            elif True:
                                d_19_completeNow_: bool
                                d_19_completeNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                                d_20_validCount_: int
                                out10_: int
                                out10_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                                d_20_validCount_ = out10_
                                if (d_19_completeNow_) and ((((d_3_minCloseLen_) <= (len(currentConstrainedOut))) or ((d_20_validCount_) <= (d_4_nearEndChoices_))) or ((d_18_remaining2_) == (1))):
                                    d_21_closedGenerated_: _dafny.Seq
                                    d_22_closedInside_: bool
                                    d_23_closedCurrent_: _dafny.Seq
                                    out11_: _dafny.Seq
                                    out12_: bool
                                    out13_: _dafny.Seq
                                    out11_, out12_, out13_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_21_closedGenerated_ = out11_
                                    d_22_closedInside_ = out12_
                                    d_23_closedCurrent_ = out13_
                                    generated = d_21_closedGenerated_
                                    insideConstrainedOut = d_22_closedInside_
                                    currentConstrainedOut = d_23_closedCurrent_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    raise _dafny.Break("0")
                                elif True:
                                    if d_19_completeNow_:
                                        d_24_narrowed_: bool
                                        out14_: bool
                                        out14_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 2)
                                        d_24_narrowed_ = out14_
                                        if (d_24_narrowed_) and ((d_18_remaining2_) == (1)):
                                            d_25_closedGenerated2_: _dafny.Seq
                                            d_26_closedInside2_: bool
                                            d_27_closedCurrent2_: _dafny.Seq
                                            out15_: _dafny.Seq
                                            out16_: bool
                                            out17_: _dafny.Seq
                                            out15_, out16_, out17_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                            d_25_closedGenerated2_ = out15_
                                            d_26_closedInside2_ = out16_
                                            d_27_closedCurrent2_ = out17_
                                            generated = d_25_closedGenerated2_
                                            insideConstrainedOut = d_26_closedInside2_
                                            currentConstrainedOut = d_27_closedCurrent2_
                                            d_1_steps_ = (d_1_steps_) + (1)
                                            raise _dafny.Break("0")
                                        elif True:
                                            raise _dafny.Break("0")
                                    elif True:
                                        (lm).GenerateLogits((prompt) + (generated))
                                        d_28_candidates_: _dafny.Seq
                                        out18_: _dafny.Seq
                                        out18_ = (d_0_helpers_).TopValidCandidates(lm, parser, prompt, currentConstrainedOut, d_5_candidateCap_, eosToken)
                                        d_28_candidates_ = out18_
                                        (d_0_helpers_).BoostTokenLogits(lm, d_28_candidates_, _dafny.BigRational('8e0'))
                                        (d_0_helpers_).ScaleAllLogits(lm, _dafny.BigRational('125e-2'))
                                        d_29_next_: _dafny.Seq
                                        out19_: _dafny.Seq
                                        out19_ = (lm).ChooseNextToken()
                                        d_29_next_ = out19_
                                        (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                                        if (d_29_next_) == (eosToken):
                                            raise _dafny.Break("0")
                                        elif True:
                                            d_30_validNext_: bool
                                            out20_: bool
                                            out20_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_29_next_)
                                            d_30_validNext_ = out20_
                                            if d_30_validNext_:
                                                d_31_appendedGenerated_: _dafny.Seq
                                                d_32_appendedInside_: bool
                                                d_33_appendedCurrent_: _dafny.Seq
                                                out21_: _dafny.Seq
                                                out22_: bool
                                                out23_: _dafny.Seq
                                                out21_, out22_, out23_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_29_next_)
                                                d_31_appendedGenerated_ = out21_
                                                d_32_appendedInside_ = out22_
                                                d_33_appendedCurrent_ = out23_
                                                generated = d_31_appendedGenerated_
                                                insideConstrainedOut = d_32_appendedInside_
                                                currentConstrainedOut = d_33_appendedCurrent_
                                                d_1_steps_ = (d_1_steps_) + (1)
                                            elif True:
                                                d_34_fallback_: _dafny.Seq
                                                out24_: _dafny.Seq
                                                out24_ = (d_0_helpers_).ConstrainedStep(lm, parser, prompt, currentConstrainedOut, eosToken)
                                                d_34_fallback_ = out24_
                                                if (d_34_fallback_) == (eosToken):
                                                    raise _dafny.Break("0")
                                                elif True:
                                                    d_35_appendedGenerated2_: _dafny.Seq
                                                    d_36_appendedInside2_: bool
                                                    d_37_appendedCurrent2_: _dafny.Seq
                                                    out25_: _dafny.Seq
                                                    out26_: bool
                                                    out27_: _dafny.Seq
                                                    out25_, out26_, out27_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_34_fallback_)
                                                    d_35_appendedGenerated2_ = out25_
                                                    d_36_appendedInside2_ = out26_
                                                    d_37_appendedCurrent2_ = out27_
                                                    generated = d_35_appendedGenerated2_
                                                    insideConstrainedOut = d_36_appendedInside2_
                                                    currentConstrainedOut = d_37_appendedCurrent2_
                                                    d_1_steps_ = (d_1_steps_) + (1)
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

