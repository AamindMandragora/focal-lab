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
        d_3_candidateCap_: int
        d_3_candidateCap_ = 8
        d_4_forcedCloseChoices_: int
        d_4_forcedCloseChoices_ = 1
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_5_remaining_: int
                        d_5_remaining_ = (maxSteps) - (d_1_steps_)
                        if (d_5_remaining_) == (0):
                            raise _dafny.Break("0")
                        elif True:
                            d_6_chunkGenerated_: _dafny.Seq
                            d_7_stoppedOnOpenSpan_: bool
                            d_8_stoppedOnEos_: bool
                            d_9_stepsUsed_: int
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: bool
                            out3_: int
                            out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_5_remaining_, d_2_openTok_, eosToken)
                            d_6_chunkGenerated_ = out0_
                            d_7_stoppedOnOpenSpan_ = out1_
                            d_8_stoppedOnEos_ = out2_
                            d_9_stepsUsed_ = out3_
                            generated = d_6_chunkGenerated_
                            d_1_steps_ = (d_1_steps_) + (d_9_stepsUsed_)
                            if d_8_stoppedOnEos_:
                                raise _dafny.Break("0")
                            elif True:
                                if (d_7_stoppedOnOpenSpan_) and ((d_1_steps_) < (maxSteps)):
                                    d_10_openedGenerated_: _dafny.Seq
                                    d_11_openedInside_: bool
                                    d_12_openedCurrent_: _dafny.Seq
                                    out4_: _dafny.Seq
                                    out5_: bool
                                    out6_: _dafny.Seq
                                    out4_, out5_, out6_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                    d_10_openedGenerated_ = out4_
                                    d_11_openedInside_ = out5_
                                    d_12_openedCurrent_ = out6_
                                    generated = d_10_openedGenerated_
                                    insideConstrainedOut = d_11_openedInside_
                                    currentConstrainedOut = d_12_openedCurrent_
                                    d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_13_deadEnd_: bool
                        out7_: bool
                        out7_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 0)
                        d_13_deadEnd_ = out7_
                        if d_13_deadEnd_:
                            d_14_stablePrefix_: _dafny.Seq
                            d_14_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_15_repairedGenerated_: _dafny.Seq
                            d_16_repairedCurrent_: _dafny.Seq
                            out8_: _dafny.Seq
                            out9_: _dafny.Seq
                            out8_, out9_ = (d_0_helpers_).RollbackConstrainedSpan(parser, d_14_stablePrefix_, generated, currentConstrainedOut)
                            d_15_repairedGenerated_ = out8_
                            d_16_repairedCurrent_ = out9_
                            generated = d_15_repairedGenerated_
                            currentConstrainedOut = d_16_repairedCurrent_
                        elif True:
                            d_17_remaining2_: int
                            d_17_remaining2_ = (maxSteps) - (d_1_steps_)
                            if (d_17_remaining2_) == (0):
                                raise _dafny.Break("0")
                            elif True:
                                d_18_completeNow_: bool
                                d_18_completeNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                                d_19_validCount_: int
                                out10_: int
                                out10_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                                d_19_validCount_ = out10_
                                if (d_18_completeNow_) and (((d_17_remaining2_) == (1)) or ((d_19_validCount_) <= (d_4_forcedCloseChoices_))):
                                    d_20_closedGenerated_: _dafny.Seq
                                    d_21_closedInside_: bool
                                    d_22_closedCurrent_: _dafny.Seq
                                    out11_: _dafny.Seq
                                    out12_: bool
                                    out13_: _dafny.Seq
                                    out11_, out12_, out13_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_20_closedGenerated_ = out11_
                                    d_21_closedInside_ = out12_
                                    d_22_closedCurrent_ = out13_
                                    generated = d_20_closedGenerated_
                                    insideConstrainedOut = d_21_closedInside_
                                    currentConstrainedOut = d_22_closedCurrent_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    raise _dafny.Break("0")
                                elif True:
                                    if d_18_completeNow_:
                                        raise _dafny.Break("0")
                                    elif True:
                                        (lm).GenerateLogits((prompt) + (generated))
                                        d_23_candidates_: _dafny.Seq
                                        out14_: _dafny.Seq
                                        out14_ = (d_0_helpers_).TopValidCandidates(lm, parser, prompt, currentConstrainedOut, d_3_candidateCap_, eosToken)
                                        d_23_candidates_ = out14_
                                        (d_0_helpers_).BoostTokenLogits(lm, d_23_candidates_, _dafny.BigRational('12e0'))
                                        (d_0_helpers_).ScaleAllLogits(lm, _dafny.BigRational('135e-2'))
                                        d_24_next_: _dafny.Seq
                                        out15_: _dafny.Seq
                                        out15_ = (lm).ChooseNextToken()
                                        d_24_next_ = out15_
                                        (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                                        if (d_24_next_) == (eosToken):
                                            raise _dafny.Break("0")
                                        elif True:
                                            d_25_validNext_: bool
                                            out16_: bool
                                            out16_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_24_next_)
                                            d_25_validNext_ = out16_
                                            if d_25_validNext_:
                                                d_26_appendedGenerated_: _dafny.Seq
                                                d_27_appendedInside_: bool
                                                d_28_appendedCurrent_: _dafny.Seq
                                                out17_: _dafny.Seq
                                                out18_: bool
                                                out19_: _dafny.Seq
                                                out17_, out18_, out19_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_24_next_)
                                                d_26_appendedGenerated_ = out17_
                                                d_27_appendedInside_ = out18_
                                                d_28_appendedCurrent_ = out19_
                                                generated = d_26_appendedGenerated_
                                                insideConstrainedOut = d_27_appendedInside_
                                                currentConstrainedOut = d_28_appendedCurrent_
                                                d_1_steps_ = (d_1_steps_) + (1)
                                            elif True:
                                                d_29_fallback_: _dafny.Seq
                                                out20_: _dafny.Seq
                                                out20_ = (d_0_helpers_).ConstrainedStep(lm, parser, prompt, currentConstrainedOut, eosToken)
                                                d_29_fallback_ = out20_
                                                if (d_29_fallback_) == (eosToken):
                                                    if (d_18_completeNow_) and ((d_1_steps_) < (maxSteps)):
                                                        d_30_closedGenerated2_: _dafny.Seq
                                                        d_31_closedInside2_: bool
                                                        d_32_closedCurrent2_: _dafny.Seq
                                                        out21_: _dafny.Seq
                                                        out22_: bool
                                                        out23_: _dafny.Seq
                                                        out21_, out22_, out23_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                                        d_30_closedGenerated2_ = out21_
                                                        d_31_closedInside2_ = out22_
                                                        d_32_closedCurrent2_ = out23_
                                                        generated = d_30_closedGenerated2_
                                                        insideConstrainedOut = d_31_closedInside2_
                                                        currentConstrainedOut = d_32_closedCurrent2_
                                                        d_1_steps_ = (d_1_steps_) + (1)
                                                        raise _dafny.Break("0")
                                                    elif True:
                                                        raise _dafny.Break("0")
                                                elif True:
                                                    d_33_appendedGenerated2_: _dafny.Seq
                                                    d_34_appendedInside2_: bool
                                                    d_35_appendedCurrent2_: _dafny.Seq
                                                    out24_: _dafny.Seq
                                                    out25_: bool
                                                    out26_: _dafny.Seq
                                                    out24_, out25_, out26_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_29_fallback_)
                                                    d_33_appendedGenerated2_ = out24_
                                                    d_34_appendedInside2_ = out25_
                                                    d_35_appendedCurrent2_ = out26_
                                                    generated = d_33_appendedGenerated2_
                                                    insideConstrainedOut = d_34_appendedInside2_
                                                    currentConstrainedOut = d_35_appendedCurrent2_
                                                    d_1_steps_ = (d_1_steps_) + (1)
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

