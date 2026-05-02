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
        d_3_candidateCap_ = 12
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_4_remaining_: int
                        d_4_remaining_ = (maxSteps) - (d_1_steps_)
                        if (d_4_remaining_) == (0):
                            raise _dafny.Break("0")
                        elif True:
                            d_5_chunkGenerated_: _dafny.Seq
                            d_6_stoppedOnOpenSpan_: bool
                            d_7_stoppedOnEos_: bool
                            d_8_stepsUsed_: int
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: bool
                            out3_: int
                            out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_4_remaining_, d_2_openTok_, eosToken)
                            d_5_chunkGenerated_ = out0_
                            d_6_stoppedOnOpenSpan_ = out1_
                            d_7_stoppedOnEos_ = out2_
                            d_8_stepsUsed_ = out3_
                            generated = d_5_chunkGenerated_
                            d_1_steps_ = (d_1_steps_) + (d_8_stepsUsed_)
                            if d_7_stoppedOnEos_:
                                raise _dafny.Break("0")
                            elif (d_6_stoppedOnOpenSpan_) and ((d_1_steps_) < (maxSteps)):
                                d_9_openedGenerated_: _dafny.Seq
                                d_10_openedInside_: bool
                                d_11_openedCurrent_: _dafny.Seq
                                out4_: _dafny.Seq
                                out5_: bool
                                out6_: _dafny.Seq
                                out4_, out5_, out6_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                d_9_openedGenerated_ = out4_
                                d_10_openedInside_ = out5_
                                d_11_openedCurrent_ = out6_
                                generated = d_9_openedGenerated_
                                insideConstrainedOut = d_10_openedInside_
                                currentConstrainedOut = d_11_openedCurrent_
                                d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_12_deadEnd_: bool
                        out7_: bool
                        out7_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 0)
                        d_12_deadEnd_ = out7_
                        if d_12_deadEnd_:
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
                        elif True:
                            d_16_remaining2_: int
                            d_16_remaining2_ = (maxSteps) - (d_1_steps_)
                            if (d_16_remaining2_) == (0):
                                raise _dafny.Break("0")
                            elif True:
                                d_17_completeNow_: bool
                                d_17_completeNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                                d_18_validCount_: int
                                out10_: int
                                out10_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                                d_18_validCount_ = out10_
                                if (d_17_completeNow_) and ((d_16_remaining2_) == (1)):
                                    d_19_closedGenerated_: _dafny.Seq
                                    d_20_closedInside_: bool
                                    d_21_closedCurrent_: _dafny.Seq
                                    out11_: _dafny.Seq
                                    out12_: bool
                                    out13_: _dafny.Seq
                                    out11_, out12_, out13_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_19_closedGenerated_ = out11_
                                    d_20_closedInside_ = out12_
                                    d_21_closedCurrent_ = out13_
                                    generated = d_19_closedGenerated_
                                    insideConstrainedOut = d_20_closedInside_
                                    currentConstrainedOut = d_21_closedCurrent_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    raise _dafny.Break("0")
                                elif (d_17_completeNow_) and ((d_18_validCount_) == (0)):
                                    d_22_closedGenerated2_: _dafny.Seq
                                    d_23_closedInside2_: bool
                                    d_24_closedCurrent2_: _dafny.Seq
                                    out14_: _dafny.Seq
                                    out15_: bool
                                    out16_: _dafny.Seq
                                    out14_, out15_, out16_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_22_closedGenerated2_ = out14_
                                    d_23_closedInside2_ = out15_
                                    d_24_closedCurrent2_ = out16_
                                    generated = d_22_closedGenerated2_
                                    insideConstrainedOut = d_23_closedInside2_
                                    currentConstrainedOut = d_24_closedCurrent2_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    raise _dafny.Break("0")
                                elif True:
                                    (lm).GenerateLogits((prompt) + (generated))
                                    d_25_candidates_: _dafny.Seq
                                    out17_: _dafny.Seq
                                    out17_ = (d_0_helpers_).TopValidCandidates(lm, parser, prompt, currentConstrainedOut, d_3_candidateCap_, eosToken)
                                    d_25_candidates_ = out17_
                                    (d_0_helpers_).BoostTokenLogits(lm, d_25_candidates_, _dafny.BigRational('18e0'))
                                    (d_0_helpers_).ScaleAllLogits(lm, _dafny.BigRational('15e-1'))
                                    d_26_next_: _dafny.Seq
                                    out18_: _dafny.Seq
                                    out18_ = (lm).ChooseNextToken()
                                    d_26_next_ = out18_
                                    (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                                    if (d_26_next_) == (eosToken):
                                        if (d_17_completeNow_) and ((d_1_steps_) < (maxSteps)):
                                            d_27_closedGenerated3_: _dafny.Seq
                                            d_28_closedInside3_: bool
                                            d_29_closedCurrent3_: _dafny.Seq
                                            out19_: _dafny.Seq
                                            out20_: bool
                                            out21_: _dafny.Seq
                                            out19_, out20_, out21_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                            d_27_closedGenerated3_ = out19_
                                            d_28_closedInside3_ = out20_
                                            d_29_closedCurrent3_ = out21_
                                            generated = d_27_closedGenerated3_
                                            insideConstrainedOut = d_28_closedInside3_
                                            currentConstrainedOut = d_29_closedCurrent3_
                                            d_1_steps_ = (d_1_steps_) + (1)
                                            raise _dafny.Break("0")
                                        elif True:
                                            raise _dafny.Break("0")
                                    elif True:
                                        d_30_validNext_: bool
                                        out22_: bool
                                        out22_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_26_next_)
                                        d_30_validNext_ = out22_
                                        if d_30_validNext_:
                                            if d_17_completeNow_:
                                                d_31_closedGenerated4_: _dafny.Seq
                                                d_32_closedInside4_: bool
                                                d_33_closedCurrent4_: _dafny.Seq
                                                out23_: _dafny.Seq
                                                out24_: bool
                                                out25_: _dafny.Seq
                                                out23_, out24_, out25_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                                d_31_closedGenerated4_ = out23_
                                                d_32_closedInside4_ = out24_
                                                d_33_closedCurrent4_ = out25_
                                                generated = d_31_closedGenerated4_
                                                insideConstrainedOut = d_32_closedInside4_
                                                currentConstrainedOut = d_33_closedCurrent4_
                                                d_1_steps_ = (d_1_steps_) + (1)
                                                raise _dafny.Break("0")
                                            elif True:
                                                d_34_appendedGenerated_: _dafny.Seq
                                                d_35_appendedInside_: bool
                                                d_36_appendedCurrent_: _dafny.Seq
                                                out26_: _dafny.Seq
                                                out27_: bool
                                                out28_: _dafny.Seq
                                                out26_, out27_, out28_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_26_next_)
                                                d_34_appendedGenerated_ = out26_
                                                d_35_appendedInside_ = out27_
                                                d_36_appendedCurrent_ = out28_
                                                generated = d_34_appendedGenerated_
                                                insideConstrainedOut = d_35_appendedInside_
                                                currentConstrainedOut = d_36_appendedCurrent_
                                                d_1_steps_ = (d_1_steps_) + (1)
                                        elif True:
                                            d_37_fallback_: _dafny.Seq
                                            out29_: _dafny.Seq
                                            out29_ = (d_0_helpers_).ConstrainedStep(lm, parser, prompt, currentConstrainedOut, eosToken)
                                            d_37_fallback_ = out29_
                                            if (d_37_fallback_) == (eosToken):
                                                if (d_17_completeNow_) and ((d_1_steps_) < (maxSteps)):
                                                    d_38_closedGenerated5_: _dafny.Seq
                                                    d_39_closedInside5_: bool
                                                    d_40_closedCurrent5_: _dafny.Seq
                                                    out30_: _dafny.Seq
                                                    out31_: bool
                                                    out32_: _dafny.Seq
                                                    out30_, out31_, out32_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                                    d_38_closedGenerated5_ = out30_
                                                    d_39_closedInside5_ = out31_
                                                    d_40_closedCurrent5_ = out32_
                                                    generated = d_38_closedGenerated5_
                                                    insideConstrainedOut = d_39_closedInside5_
                                                    currentConstrainedOut = d_40_closedCurrent5_
                                                    d_1_steps_ = (d_1_steps_) + (1)
                                                    raise _dafny.Break("0")
                                                elif True:
                                                    raise _dafny.Break("0")
                                            elif True:
                                                if d_17_completeNow_:
                                                    d_41_closedGenerated6_: _dafny.Seq
                                                    d_42_closedInside6_: bool
                                                    d_43_closedCurrent6_: _dafny.Seq
                                                    out33_: _dafny.Seq
                                                    out34_: bool
                                                    out35_: _dafny.Seq
                                                    out33_, out34_, out35_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                                    d_41_closedGenerated6_ = out33_
                                                    d_42_closedInside6_ = out34_
                                                    d_43_closedCurrent6_ = out35_
                                                    generated = d_41_closedGenerated6_
                                                    insideConstrainedOut = d_42_closedInside6_
                                                    currentConstrainedOut = d_43_closedCurrent6_
                                                    d_1_steps_ = (d_1_steps_) + (1)
                                                    raise _dafny.Break("0")
                                                elif True:
                                                    d_44_appendedGenerated2_: _dafny.Seq
                                                    d_45_appendedInside2_: bool
                                                    d_46_appendedCurrent2_: _dafny.Seq
                                                    out36_: _dafny.Seq
                                                    out37_: bool
                                                    out38_: _dafny.Seq
                                                    out36_, out37_, out38_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_37_fallback_)
                                                    d_44_appendedGenerated2_ = out36_
                                                    d_45_appendedInside2_ = out37_
                                                    d_46_appendedCurrent2_ = out38_
                                                    generated = d_44_appendedGenerated2_
                                                    insideConstrainedOut = d_45_appendedInside2_
                                                    currentConstrainedOut = d_46_appendedCurrent2_
                                                    d_1_steps_ = (d_1_steps_) + (1)
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

