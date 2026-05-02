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
        d_2_haveSpan_: bool
        d_2_haveSpan_ = False
        if insideConstrainedOut:
            d_2_haveSpan_ = True
        elif True:
            d_3_scan_: int
            d_3_scan_ = 0
            while ((d_3_scan_) + (1)) < (len(generated)):
                if ((generated)[d_3_scan_]) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                    if ((generated)[(d_3_scan_) + (1)]) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))):
                        d_2_haveSpan_ = True
                d_3_scan_ = (d_3_scan_) + (1)
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if insideConstrainedOut:
                        d_4_completeNow_: bool
                        d_4_completeNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_4_completeNow_:
                            d_5_closedGenerated_: _dafny.Seq
                            d_6_closedInside_: bool
                            d_7_closedCurrent_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_5_closedGenerated_ = out0_
                            d_6_closedInside_ = out1_
                            d_7_closedCurrent_ = out2_
                            generated = d_5_closedGenerated_
                            insideConstrainedOut = d_6_closedInside_
                            currentConstrainedOut = d_7_closedCurrent_
                            d_2_haveSpan_ = True
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_8_remainingInside_: int
                            d_8_remainingInside_ = (maxSteps) - (d_1_steps_)
                            if (d_8_remainingInside_) <= (1):
                                d_9_stablePrefix_: _dafny.Seq
                                d_9_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                d_10_rolled_: _dafny.Seq
                                out3_: _dafny.Seq
                                out3_ = VerifiedDecoderAgent.CSDHelpers.RollbackToValidPrefix(parser, currentConstrainedOut)
                                d_10_rolled_ = out3_
                                d_11_rbGenerated_: _dafny.Seq
                                d_12_rbCurrent_: _dafny.Seq
                                out4_: _dafny.Seq
                                out5_: _dafny.Seq
                                out4_, out5_ = (d_0_helpers_).RollbackConstrainedSpan(parser, d_9_stablePrefix_, generated, currentConstrainedOut)
                                d_11_rbGenerated_ = out4_
                                d_12_rbCurrent_ = out5_
                                generated = d_11_rbGenerated_
                                currentConstrainedOut = d_12_rbCurrent_
                                d_13_completeAfterRollback_: bool
                                d_13_completeAfterRollback_ = (parser).IsCompletePrefix(currentConstrainedOut)
                                if (d_13_completeAfterRollback_) and ((d_1_steps_) < (maxSteps)):
                                    d_14_closedGenerated2_: _dafny.Seq
                                    d_15_closedInside2_: bool
                                    d_16_closedCurrent2_: _dafny.Seq
                                    out6_: _dafny.Seq
                                    out7_: bool
                                    out8_: _dafny.Seq
                                    out6_, out7_, out8_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_14_closedGenerated2_ = out6_
                                    d_15_closedInside2_ = out7_
                                    d_16_closedCurrent2_ = out8_
                                    generated = d_14_closedGenerated2_
                                    insideConstrainedOut = d_15_closedInside2_
                                    currentConstrainedOut = d_16_closedCurrent2_
                                    d_2_haveSpan_ = True
                                    d_1_steps_ = (d_1_steps_) + (1)
                                elif True:
                                    generated = d_9_stablePrefix_
                                    insideConstrainedOut = False
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                    raise _dafny.Break("0")
                            elif True:
                                d_17_narrow_: bool
                                out9_: bool
                                out9_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 2)
                                d_17_narrow_ = out9_
                                if d_17_narrow_:
                                    d_18_stablePrefix2_: _dafny.Seq
                                    d_18_stablePrefix2_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                    d_19_rolled2_: _dafny.Seq
                                    out10_: _dafny.Seq
                                    out10_ = VerifiedDecoderAgent.CSDHelpers.RollbackToValidPrefix(parser, currentConstrainedOut)
                                    d_19_rolled2_ = out10_
                                    d_20_rbGenerated2_: _dafny.Seq
                                    d_21_rbCurrent2_: _dafny.Seq
                                    out11_: _dafny.Seq
                                    out12_: _dafny.Seq
                                    out11_, out12_ = (d_0_helpers_).RollbackConstrainedSpan(parser, d_18_stablePrefix2_, generated, currentConstrainedOut)
                                    d_20_rbGenerated2_ = out11_
                                    d_21_rbCurrent2_ = out12_
                                    generated = d_20_rbGenerated2_
                                    currentConstrainedOut = d_21_rbCurrent2_
                                    d_22_completeRolled_: bool
                                    d_22_completeRolled_ = (parser).IsCompletePrefix(currentConstrainedOut)
                                    if d_22_completeRolled_:
                                        d_23_closedGenerated3_: _dafny.Seq
                                        d_24_closedInside3_: bool
                                        d_25_closedCurrent3_: _dafny.Seq
                                        out13_: _dafny.Seq
                                        out14_: bool
                                        out15_: _dafny.Seq
                                        out13_, out14_, out15_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                        d_23_closedGenerated3_ = out13_
                                        d_24_closedInside3_ = out14_
                                        d_25_closedCurrent3_ = out15_
                                        generated = d_23_closedGenerated3_
                                        insideConstrainedOut = d_24_closedInside3_
                                        currentConstrainedOut = d_25_closedCurrent3_
                                        d_2_haveSpan_ = True
                                        d_1_steps_ = (d_1_steps_) + (1)
                                    elif True:
                                        (lm).GenerateLogits((prompt) + (generated))
                                        d_26_candidates_: _dafny.Seq
                                        out16_: _dafny.Seq
                                        out16_ = (d_0_helpers_).TopValidCandidates(lm, parser, prompt, currentConstrainedOut, 4, eosToken)
                                        d_26_candidates_ = out16_
                                        (d_0_helpers_).BoostTokenLogits(lm, d_26_candidates_, _dafny.BigRational('1e2'))
                                        d_27_nextInside_: _dafny.Seq
                                        out17_: _dafny.Seq
                                        out17_ = (d_0_helpers_).ConstrainedStep(lm, parser, prompt, currentConstrainedOut, eosToken)
                                        d_27_nextInside_ = out17_
                                        d_1_steps_ = (d_1_steps_) + (1)
                                        if (d_27_nextInside_) == (eosToken):
                                            raise _dafny.Break("0")
                                        elif True:
                                            d_28_appendedGenerated_: _dafny.Seq
                                            d_29_appendedInside_: bool
                                            d_30_appendedCurrent_: _dafny.Seq
                                            out18_: _dafny.Seq
                                            out19_: bool
                                            out20_: _dafny.Seq
                                            out18_, out19_, out20_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_27_nextInside_)
                                            d_28_appendedGenerated_ = out18_
                                            d_29_appendedInside_ = out19_
                                            d_30_appendedCurrent_ = out20_
                                            generated = d_28_appendedGenerated_
                                            insideConstrainedOut = d_29_appendedInside_
                                            currentConstrainedOut = d_30_appendedCurrent_
                                elif True:
                                    (lm).GenerateLogits((prompt) + (generated))
                                    d_31_candidates2_: _dafny.Seq
                                    out21_: _dafny.Seq
                                    out21_ = (d_0_helpers_).TopValidCandidates(lm, parser, prompt, currentConstrainedOut, 4, eosToken)
                                    d_31_candidates2_ = out21_
                                    (d_0_helpers_).BoostTokenLogits(lm, d_31_candidates2_, _dafny.BigRational('1e2'))
                                    d_32_nextInside2_: _dafny.Seq
                                    out22_: _dafny.Seq
                                    out22_ = (d_0_helpers_).ConstrainedStep(lm, parser, prompt, currentConstrainedOut, eosToken)
                                    d_32_nextInside2_ = out22_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    if (d_32_nextInside2_) == (eosToken):
                                        raise _dafny.Break("0")
                                    elif True:
                                        d_33_appendedGenerated2_: _dafny.Seq
                                        d_34_appendedInside2_: bool
                                        d_35_appendedCurrent2_: _dafny.Seq
                                        out23_: _dafny.Seq
                                        out24_: bool
                                        out25_: _dafny.Seq
                                        out23_, out24_, out25_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_32_nextInside2_)
                                        d_33_appendedGenerated2_ = out23_
                                        d_34_appendedInside2_ = out24_
                                        d_35_appendedCurrent2_ = out25_
                                        generated = d_33_appendedGenerated2_
                                        insideConstrainedOut = d_34_appendedInside2_
                                        currentConstrainedOut = d_35_appendedCurrent2_
                    elif True:
                        d_36_remaining_: int
                        d_36_remaining_ = (maxSteps) - (d_1_steps_)
                        if (d_36_remaining_) < (3):
                            d_37_nextOutside_: _dafny.Seq
                            out26_: _dafny.Seq
                            out26_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_37_nextOutside_ = out26_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_37_nextOutside_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_37_nextOutside_]))
                        elif True:
                            (lm).GenerateLogits((prompt) + (generated))
                            (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]), _dafny.BigRational('3e0'))
                            (d_0_helpers_).PenalizeTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]), _dafny.BigRational('8e0'))
                            d_38_nextOutside2_: _dafny.Seq
                            out27_: _dafny.Seq
                            out27_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_38_nextOutside2_ = out27_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_38_nextOutside2_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                if (d_38_nextOutside2_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    d_39_openedGenerated_: _dafny.Seq
                                    d_40_openedInside_: bool
                                    d_41_openedCurrent_: _dafny.Seq
                                    out28_: _dafny.Seq
                                    out29_: bool
                                    out30_: _dafny.Seq
                                    out28_, out29_, out30_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                    d_39_openedGenerated_ = out28_
                                    d_40_openedInside_ = out29_
                                    d_41_openedCurrent_ = out30_
                                    generated = d_39_openedGenerated_
                                    insideConstrainedOut = d_40_openedInside_
                                    currentConstrainedOut = d_41_openedCurrent_
                                elif True:
                                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_38_nextOutside2_]))
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

