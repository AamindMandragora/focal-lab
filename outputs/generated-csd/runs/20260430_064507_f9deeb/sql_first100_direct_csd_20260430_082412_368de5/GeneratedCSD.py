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
    def MyCSDStrategy(lm, parser, prompt, generatedPrefix, insideConstrained, currentConstrained, maxSteps, stepTokenBudget, validTokenGroups, eosToken):
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
        d_2_narrowThreshold_: int
        d_2_narrowThreshold_ = 12
        d_3_deadEndMinValidCount_: int
        d_3_deadEndMinValidCount_ = 2
        d_4_focusSep_: _dafny.Seq
        d_4_focusSep_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "."))
        d_5_focusedToken_: _dafny.Seq
        d_5_focusedToken_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
        d_6_hasFocusedToken_: bool
        d_6_hasFocusedToken_ = False
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_7_chunkBudget_: int
                        d_7_chunkBudget_ = (maxSteps) - (d_1_steps_)
                        d_8_chunkedGenerated_: _dafny.Seq
                        d_9_stoppedOnOpenSpan_: bool
                        d_10_stoppedOnEos_: bool
                        d_11_stepsUsed_: int
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: bool
                        out3_: int
                        out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_7_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_8_chunkedGenerated_ = out0_
                        d_9_stoppedOnOpenSpan_ = out1_
                        d_10_stoppedOnEos_ = out2_
                        d_11_stepsUsed_ = out3_
                        generated = d_8_chunkedGenerated_
                        d_1_steps_ = (d_1_steps_) + (d_11_stepsUsed_)
                        if d_10_stoppedOnEos_:
                            raise _dafny.Break("0")
                        elif d_9_stoppedOnOpenSpan_:
                            insideConstrainedOut = True
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                            d_6_hasFocusedToken_ = False
                            d_5_focusedToken_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                    elif True:
                        d_12_completeNow_: bool
                        d_12_completeNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_12_completeNow_:
                            d_13_closedGenerated_: _dafny.Seq
                            d_14_closedInside_: bool
                            d_15_closedCurrent_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_13_closedGenerated_ = out4_
                            d_14_closedInside_ = out5_
                            d_15_closedCurrent_ = out6_
                            generated = d_13_closedGenerated_
                            insideConstrainedOut = d_14_closedInside_
                            currentConstrainedOut = d_15_closedCurrent_
                            d_6_hasFocusedToken_ = False
                            d_5_focusedToken_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_16_lastBeforeDot_: _dafny.Seq
                            d_17_foundFocus_: bool
                            out7_: _dafny.Seq
                            out8_: bool
                            out7_, out8_ = (d_0_helpers_).LastTokenBefore(currentConstrainedOut, d_4_focusSep_)
                            d_16_lastBeforeDot_ = out7_
                            d_17_foundFocus_ = out8_
                            if d_17_foundFocus_:
                                d_5_focusedToken_ = d_16_lastBeforeDot_
                                d_6_hasFocusedToken_ = True
                            d_18_constrainedPrompt_: _dafny.Seq
                            d_18_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_19_validCount_: int
                            out9_: int
                            out9_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                            d_19_validCount_ = out9_
                            d_20_narrow_: bool
                            out10_: bool
                            out10_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, d_3_deadEndMinValidCount_)
                            d_20_narrow_ = out10_
                            if (d_20_narrow_) or ((d_19_validCount_) <= (d_2_narrowThreshold_)):
                                d_21_next_: _dafny.Seq
                                out11_: _dafny.Seq
                                out11_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_18_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_21_next_ = out11_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_21_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_22_appendedGenerated_: _dafny.Seq
                                    d_23_appendedInside_: bool
                                    d_24_appendedCurrent_: _dafny.Seq
                                    out12_: _dafny.Seq
                                    out13_: bool
                                    out14_: _dafny.Seq
                                    out12_, out13_, out14_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_21_next_)
                                    d_22_appendedGenerated_ = out12_
                                    d_23_appendedInside_ = out13_
                                    d_24_appendedCurrent_ = out14_
                                    generated = d_22_appendedGenerated_
                                    insideConstrainedOut = d_23_appendedInside_
                                    currentConstrainedOut = d_24_appendedCurrent_
                            elif True:
                                (lm).GenerateLogits((d_18_constrainedPrompt_) + (currentConstrainedOut))
                                d_25_candidates_: _dafny.Seq
                                out15_: _dafny.Seq
                                out15_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_18_constrainedPrompt_, currentConstrainedOut, 30, eosToken)
                                d_25_candidates_ = out15_
                                if ((len(validTokenGroups)) > (0)) and ((len(d_25_candidates_)) > (0)):
                                    d_26_flatPreferred_: _dafny.Seq
                                    out16_: _dafny.Seq
                                    out16_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
                                    d_26_flatPreferred_ = out16_
                                    if (len(d_26_flatPreferred_)) > (0):
                                        d_27_preferred_: _dafny.Seq
                                        out17_: _dafny.Seq
                                        out17_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_25_candidates_, d_26_flatPreferred_)
                                        d_27_preferred_ = out17_
                                        if (len(d_27_preferred_)) > (0):
                                            (d_0_helpers_).BoostTokenLogits(lm, d_27_preferred_, _dafny.BigRational('4e0'))
                                    if d_6_hasFocusedToken_:
                                        d_28_groupIdx_: int
                                        out18_: int
                                        out18_ = VerifiedDecoderAgent.CSDHelpers.GroupContaining(validTokenGroups, d_5_focusedToken_)
                                        d_28_groupIdx_ = out18_
                                        if (0) <= (d_28_groupIdx_):
                                            d_29_focusedGroup_: _dafny.Seq
                                            d_29_focusedGroup_ = (validTokenGroups)[d_28_groupIdx_]
                                            d_30_focusedCandidates_: _dafny.Seq
                                            out19_: _dafny.Seq
                                            out19_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_25_candidates_, d_29_focusedGroup_)
                                            d_30_focusedCandidates_ = out19_
                                            if (len(d_30_focusedCandidates_)) > (0):
                                                (d_0_helpers_).BoostTokenLogits(lm, d_30_focusedCandidates_, _dafny.BigRational('8e0'))
                                (lm).MaskValidNextAndEos(parser, currentConstrainedOut, eosToken)
                                d_31_next_: _dafny.Seq
                                out20_: _dafny.Seq
                                out20_ = (lm).ChooseNextToken()
                                d_31_next_ = out20_
                                (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_31_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_32_appendedGenerated2_: _dafny.Seq
                                    d_33_appendedInside2_: bool
                                    d_34_appendedCurrent2_: _dafny.Seq
                                    out21_: _dafny.Seq
                                    out22_: bool
                                    out23_: _dafny.Seq
                                    out21_, out22_, out23_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_31_next_)
                                    d_32_appendedGenerated2_ = out21_
                                    d_33_appendedInside2_ = out22_
                                    d_34_appendedCurrent2_ = out23_
                                    generated = d_32_appendedGenerated2_
                                    insideConstrainedOut = d_33_appendedInside2_
                                    currentConstrainedOut = d_34_appendedCurrent2_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

