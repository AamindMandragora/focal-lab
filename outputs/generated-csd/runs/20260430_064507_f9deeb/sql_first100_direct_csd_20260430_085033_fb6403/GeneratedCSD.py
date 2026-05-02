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
        d_2_openSpanToken_: _dafny.Seq
        d_2_openSpanToken_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))
        d_3_lastChosen_: _dafny.Seq
        d_3_lastChosen_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
        d_4_flatGroups_: _dafny.Seq
        out0_: _dafny.Seq
        out0_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
        d_4_flatGroups_ = out0_
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_5_remaining_: int
                        d_5_remaining_ = (maxSteps) - (d_1_steps_)
                        d_6_chunkBudget_: int
                        if (d_5_remaining_) < (3):
                            d_6_chunkBudget_ = d_5_remaining_
                        elif True:
                            d_6_chunkBudget_ = 3
                        d_7_chunkGenerated_: _dafny.Seq
                        d_8_stoppedOnOpenSpan_: bool
                        d_9_stoppedOnEos_: bool
                        d_10_stepsUsed_: int
                        out1_: _dafny.Seq
                        out2_: bool
                        out3_: bool
                        out4_: int
                        out1_, out2_, out3_, out4_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_6_chunkBudget_, d_2_openSpanToken_, eosToken)
                        d_7_chunkGenerated_ = out1_
                        d_8_stoppedOnOpenSpan_ = out2_
                        d_9_stoppedOnEos_ = out3_
                        d_10_stepsUsed_ = out4_
                        generated = d_7_chunkGenerated_
                        d_1_steps_ = (d_1_steps_) + (d_10_stepsUsed_)
                        if d_9_stoppedOnEos_:
                            raise _dafny.Break("0")
                        elif d_8_stoppedOnOpenSpan_:
                            insideConstrainedOut = True
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                            d_3_lastChosen_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                    elif True:
                        if (parser).IsCompletePrefix(currentConstrainedOut):
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
                            d_3_lastChosen_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_14_dead_: bool
                            out8_: bool
                            out8_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                            d_14_dead_ = out8_
                            if d_14_dead_:
                                d_15_repaired_: _dafny.Seq
                                out9_: _dafny.Seq
                                out9_ = VerifiedDecoderAgent.CSDHelpers.RollbackToBoundary(parser, currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ",")))
                                d_15_repaired_ = out9_
                                d_16_repaired2_: _dafny.Seq
                                d_16_repaired2_ = d_15_repaired_
                                if (len(d_15_repaired_)) < (len(currentConstrainedOut)):
                                    d_16_repaired2_ = d_15_repaired_
                                elif True:
                                    d_17_rolled_: _dafny.Seq
                                    out10_: _dafny.Seq
                                    out10_ = VerifiedDecoderAgent.CSDHelpers.RollbackToValidPrefix(parser, currentConstrainedOut)
                                    d_17_rolled_ = out10_
                                    d_16_repaired2_ = d_17_rolled_
                                d_18_stablePrefix_: _dafny.Seq
                                d_18_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                d_19_repairedGenerated_: _dafny.Seq
                                d_20_repairedCurrent_: _dafny.Seq
                                out11_: _dafny.Seq
                                out12_: _dafny.Seq
                                out11_, out12_ = (d_0_helpers_).RollbackConstrainedSpan(parser, d_18_stablePrefix_, generated, d_16_repaired2_)
                                d_19_repairedGenerated_ = out11_
                                d_20_repairedCurrent_ = out12_
                                generated = d_19_repairedGenerated_
                                currentConstrainedOut = d_20_repairedCurrent_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                d_21_stablePrefix2_: _dafny.Seq
                                d_21_stablePrefix2_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                d_22_constrainedPrompt_: _dafny.Seq
                                d_22_constrainedPrompt_ = (prompt) + (d_21_stablePrefix2_)
                                (lm).GenerateLogits((d_22_constrainedPrompt_) + (currentConstrainedOut))
                                d_23_topCandidates_: _dafny.Seq
                                out13_: _dafny.Seq
                                out13_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_22_constrainedPrompt_, currentConstrainedOut, 12, eosToken)
                                d_23_topCandidates_ = out13_
                                if (len(d_23_topCandidates_)) > (0):
                                    (d_0_helpers_).BoostTokenLogits(lm, d_23_topCandidates_, _dafny.BigRational('2e0'))
                                if (d_3_lastChosen_) != (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))):
                                    (d_0_helpers_).PenalizeTokenLogits(lm, _dafny.SeqWithoutIsStrInference([d_3_lastChosen_]), _dafny.BigRational('3e0'))
                                    d_24_gidx_: int
                                    out14_: int
                                    out14_ = VerifiedDecoderAgent.CSDHelpers.GroupContaining(validTokenGroups, d_3_lastChosen_)
                                    d_24_gidx_ = out14_
                                    if (0) <= (d_24_gidx_):
                                        d_25_group_: _dafny.Seq
                                        d_25_group_ = (validTokenGroups)[d_24_gidx_]
                                        d_26_anyValid_: bool
                                        out15_: bool
                                        out15_ = (d_0_helpers_).GroupHasValidMember(parser, currentConstrainedOut, d_25_group_)
                                        d_26_anyValid_ = out15_
                                        if d_26_anyValid_:
                                            d_27_focused_: _dafny.Seq
                                            out16_: _dafny.Seq
                                            out16_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_23_topCandidates_, d_25_group_)
                                            d_27_focused_ = out16_
                                            if (len(d_27_focused_)) > (0):
                                                (d_0_helpers_).BoostTokenLogits(lm, d_27_focused_, _dafny.BigRational('15e-1'))
                                elif True:
                                    if (len(d_4_flatGroups_)) > (0):
                                        d_28_anyFlatValid_: bool
                                        out17_: bool
                                        out17_ = (d_0_helpers_).GroupHasValidMember(parser, currentConstrainedOut, d_4_flatGroups_)
                                        d_28_anyFlatValid_ = out17_
                                        if d_28_anyFlatValid_:
                                            d_29_focused2_: _dafny.Seq
                                            out18_: _dafny.Seq
                                            out18_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_23_topCandidates_, d_4_flatGroups_)
                                            d_29_focused2_ = out18_
                                            if (len(d_29_focused2_)) > (0):
                                                (d_0_helpers_).BoostTokenLogits(lm, d_29_focused2_, _dafny.BigRational('1e0'))
                                (lm).MaskValidNextAndEos(parser, currentConstrainedOut, eosToken)
                                d_30_next_: _dafny.Seq
                                out19_: _dafny.Seq
                                out19_ = (lm).ChooseNextToken()
                                d_30_next_ = out19_
                                (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_30_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_31_appendedGenerated_: _dafny.Seq
                                    d_32_appendedInside_: bool
                                    d_33_appendedCurrent_: _dafny.Seq
                                    out20_: _dafny.Seq
                                    out21_: bool
                                    out22_: _dafny.Seq
                                    out20_, out21_, out22_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_30_next_)
                                    d_31_appendedGenerated_ = out20_
                                    d_32_appendedInside_ = out21_
                                    d_33_appendedCurrent_ = out22_
                                    generated = d_31_appendedGenerated_
                                    insideConstrainedOut = d_32_appendedInside_
                                    currentConstrainedOut = d_33_appendedCurrent_
                                    d_3_lastChosen_ = d_30_next_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

