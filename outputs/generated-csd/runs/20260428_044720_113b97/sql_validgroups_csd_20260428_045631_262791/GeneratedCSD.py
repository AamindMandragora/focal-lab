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
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_2_chunkBudget_: int
                        d_2_chunkBudget_ = (maxSteps) - (d_1_steps_)
                        d_3_chunkedGenerated_: _dafny.Seq
                        d_4_stoppedOnOpenSpan_: bool
                        d_5_stoppedOnEos_: bool
                        d_6_stepsUsed_: int
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: bool
                        out3_: int
                        out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_2_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_3_chunkedGenerated_ = out0_
                        d_4_stoppedOnOpenSpan_ = out1_
                        d_5_stoppedOnEos_ = out2_
                        d_6_stepsUsed_ = out3_
                        generated = d_3_chunkedGenerated_
                        d_1_steps_ = (d_1_steps_) + (d_6_stepsUsed_)
                        if d_5_stoppedOnEos_:
                            raise _dafny.Break("0")
                        elif True:
                            if d_4_stoppedOnOpenSpan_:
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        d_7_narrow_: bool
                        out4_: bool
                        out4_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 2)
                        d_7_narrow_ = out4_
                        if (d_7_narrow_) and ((0) < (len(currentConstrainedOut))):
                            d_8_stablePrefixRepair_: _dafny.Seq
                            d_8_stablePrefixRepair_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_9_repairedCurrent_: _dafny.Seq
                            out5_: _dafny.Seq
                            out5_ = VerifiedDecoderAgent.CSDHelpers.RollbackToBoundary(parser, currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ",")))
                            d_9_repairedCurrent_ = out5_
                            currentConstrainedOut = d_9_repairedCurrent_
                            generated = (d_8_stablePrefixRepair_) + (currentConstrainedOut)
                            raise _dafny.Break("0")
                        elif True:
                            d_10_completeNow_: bool
                            d_10_completeNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                            if d_10_completeNow_:
                                d_11_closedGenerated_: _dafny.Seq
                                d_12_closedInside_: bool
                                d_13_closedCurrent_: _dafny.Seq
                                out6_: _dafny.Seq
                                out7_: bool
                                out8_: _dafny.Seq
                                out6_, out7_, out8_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_11_closedGenerated_ = out6_
                                d_12_closedInside_ = out7_
                                d_13_closedCurrent_ = out8_
                                generated = d_11_closedGenerated_
                                insideConstrainedOut = d_12_closedInside_
                                currentConstrainedOut = d_13_closedCurrent_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                d_14_stablePrefix_: _dafny.Seq
                                d_14_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                d_15_constrainedPrompt_: _dafny.Seq
                                d_15_constrainedPrompt_ = (prompt) + (d_14_stablePrefix_)
                                (lm).GenerateLogits((d_15_constrainedPrompt_) + (currentConstrainedOut))
                                if (len(validTokenGroups)) > (0):
                                    d_16_candidates_: _dafny.Seq
                                    out9_: _dafny.Seq
                                    out9_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_15_constrainedPrompt_, currentConstrainedOut, 24, eosToken)
                                    d_16_candidates_ = out9_
                                    d_17_best_: _dafny.Seq
                                    out10_: _dafny.Seq
                                    out10_ = (d_0_helpers_).GetHighestLogitToken(lm)
                                    d_17_best_ = out10_
                                    d_18_bestGroupIdx_: int
                                    out11_: int
                                    out11_ = VerifiedDecoderAgent.CSDHelpers.GroupContaining(validTokenGroups, d_17_best_)
                                    d_18_bestGroupIdx_ = out11_
                                    if (d_18_bestGroupIdx_) >= (0):
                                        d_19_favored_: _dafny.Seq
                                        out12_: _dafny.Seq
                                        out12_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_16_candidates_, (validTokenGroups)[d_18_bestGroupIdx_])
                                        d_19_favored_ = out12_
                                        if (len(d_19_favored_)) > (0):
                                            (d_0_helpers_).BoostTokenLogits(lm, d_19_favored_, _dafny.BigRational('8e0'))
                                        d_20_flatAll_: _dafny.Seq
                                        out13_: _dafny.Seq
                                        out13_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
                                        d_20_flatAll_ = out13_
                                        d_21_groupedCandidates_: _dafny.Seq
                                        out14_: _dafny.Seq
                                        out14_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_16_candidates_, d_20_flatAll_)
                                        d_21_groupedCandidates_ = out14_
                                        d_22_disfavored_: _dafny.Seq
                                        out15_: _dafny.Seq
                                        out15_ = VerifiedDecoderAgent.CSDHelpers.SubtractTokenSets(d_21_groupedCandidates_, d_19_favored_)
                                        d_22_disfavored_ = out15_
                                        if (len(d_22_disfavored_)) > (0):
                                            (d_0_helpers_).PenalizeTokenLogits(lm, d_22_disfavored_, _dafny.BigRational('2e0'))
                                    elif True:
                                        d_23_flatPreferred_: _dafny.Seq
                                        out16_: _dafny.Seq
                                        out16_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
                                        d_23_flatPreferred_ = out16_
                                        d_24_preferred_: _dafny.Seq
                                        out17_: _dafny.Seq
                                        out17_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_16_candidates_, d_23_flatPreferred_)
                                        d_24_preferred_ = out17_
                                        if (len(d_24_preferred_)) > (0):
                                            (d_0_helpers_).BoostTokenLogits(lm, d_24_preferred_, _dafny.BigRational('3e0'))
                                d_25_symbolBudget_: int
                                d_25_symbolBudget_ = stepTokenBudget
                                if (d_25_symbolBudget_) == (0):
                                    d_25_symbolBudget_ = 1
                                d_26_remaining_: int
                                d_26_remaining_ = (maxSteps) - (d_1_steps_)
                                if (d_25_symbolBudget_) > (d_26_remaining_):
                                    d_25_symbolBudget_ = d_26_remaining_
                                d_27_symbolCurrent_: _dafny.Seq
                                d_28_hitEos_: bool
                                d_29_symbolSteps_: int
                                out18_: _dafny.Seq
                                out19_: bool
                                out20_: int
                                out18_, out19_, out20_ = (d_0_helpers_).ConstrainedSymbol(lm, parser, d_15_constrainedPrompt_, currentConstrainedOut, d_25_symbolBudget_, eosToken)
                                d_27_symbolCurrent_ = out18_
                                d_28_hitEos_ = out19_
                                d_29_symbolSteps_ = out20_
                                d_1_steps_ = (d_1_steps_) + (d_29_symbolSteps_)
                                if d_28_hitEos_:
                                    raise _dafny.Break("0")
                                elif True:
                                    currentConstrainedOut = d_27_symbolCurrent_
                                    generated = (d_14_stablePrefix_) + (currentConstrainedOut)
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

