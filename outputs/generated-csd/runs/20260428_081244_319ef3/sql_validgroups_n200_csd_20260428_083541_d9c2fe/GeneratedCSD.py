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
        d_2_selectTok_: _dafny.Seq
        d_2_selectTok_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SELECT"))
        d_3_fromTok_: _dafny.Seq
        d_3_fromTok_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "FROM"))
        d_4_whereTok_: _dafny.Seq
        d_4_whereTok_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "WHERE"))
        d_5_flatGroups_: _dafny.Seq
        out0_: _dafny.Seq
        out0_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
        d_5_flatGroups_ = out0_
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_6_openedGenerated_: _dafny.Seq
                        d_7_openedInside_: bool
                        d_8_openedCurrent_: _dafny.Seq
                        out1_: _dafny.Seq
                        out2_: bool
                        out3_: _dafny.Seq
                        out1_, out2_, out3_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                        d_6_openedGenerated_ = out1_
                        d_7_openedInside_ = out2_
                        d_8_openedCurrent_ = out3_
                        generated = d_6_openedGenerated_
                        insideConstrainedOut = d_7_openedInside_
                        currentConstrainedOut = d_8_openedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_9_completeNow_: bool
                        d_9_completeNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_9_completeNow_:
                            d_10_closedGenerated_: _dafny.Seq
                            d_11_closedInside_: bool
                            d_12_closedCurrent_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_10_closedGenerated_ = out4_
                            d_11_closedInside_ = out5_
                            d_12_closedCurrent_ = out6_
                            generated = d_10_closedGenerated_
                            insideConstrainedOut = d_11_closedInside_
                            currentConstrainedOut = d_12_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                            raise _dafny.Break("0")
                        elif True:
                            d_13_deadEnd_: bool
                            out7_: bool
                            out7_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                            d_13_deadEnd_ = out7_
                            if d_13_deadEnd_:
                                d_14_repaired_: _dafny.Seq
                                out8_: _dafny.Seq
                                out8_ = VerifiedDecoderAgent.CSDHelpers.RollbackToValidPrefix(parser, currentConstrainedOut)
                                d_14_repaired_ = out8_
                                if (len(d_14_repaired_)) == (len(currentConstrainedOut)):
                                    d_15_repairedWhere_: _dafny.Seq
                                    out9_: _dafny.Seq
                                    out9_ = VerifiedDecoderAgent.CSDHelpers.RollbackToBoundary(parser, currentConstrainedOut, d_4_whereTok_)
                                    d_15_repairedWhere_ = out9_
                                    d_16_repairedFrom_: _dafny.Seq
                                    out10_: _dafny.Seq
                                    out10_ = VerifiedDecoderAgent.CSDHelpers.RollbackToBoundary(parser, currentConstrainedOut, d_3_fromTok_)
                                    d_16_repairedFrom_ = out10_
                                    d_17_repairedSelect_: _dafny.Seq
                                    out11_: _dafny.Seq
                                    out11_ = VerifiedDecoderAgent.CSDHelpers.RollbackToBoundary(parser, currentConstrainedOut, d_2_selectTok_)
                                    d_17_repairedSelect_ = out11_
                                    d_14_repaired_ = d_15_repairedWhere_
                                    if (len(d_16_repairedFrom_)) < (len(d_14_repaired_)):
                                        d_14_repaired_ = d_16_repairedFrom_
                                    if (len(d_17_repairedSelect_)) < (len(d_14_repaired_)):
                                        d_14_repaired_ = d_17_repairedSelect_
                                if (len(d_14_repaired_)) == (len(currentConstrainedOut)):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_18_stablePrefix_: _dafny.Seq
                                    d_18_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                    generated = (d_18_stablePrefix_) + (d_14_repaired_)
                                    currentConstrainedOut = d_14_repaired_
                            elif True:
                                d_19_stablePrefix2_: _dafny.Seq
                                d_19_stablePrefix2_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                d_20_constrainedPrompt_: _dafny.Seq
                                d_20_constrainedPrompt_ = (prompt) + (d_19_stablePrefix2_)
                                d_21_budget_: int
                                d_21_budget_ = stepTokenBudget
                                if (d_21_budget_) > ((maxSteps) - (d_1_steps_)):
                                    d_21_budget_ = (maxSteps) - (d_1_steps_)
                                if (d_21_budget_) == (0):
                                    raise _dafny.Break("0")
                                elif True:
                                    (lm).GenerateLogits((d_20_constrainedPrompt_) + (currentConstrainedOut))
                                    d_22_candidates_: _dafny.Seq
                                    out12_: _dafny.Seq
                                    out12_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_20_constrainedPrompt_, currentConstrainedOut, 8, eosToken)
                                    d_22_candidates_ = out12_
                                    d_23_hinted_: _dafny.Seq
                                    out13_: _dafny.Seq
                                    out13_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_22_candidates_, d_5_flatGroups_)
                                    d_23_hinted_ = out13_
                                    if (len(d_23_hinted_)) > (0):
                                        (d_0_helpers_).BoostTokenLogits(lm, d_23_hinted_, _dafny.BigRational('3e0'))
                                    d_24_currentOut_: _dafny.Seq
                                    d_25_hitEos_: bool
                                    d_26_stepsUsed_: int
                                    out14_: _dafny.Seq
                                    out15_: bool
                                    out16_: int
                                    out14_, out15_, out16_ = (d_0_helpers_).ConstrainedSymbol(lm, parser, d_20_constrainedPrompt_, currentConstrainedOut, d_21_budget_, eosToken)
                                    d_24_currentOut_ = out14_
                                    d_25_hitEos_ = out15_
                                    d_26_stepsUsed_ = out16_
                                    d_1_steps_ = (d_1_steps_) + (d_26_stepsUsed_)
                                    if d_25_hitEos_:
                                        raise _dafny.Break("0")
                                    elif True:
                                        currentConstrainedOut = d_24_currentOut_
                                        generated = (d_19_stablePrefix2_) + (currentConstrainedOut)
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

