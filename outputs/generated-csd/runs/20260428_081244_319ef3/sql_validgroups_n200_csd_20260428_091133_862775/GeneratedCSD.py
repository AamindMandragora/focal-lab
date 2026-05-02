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
        d_2_flatGroups_: _dafny.Seq
        out0_: _dafny.Seq
        out0_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
        d_2_flatGroups_ = out0_
        d_3_continuationTokens_: _dafny.Seq
        d_3_continuationTokens_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "FROM")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "JOIN")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "ON")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "WHERE")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "GROUP")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "ORDER")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "HAVING")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "LIMIT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "UNION")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "INTERSECT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "EXCEPT"))])
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_4_openedGenerated_: _dafny.Seq
                        d_5_openedInside_: bool
                        d_6_openedCurrent_: _dafny.Seq
                        out1_: _dafny.Seq
                        out2_: bool
                        out3_: _dafny.Seq
                        out1_, out2_, out3_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                        d_4_openedGenerated_ = out1_
                        d_5_openedInside_ = out2_
                        d_6_openedCurrent_ = out3_
                        generated = d_4_openedGenerated_
                        insideConstrainedOut = d_5_openedInside_
                        currentConstrainedOut = d_6_openedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_7_completeNow_: bool
                        d_7_completeNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_7_completeNow_:
                            d_8_closedGenerated_: _dafny.Seq
                            d_9_closedInside_: bool
                            d_10_closedCurrent_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_8_closedGenerated_ = out4_
                            d_9_closedInside_ = out5_
                            d_10_closedCurrent_ = out6_
                            generated = d_8_closedGenerated_
                            insideConstrainedOut = d_9_closedInside_
                            currentConstrainedOut = d_10_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                            raise _dafny.Break("0")
                        elif True:
                            d_11_deadEnd_: bool
                            out7_: bool
                            out7_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                            d_11_deadEnd_ = out7_
                            if d_11_deadEnd_:
                                d_12_repaired_: _dafny.Seq
                                out8_: _dafny.Seq
                                out8_ = VerifiedDecoderAgent.CSDHelpers.RollbackToValidPrefix(parser, currentConstrainedOut)
                                d_12_repaired_ = out8_
                                if (len(d_12_repaired_)) == (len(currentConstrainedOut)):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_13_stablePrefix_: _dafny.Seq
                                    d_13_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                    generated = (d_13_stablePrefix_) + (d_12_repaired_)
                                    currentConstrainedOut = d_12_repaired_
                            elif True:
                                d_14_stablePrefix2_: _dafny.Seq
                                d_14_stablePrefix2_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                d_15_constrainedPrompt_: _dafny.Seq
                                d_15_constrainedPrompt_ = (prompt) + (d_14_stablePrefix2_)
                                (lm).GenerateLogits((d_15_constrainedPrompt_) + (currentConstrainedOut))
                                d_16_candidates_: _dafny.Seq
                                out9_: _dafny.Seq
                                out9_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_15_constrainedPrompt_, currentConstrainedOut, 12, eosToken)
                                d_16_candidates_ = out9_
                                d_17_hinted_: _dafny.Seq
                                out10_: _dafny.Seq
                                out10_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_16_candidates_, d_2_flatGroups_)
                                d_17_hinted_ = out10_
                                if (len(d_17_hinted_)) > (0):
                                    (d_0_helpers_).BoostTokenLogits(lm, d_17_hinted_, _dafny.BigRational('4e0'))
                                if (len(currentConstrainedOut)) >= (8):
                                    d_18_driftCandidates_: _dafny.Seq
                                    out11_: _dafny.Seq
                                    out11_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_16_candidates_, d_3_continuationTokens_)
                                    d_18_driftCandidates_ = out11_
                                    if (len(d_18_driftCandidates_)) > (0):
                                        (d_0_helpers_).PenalizeTokenLogits(lm, d_18_driftCandidates_, _dafny.BigRational('25e-1'))
                                d_19_budget_: int
                                d_19_budget_ = stepTokenBudget
                                if (d_19_budget_) == (0):
                                    d_19_budget_ = 1
                                if ((maxSteps) - (d_1_steps_)) < (d_19_budget_):
                                    d_19_budget_ = (maxSteps) - (d_1_steps_)
                                d_20_currentOut_: _dafny.Seq
                                d_21_hitEos_: bool
                                d_22_stepsUsed_: int
                                out12_: _dafny.Seq
                                out13_: bool
                                out14_: int
                                out12_, out13_, out14_ = (d_0_helpers_).ConstrainedSymbol(lm, parser, d_15_constrainedPrompt_, currentConstrainedOut, d_19_budget_, eosToken)
                                d_20_currentOut_ = out12_
                                d_21_hitEos_ = out13_
                                d_22_stepsUsed_ = out14_
                                d_1_steps_ = (d_1_steps_) + (d_22_stepsUsed_)
                                if d_21_hitEos_:
                                    raise _dafny.Break("0")
                                elif True:
                                    generated = (d_14_stablePrefix2_) + (d_20_currentOut_)
                                    currentConstrainedOut = d_20_currentOut_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

