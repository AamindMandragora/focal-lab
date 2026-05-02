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
                        d_2_openedGenerated_: _dafny.Seq
                        d_3_openedInside_: bool
                        d_4_openedCurrent_: _dafny.Seq
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: _dafny.Seq
                        out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                        d_2_openedGenerated_ = out0_
                        d_3_openedInside_ = out1_
                        d_4_openedCurrent_ = out2_
                        generated = d_2_openedGenerated_
                        insideConstrainedOut = d_3_openedInside_
                        currentConstrainedOut = d_4_openedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_5_completeNow_: bool
                        d_5_completeNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_5_completeNow_:
                            d_6_closedGenerated_: _dafny.Seq
                            d_7_closedInside_: bool
                            d_8_closedCurrent_: _dafny.Seq
                            out3_: _dafny.Seq
                            out4_: bool
                            out5_: _dafny.Seq
                            out3_, out4_, out5_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_6_closedGenerated_ = out3_
                            d_7_closedInside_ = out4_
                            d_8_closedCurrent_ = out5_
                            generated = d_6_closedGenerated_
                            insideConstrainedOut = d_7_closedInside_
                            currentConstrainedOut = d_8_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_9_narrow_: bool
                            out6_: bool
                            out6_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                            d_9_narrow_ = out6_
                            if (d_9_narrow_) and ((0) < (len(currentConstrainedOut))):
                                d_10_stablePrefixRepair_: _dafny.Seq
                                d_10_stablePrefixRepair_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                d_11_repairedCurrent_: _dafny.Seq
                                out7_: _dafny.Seq
                                out7_ = VerifiedDecoderAgent.CSDHelpers.RollbackToValidPrefix(parser, currentConstrainedOut)
                                d_11_repairedCurrent_ = out7_
                                currentConstrainedOut = d_11_repairedCurrent_
                                generated = (d_10_stablePrefixRepair_) + (currentConstrainedOut)
                            elif True:
                                d_12_stablePrefix_: _dafny.Seq
                                d_12_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                d_13_constrainedPrompt_: _dafny.Seq
                                d_13_constrainedPrompt_ = (prompt) + (d_12_stablePrefix_)
                                (lm).GenerateLogits((d_13_constrainedPrompt_) + (currentConstrainedOut))
                                if (len(validTokenGroups)) > (0):
                                    d_14_candidates_: _dafny.Seq
                                    out8_: _dafny.Seq
                                    out8_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_13_constrainedPrompt_, currentConstrainedOut, 32, eosToken)
                                    d_14_candidates_ = out8_
                                    d_15_flatPreferred_: _dafny.Seq
                                    out9_: _dafny.Seq
                                    out9_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
                                    d_15_flatPreferred_ = out9_
                                    d_16_preferred_: _dafny.Seq
                                    out10_: _dafny.Seq
                                    out10_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_14_candidates_, d_15_flatPreferred_)
                                    d_16_preferred_ = out10_
                                    if (len(d_16_preferred_)) > (0):
                                        (d_0_helpers_).BoostTokenLogits(lm, d_16_preferred_, _dafny.BigRational('4e0'))
                                        d_17_groupedCandidates_: _dafny.Seq
                                        out11_: _dafny.Seq
                                        out11_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_14_candidates_, d_15_flatPreferred_)
                                        d_17_groupedCandidates_ = out11_
                                        d_18_otherCandidates_: _dafny.Seq
                                        out12_: _dafny.Seq
                                        out12_ = VerifiedDecoderAgent.CSDHelpers.SubtractTokenSets(d_17_groupedCandidates_, d_16_preferred_)
                                        d_18_otherCandidates_ = out12_
                                        if (len(d_18_otherCandidates_)) > (0):
                                            (d_0_helpers_).PenalizeTokenLogits(lm, d_18_otherCandidates_, _dafny.BigRational('1e0'))
                                d_19_symbolBudget_: int
                                d_19_symbolBudget_ = stepTokenBudget
                                if (d_19_symbolBudget_) == (0):
                                    d_19_symbolBudget_ = 1
                                d_20_remaining_: int
                                d_20_remaining_ = (maxSteps) - (d_1_steps_)
                                if (d_19_symbolBudget_) > (d_20_remaining_):
                                    d_19_symbolBudget_ = d_20_remaining_
                                d_21_symbolCurrent_: _dafny.Seq
                                d_22_hitEos_: bool
                                d_23_symbolSteps_: int
                                out13_: _dafny.Seq
                                out14_: bool
                                out15_: int
                                out13_, out14_, out15_ = (d_0_helpers_).ConstrainedSymbol(lm, parser, d_13_constrainedPrompt_, currentConstrainedOut, d_19_symbolBudget_, eosToken)
                                d_21_symbolCurrent_ = out13_
                                d_22_hitEos_ = out14_
                                d_23_symbolSteps_ = out15_
                                if d_22_hitEos_:
                                    d_1_steps_ = (d_1_steps_) + (d_23_symbolSteps_)
                                    raise _dafny.Break("0")
                                elif True:
                                    if (d_23_symbolSteps_) == (0):
                                        raise _dafny.Break("0")
                                    elif True:
                                        d_1_steps_ = (d_1_steps_) + (d_23_symbolSteps_)
                                        currentConstrainedOut = d_21_symbolCurrent_
                                        generated = (d_12_stablePrefix_) + (currentConstrainedOut)
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

