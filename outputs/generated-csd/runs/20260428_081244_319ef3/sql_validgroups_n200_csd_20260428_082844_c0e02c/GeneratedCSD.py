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
        insideConstrainedOut = True
        if insideConstrained:
            currentConstrainedOut = currentConstrained
        elif True:
            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
        if not(insideConstrained):
            generated = generatedPrefix
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
                    d_6_completeNow_: bool
                    d_6_completeNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                    if d_6_completeNow_:
                        d_7_closedGenerated_: _dafny.Seq
                        d_8_closedInside_: bool
                        d_9_closedCurrent_: _dafny.Seq
                        out1_: _dafny.Seq
                        out2_: bool
                        out3_: _dafny.Seq
                        out1_, out2_, out3_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_7_closedGenerated_ = out1_
                        d_8_closedInside_ = out2_
                        d_9_closedCurrent_ = out3_
                        generated = d_7_closedGenerated_
                        insideConstrainedOut = d_8_closedInside_
                        currentConstrainedOut = d_9_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                        raise _dafny.Break("0")
                    elif True:
                        d_10_deadEnd_: bool
                        out4_: bool
                        out4_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                        d_10_deadEnd_ = out4_
                        if d_10_deadEnd_:
                            d_11_repaired_: _dafny.Seq
                            out5_: _dafny.Seq
                            out5_ = VerifiedDecoderAgent.CSDHelpers.RollbackToBoundary(parser, currentConstrainedOut, d_2_selectTok_)
                            d_11_repaired_ = out5_
                            d_12_repairedFrom_: _dafny.Seq
                            out6_: _dafny.Seq
                            out6_ = VerifiedDecoderAgent.CSDHelpers.RollbackToBoundary(parser, currentConstrainedOut, d_3_fromTok_)
                            d_12_repairedFrom_ = out6_
                            if (len(d_12_repairedFrom_)) < (len(d_11_repaired_)):
                                d_11_repaired_ = d_12_repairedFrom_
                            d_13_repairedWhere_: _dafny.Seq
                            out7_: _dafny.Seq
                            out7_ = VerifiedDecoderAgent.CSDHelpers.RollbackToBoundary(parser, currentConstrainedOut, d_4_whereTok_)
                            d_13_repairedWhere_ = out7_
                            if (len(d_13_repairedWhere_)) < (len(d_11_repaired_)):
                                d_11_repaired_ = d_13_repairedWhere_
                            d_14_prevTok_: _dafny.Seq
                            d_15_foundPrev_: bool
                            out8_: _dafny.Seq
                            out9_: bool
                            out8_, out9_ = (d_0_helpers_).LastTokenBefore(currentConstrainedOut, d_3_fromTok_)
                            d_14_prevTok_ = out8_
                            d_15_foundPrev_ = out9_
                            if d_15_foundPrev_:
                                d_16_grpIdx_: int
                                out10_: int
                                out10_ = VerifiedDecoderAgent.CSDHelpers.GroupContaining(validTokenGroups, d_14_prevTok_)
                                d_16_grpIdx_ = out10_
                                if (d_16_grpIdx_) >= (0):
                                    d_17_repairedHint_: _dafny.Seq
                                    out11_: _dafny.Seq
                                    out11_ = VerifiedDecoderAgent.CSDHelpers.RollbackToBoundary(parser, currentConstrainedOut, d_14_prevTok_)
                                    d_17_repairedHint_ = out11_
                                    if (len(d_17_repairedHint_)) < (len(d_11_repaired_)):
                                        d_11_repaired_ = d_17_repairedHint_
                            d_18_prevTok2_: _dafny.Seq
                            d_19_foundPrev2_: bool
                            out12_: _dafny.Seq
                            out13_: bool
                            out12_, out13_ = (d_0_helpers_).LastTokenBefore(currentConstrainedOut, d_4_whereTok_)
                            d_18_prevTok2_ = out12_
                            d_19_foundPrev2_ = out13_
                            if d_19_foundPrev2_:
                                d_20_inFlat_: bool
                                d_20_inFlat_ = (d_18_prevTok2_) in (d_5_flatGroups_)
                                if d_20_inFlat_:
                                    d_21_repairedHint2_: _dafny.Seq
                                    out14_: _dafny.Seq
                                    out14_ = VerifiedDecoderAgent.CSDHelpers.RollbackToBoundary(parser, currentConstrainedOut, d_18_prevTok2_)
                                    d_21_repairedHint2_ = out14_
                                    if (len(d_21_repairedHint2_)) < (len(d_11_repaired_)):
                                        d_11_repaired_ = d_21_repairedHint2_
                            if (len(d_11_repaired_)) == (len(currentConstrainedOut)):
                                raise _dafny.Break("0")
                            elif True:
                                d_22_stablePrefix_: _dafny.Seq
                                d_22_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                generated = (d_22_stablePrefix_) + (d_11_repaired_)
                                currentConstrainedOut = d_11_repaired_
                        elif True:
                            d_23_stablePrefix2_: _dafny.Seq
                            d_23_stablePrefix2_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_24_constrainedPrompt_: _dafny.Seq
                            d_24_constrainedPrompt_ = (prompt) + (d_23_stablePrefix2_)
                            d_25_budget_: int
                            d_25_budget_ = stepTokenBudget
                            if (d_25_budget_) > ((maxSteps) - (d_1_steps_)):
                                d_25_budget_ = (maxSteps) - (d_1_steps_)
                            if (d_25_budget_) == (0):
                                raise _dafny.Break("0")
                            elif True:
                                d_26_currentOut_: _dafny.Seq
                                d_27_hitEos_: bool
                                d_28_stepsUsed_: int
                                out15_: _dafny.Seq
                                out16_: bool
                                out17_: int
                                out15_, out16_, out17_ = (d_0_helpers_).ConstrainedSymbol(lm, parser, d_24_constrainedPrompt_, currentConstrainedOut, d_25_budget_, eosToken)
                                d_26_currentOut_ = out15_
                                d_27_hitEos_ = out16_
                                d_28_stepsUsed_ = out17_
                                d_1_steps_ = (d_1_steps_) + (d_28_stepsUsed_)
                                if d_27_hitEos_:
                                    raise _dafny.Break("0")
                                elif True:
                                    currentConstrainedOut = d_26_currentOut_
                                    generated = (d_23_stablePrefix2_) + (currentConstrainedOut)
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

