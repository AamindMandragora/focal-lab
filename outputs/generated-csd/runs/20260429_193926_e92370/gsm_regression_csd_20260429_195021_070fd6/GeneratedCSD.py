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
        d_2_narrowThreshold_ = 8
        d_3_equalsToken_: _dafny.Seq
        d_3_equalsToken_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "="))
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_4_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_4_next_ = out0_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_4_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_4_next_]))
                            if (d_4_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        d_5_isComplete_: bool
                        d_5_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_5_isComplete_:
                            d_6_closedGenerated_: _dafny.Seq
                            d_7_closedInside_: bool
                            d_8_closedCurrent_: _dafny.Seq
                            out1_: _dafny.Seq
                            out2_: bool
                            out3_: _dafny.Seq
                            out1_, out2_, out3_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_6_closedGenerated_ = out1_
                            d_7_closedInside_ = out2_
                            d_8_closedCurrent_ = out3_
                            generated = d_6_closedGenerated_
                            insideConstrainedOut = d_7_closedInside_
                            currentConstrainedOut = d_8_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_9_stablePrefix_: _dafny.Seq
                            d_9_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_10_constrainedPrompt_: _dafny.Seq
                            d_10_constrainedPrompt_ = (prompt) + (d_9_stablePrefix_)
                            d_11_validCount_: int
                            out4_: int
                            out4_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                            d_11_validCount_ = out4_
                            d_12_afterEquals_: _dafny.Seq
                            out5_: _dafny.Seq
                            out5_ = VerifiedDecoderAgent.CSDHelpers.ExtractAfterKeyword(currentConstrainedOut, d_3_equalsToken_)
                            d_12_afterEquals_ = out5_
                            if (d_11_validCount_) <= (d_2_narrowThreshold_):
                                d_13_next_: _dafny.Seq
                                out6_: _dafny.Seq
                                out6_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_10_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_13_next_ = out6_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_13_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_14_appendedGenerated_: _dafny.Seq
                                    d_15_appendedInside_: bool
                                    d_16_appendedCurrent_: _dafny.Seq
                                    out7_: _dafny.Seq
                                    out8_: bool
                                    out9_: _dafny.Seq
                                    out7_, out8_, out9_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_13_next_)
                                    d_14_appendedGenerated_ = out7_
                                    d_15_appendedInside_ = out8_
                                    d_16_appendedCurrent_ = out9_
                                    generated = d_14_appendedGenerated_
                                    insideConstrainedOut = d_15_appendedInside_
                                    currentConstrainedOut = d_16_appendedCurrent_
                            elif True:
                                (lm).GenerateLogits((d_10_constrainedPrompt_) + (currentConstrainedOut))
                                if (len(d_12_afterEquals_)) > (0):
                                    d_17_candidates_: _dafny.Seq
                                    out10_: _dafny.Seq
                                    out10_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_10_constrainedPrompt_, currentConstrainedOut, 20, eosToken)
                                    d_17_candidates_ = out10_
                                    d_18_focused_: _dafny.Seq
                                    out11_: _dafny.Seq
                                    out11_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_17_candidates_, d_12_afterEquals_)
                                    d_18_focused_ = out11_
                                    if (len(d_18_focused_)) > (0):
                                        (d_0_helpers_).BoostTokenLogits(lm, d_18_focused_, _dafny.BigRational('4e0'))
                                if (len(validTokenGroups)) > (0):
                                    d_19_flatPreferred_: _dafny.Seq
                                    out12_: _dafny.Seq
                                    out12_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
                                    d_19_flatPreferred_ = out12_
                                    if (len(d_19_flatPreferred_)) > (0):
                                        d_20_anyValid_: bool
                                        out13_: bool
                                        out13_ = (d_0_helpers_).GroupHasValidMember(parser, currentConstrainedOut, d_19_flatPreferred_)
                                        d_20_anyValid_ = out13_
                                        if d_20_anyValid_:
                                            d_21_candidates2_: _dafny.Seq
                                            out14_: _dafny.Seq
                                            out14_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_10_constrainedPrompt_, currentConstrainedOut, 20, eosToken)
                                            d_21_candidates2_ = out14_
                                            d_22_preferred_: _dafny.Seq
                                            out15_: _dafny.Seq
                                            out15_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_21_candidates2_, d_19_flatPreferred_)
                                            d_22_preferred_ = out15_
                                            if (len(d_22_preferred_)) > (0):
                                                (d_0_helpers_).BoostTokenLogits(lm, d_22_preferred_, _dafny.BigRational('3e0'))
                                d_23_remaining_: int
                                d_23_remaining_ = (maxSteps) - (d_1_steps_)
                                d_24_budget_: int
                                d_24_budget_ = stepTokenBudget
                                if (d_24_budget_) == (0):
                                    d_24_budget_ = 1
                                if (d_23_remaining_) < (d_24_budget_):
                                    d_24_budget_ = d_23_remaining_
                                d_25_symbolOut_: _dafny.Seq
                                d_26_hitEos_: bool
                                d_27_stepsUsed_: int
                                out16_: _dafny.Seq
                                out17_: bool
                                out18_: int
                                out16_, out17_, out18_ = (d_0_helpers_).ConstrainedSymbol(lm, parser, d_10_constrainedPrompt_, currentConstrainedOut, d_24_budget_, eosToken)
                                d_25_symbolOut_ = out16_
                                d_26_hitEos_ = out17_
                                d_27_stepsUsed_ = out18_
                                generated = (d_9_stablePrefix_) + (d_25_symbolOut_)
                                currentConstrainedOut = d_25_symbolOut_
                                d_1_steps_ = (d_1_steps_) + (d_27_stepsUsed_)
                                if d_26_hitEos_:
                                    raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

