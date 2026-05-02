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
        d_3_wideBudget_: int
        if (stepTokenBudget) == (0):
            d_3_wideBudget_ = 1
        elif True:
            d_3_wideBudget_ = stepTokenBudget
        d_4_activeGroup_: int
        d_4_activeGroup_ = -1
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_5_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_5_next_ = out0_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_5_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_5_next_]))
                            if (d_5_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                d_4_activeGroup_ = -1
                    elif True:
                        if (parser).IsCompletePrefix(currentConstrainedOut):
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
                            d_4_activeGroup_ = -1
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
                            if ((d_11_validCount_) > (d_2_narrowThreshold_)) and ((d_3_wideBudget_) <= ((maxSteps) - (d_1_steps_))):
                                d_12_symbolOut_: _dafny.Seq
                                d_13_hitEos_: bool
                                d_14_stepsUsed_: int
                                out5_: _dafny.Seq
                                out6_: bool
                                out7_: int
                                out5_, out6_, out7_ = (d_0_helpers_).ConstrainedSymbol(lm, parser, d_10_constrainedPrompt_, currentConstrainedOut, d_3_wideBudget_, eosToken)
                                d_12_symbolOut_ = out5_
                                d_13_hitEos_ = out6_
                                d_14_stepsUsed_ = out7_
                                generated = (d_9_stablePrefix_) + (d_12_symbolOut_)
                                currentConstrainedOut = d_12_symbolOut_
                                d_1_steps_ = (d_1_steps_) + (d_14_stepsUsed_)
                                if (len(currentConstrainedOut)) > (0):
                                    d_15_lastTok_: _dafny.Seq
                                    d_16_found_: bool
                                    out8_: _dafny.Seq
                                    out9_: bool
                                    out8_, out9_ = (d_0_helpers_).LastTokenBefore((currentConstrainedOut) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ","))])), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ",")))
                                    d_15_lastTok_ = out8_
                                    d_16_found_ = out9_
                                    if d_16_found_:
                                        d_17_idx_: int
                                        out10_: int
                                        out10_ = VerifiedDecoderAgent.CSDHelpers.GroupContaining(validTokenGroups, d_15_lastTok_)
                                        d_17_idx_ = out10_
                                        d_4_activeGroup_ = d_17_idx_
                                if d_13_hitEos_:
                                    raise _dafny.Break("0")
                            elif True:
                                (lm).GenerateLogits((d_10_constrainedPrompt_) + (currentConstrainedOut))
                                if (len(validTokenGroups)) > (0):
                                    d_18_flatPreferred_: _dafny.Seq
                                    out11_: _dafny.Seq
                                    out11_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
                                    d_18_flatPreferred_ = out11_
                                    if (len(d_18_flatPreferred_)) > (0):
                                        d_19_anyPreferredValid_: bool
                                        out12_: bool
                                        out12_ = (d_0_helpers_).GroupHasValidMember(parser, currentConstrainedOut, d_18_flatPreferred_)
                                        d_19_anyPreferredValid_ = out12_
                                        if d_19_anyPreferredValid_:
                                            d_20_topCandidates_: _dafny.Seq
                                            out13_: _dafny.Seq
                                            out13_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_10_constrainedPrompt_, currentConstrainedOut, 24, eosToken)
                                            d_20_topCandidates_ = out13_
                                            d_21_preferredCandidates_: _dafny.Seq
                                            out14_: _dafny.Seq
                                            out14_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_20_topCandidates_, d_18_flatPreferred_)
                                            d_21_preferredCandidates_ = out14_
                                            if (len(d_21_preferredCandidates_)) > (0):
                                                (d_0_helpers_).BoostTokenLogits(lm, d_21_preferredCandidates_, _dafny.BigRational('3e0'))
                                    if (0) <= (d_4_activeGroup_):
                                        d_22_activeTokens_: _dafny.Seq
                                        d_22_activeTokens_ = (validTokenGroups)[d_4_activeGroup_]
                                        d_23_anyActiveValid_: bool
                                        out15_: bool
                                        out15_ = (d_0_helpers_).GroupHasValidMember(parser, currentConstrainedOut, d_22_activeTokens_)
                                        d_23_anyActiveValid_ = out15_
                                        if d_23_anyActiveValid_:
                                            d_24_topCandidates2_: _dafny.Seq
                                            out16_: _dafny.Seq
                                            out16_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_10_constrainedPrompt_, currentConstrainedOut, 24, eosToken)
                                            d_24_topCandidates2_ = out16_
                                            d_25_focusedCandidates_: _dafny.Seq
                                            out17_: _dafny.Seq
                                            out17_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_24_topCandidates2_, d_22_activeTokens_)
                                            d_25_focusedCandidates_ = out17_
                                            if (len(d_25_focusedCandidates_)) > (0):
                                                (d_0_helpers_).BoostTokenLogits(lm, d_25_focusedCandidates_, _dafny.BigRational('8e0'))
                                (lm).MaskValidNextAndEos(parser, currentConstrainedOut, eosToken)
                                d_26_next_: _dafny.Seq
                                out18_: _dafny.Seq
                                out18_ = (lm).ChooseNextToken()
                                d_26_next_ = out18_
                                (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_26_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_27_appendedGenerated_: _dafny.Seq
                                    d_28_appendedInside_: bool
                                    d_29_appendedCurrent_: _dafny.Seq
                                    out19_: _dafny.Seq
                                    out20_: bool
                                    out21_: _dafny.Seq
                                    out19_, out20_, out21_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_26_next_)
                                    d_27_appendedGenerated_ = out19_
                                    d_28_appendedInside_ = out20_
                                    d_29_appendedCurrent_ = out21_
                                    generated = d_27_appendedGenerated_
                                    insideConstrainedOut = d_28_appendedInside_
                                    currentConstrainedOut = d_29_appendedCurrent_
                                    d_30_idx2_: int
                                    out22_: int
                                    out22_ = VerifiedDecoderAgent.CSDHelpers.GroupContaining(validTokenGroups, d_26_next_)
                                    d_30_idx2_ = out22_
                                    if (d_30_idx2_) >= (0):
                                        d_4_activeGroup_ = d_30_idx2_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

