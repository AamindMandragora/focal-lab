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
        d_2_wideThreshold_: int
        d_2_wideThreshold_ = 6
        d_3_mildBoost_: _dafny.BigRational
        d_3_mildBoost_ = _dafny.BigRational('15e-1')
        d_4_repeatPenalty_: _dafny.BigRational
        d_4_repeatPenalty_ = _dafny.BigRational('2e0')
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_5_chunkBudget_: int
                        d_5_chunkBudget_ = (maxSteps) - (d_1_steps_)
                        d_6_chunkedGenerated_: _dafny.Seq
                        d_7_stoppedOnOpenSpan_: bool
                        d_8_stoppedOnEos_: bool
                        d_9_stepsUsed_: int
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: bool
                        out3_: int
                        out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_5_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_6_chunkedGenerated_ = out0_
                        d_7_stoppedOnOpenSpan_ = out1_
                        d_8_stoppedOnEos_ = out2_
                        d_9_stepsUsed_ = out3_
                        generated = d_6_chunkedGenerated_
                        d_1_steps_ = (d_1_steps_) + (d_9_stepsUsed_)
                        if d_8_stoppedOnEos_:
                            raise _dafny.Break("0")
                        elif True:
                            if d_7_stoppedOnOpenSpan_:
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        d_10_isComplete_: bool
                        d_10_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_10_isComplete_:
                            d_11_closedGenerated_: _dafny.Seq
                            d_12_closedInside_: bool
                            d_13_closedCurrent_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_11_closedGenerated_ = out4_
                            d_12_closedInside_ = out5_
                            d_13_closedCurrent_ = out6_
                            generated = d_11_closedGenerated_
                            insideConstrainedOut = d_12_closedInside_
                            currentConstrainedOut = d_13_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_14_validCount_: int
                            out7_: int
                            out7_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                            d_14_validCount_ = out7_
                            d_15_stablePrefix_: _dafny.Seq
                            d_15_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_16_constrainedPrompt_: _dafny.Seq
                            d_16_constrainedPrompt_ = (prompt) + (d_15_stablePrefix_)
                            d_17_remaining_: int
                            d_17_remaining_ = (maxSteps) - (d_1_steps_)
                            if (((d_14_validCount_) >= (d_2_wideThreshold_)) and ((stepTokenBudget) > (0))) and ((d_17_remaining_) > (0)):
                                d_18_budget_: int
                                d_18_budget_ = stepTokenBudget
                                if (d_18_budget_) > (d_17_remaining_):
                                    d_18_budget_ = d_17_remaining_
                                d_19_symbolOut_: _dafny.Seq
                                d_20_hitEos_: bool
                                d_21_stepsUsed2_: int
                                out8_: _dafny.Seq
                                out9_: bool
                                out10_: int
                                out8_, out9_, out10_ = (d_0_helpers_).ConstrainedSymbol(lm, parser, d_16_constrainedPrompt_, currentConstrainedOut, d_18_budget_, eosToken)
                                d_19_symbolOut_ = out8_
                                d_20_hitEos_ = out9_
                                d_21_stepsUsed2_ = out10_
                                generated = (d_15_stablePrefix_) + (d_19_symbolOut_)
                                currentConstrainedOut = d_19_symbolOut_
                                d_1_steps_ = (d_1_steps_) + (d_21_stepsUsed2_)
                                if d_20_hitEos_:
                                    raise _dafny.Break("0")
                            elif True:
                                (lm).GenerateLogits((d_16_constrainedPrompt_) + (currentConstrainedOut))
                                if (len(validTokenGroups)) > (0):
                                    d_22_flatPreferred_: _dafny.Seq
                                    out11_: _dafny.Seq
                                    out11_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
                                    d_22_flatPreferred_ = out11_
                                    d_23_candidates_: _dafny.Seq
                                    out12_: _dafny.Seq
                                    out12_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_16_constrainedPrompt_, currentConstrainedOut, 16, eosToken)
                                    d_23_candidates_ = out12_
                                    d_24_preferred_: _dafny.Seq
                                    out13_: _dafny.Seq
                                    out13_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_23_candidates_, d_22_flatPreferred_)
                                    d_24_preferred_ = out13_
                                    if (len(d_24_preferred_)) > (0):
                                        (d_0_helpers_).BoostTokenLogits(lm, d_24_preferred_, d_3_mildBoost_)
                                if (len(currentConstrainedOut)) > (0):
                                    d_25_lastTok_: _dafny.Seq
                                    d_25_lastTok_ = (currentConstrainedOut)[(len(currentConstrainedOut)) - (1)]
                                    d_26_lastValid_: bool
                                    out14_: bool
                                    out14_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_25_lastTok_)
                                    d_26_lastValid_ = out14_
                                    d_27_count2_: int
                                    out15_: int
                                    out15_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                                    d_27_count2_ = out15_
                                    if (d_26_lastValid_) and ((d_27_count2_) > (1)):
                                        if (d_25_lastTok_) in ((lm).Tokens):
                                            (d_0_helpers_).PenalizeTokenLogits(lm, _dafny.SeqWithoutIsStrInference([d_25_lastTok_]), d_4_repeatPenalty_)
                                (lm).MaskValidNextAndEos(parser, currentConstrainedOut, eosToken)
                                d_28_next_: _dafny.Seq
                                out16_: _dafny.Seq
                                out16_ = (lm).ChooseNextToken()
                                d_28_next_ = out16_
                                (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_28_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_29_appendedGenerated_: _dafny.Seq
                                    d_30_appendedInside_: bool
                                    d_31_appendedCurrent_: _dafny.Seq
                                    out17_: _dafny.Seq
                                    out18_: bool
                                    out19_: _dafny.Seq
                                    out17_, out18_, out19_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_28_next_)
                                    d_29_appendedGenerated_ = out17_
                                    d_30_appendedInside_ = out18_
                                    d_31_appendedCurrent_ = out19_
                                    generated = d_29_appendedGenerated_
                                    insideConstrainedOut = d_30_appendedInside_
                                    currentConstrainedOut = d_31_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

