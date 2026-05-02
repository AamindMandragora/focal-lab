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
        d_3_scopeKeyword_: _dafny.Seq
        d_3_scopeKeyword_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "FROM"))
        d_4_flatPreferred_: _dafny.Seq
        out0_: _dafny.Seq
        out0_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
        d_4_flatPreferred_ = out0_
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
                        out1_: _dafny.Seq
                        out2_: bool
                        out3_: bool
                        out4_: int
                        out1_, out2_, out3_, out4_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_5_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_6_chunkedGenerated_ = out1_
                        d_7_stoppedOnOpenSpan_ = out2_
                        d_8_stoppedOnEos_ = out3_
                        d_9_stepsUsed_ = out4_
                        generated = d_6_chunkedGenerated_
                        d_1_steps_ = (d_1_steps_) + (d_9_stepsUsed_)
                        if d_8_stoppedOnEos_:
                            raise _dafny.Break("0")
                        elif d_7_stoppedOnOpenSpan_:
                            insideConstrainedOut = True
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        d_10_complete_: bool
                        d_10_complete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_10_complete_:
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
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_14_stablePrefix_: _dafny.Seq
                            d_14_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_15_constrainedPrompt_: _dafny.Seq
                            d_15_constrainedPrompt_ = (prompt) + (d_14_stablePrefix_)
                            d_16_validCount_: int
                            out8_: int
                            out8_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                            d_16_validCount_ = out8_
                            if (d_16_validCount_) <= (d_2_narrowThreshold_):
                                d_17_scopeTokens_: _dafny.Seq
                                out9_: _dafny.Seq
                                out9_ = VerifiedDecoderAgent.CSDHelpers.ExtractAfterKeyword(currentConstrainedOut, d_3_scopeKeyword_)
                                d_17_scopeTokens_ = out9_
                                (lm).GenerateLogits((d_15_constrainedPrompt_) + (currentConstrainedOut))
                                if (len(d_4_flatPreferred_)) > (0):
                                    d_18_candidates1_: _dafny.Seq
                                    out10_: _dafny.Seq
                                    out10_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_15_constrainedPrompt_, currentConstrainedOut, 24, eosToken)
                                    d_18_candidates1_ = out10_
                                    d_19_preferredNow_: _dafny.Seq
                                    out11_: _dafny.Seq
                                    out11_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_18_candidates1_, d_4_flatPreferred_)
                                    d_19_preferredNow_ = out11_
                                    if (len(d_19_preferredNow_)) > (0):
                                        (d_0_helpers_).BoostTokenLogits(lm, d_19_preferredNow_, _dafny.BigRational('5e0'))
                                if (len(d_17_scopeTokens_)) > (0):
                                    d_20_candidates2_: _dafny.Seq
                                    out12_: _dafny.Seq
                                    out12_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_15_constrainedPrompt_, currentConstrainedOut, 24, eosToken)
                                    d_20_candidates2_ = out12_
                                    d_21_scopedNow_: _dafny.Seq
                                    out13_: _dafny.Seq
                                    out13_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_20_candidates2_, d_17_scopeTokens_)
                                    d_21_scopedNow_ = out13_
                                    if (len(d_21_scopedNow_)) > (0):
                                        (d_0_helpers_).BoostTokenLogits(lm, d_21_scopedNow_, _dafny.BigRational('6e0'))
                                (lm).MaskValidNextAndEos(parser, currentConstrainedOut, eosToken)
                                d_22_next_: _dafny.Seq
                                out14_: _dafny.Seq
                                out14_ = (lm).ChooseNextToken()
                                d_22_next_ = out14_
                                (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_22_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_23_appendedGenerated_: _dafny.Seq
                                    d_24_appendedInside_: bool
                                    d_25_appendedCurrent_: _dafny.Seq
                                    out15_: _dafny.Seq
                                    out16_: bool
                                    out17_: _dafny.Seq
                                    out15_, out16_, out17_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_22_next_)
                                    d_23_appendedGenerated_ = out15_
                                    d_24_appendedInside_ = out16_
                                    d_25_appendedCurrent_ = out17_
                                    generated = d_23_appendedGenerated_
                                    insideConstrainedOut = d_24_appendedInside_
                                    currentConstrainedOut = d_25_appendedCurrent_
                            elif True:
                                d_26_remaining_: int
                                d_26_remaining_ = (maxSteps) - (d_1_steps_)
                                if (d_26_remaining_) == (0):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_27_budget_: int
                                    d_27_budget_ = stepTokenBudget
                                    if (d_27_budget_) > (d_26_remaining_):
                                        d_27_budget_ = d_26_remaining_
                                    if (d_27_budget_) == (0):
                                        raise _dafny.Break("0")
                                    elif True:
                                        d_28_symbolOut_: _dafny.Seq
                                        d_29_hitEos_: bool
                                        d_30_stepsUsed2_: int
                                        out18_: _dafny.Seq
                                        out19_: bool
                                        out20_: int
                                        out18_, out19_, out20_ = (d_0_helpers_).ConstrainedSymbol(lm, parser, d_15_constrainedPrompt_, currentConstrainedOut, d_27_budget_, eosToken)
                                        d_28_symbolOut_ = out18_
                                        d_29_hitEos_ = out19_
                                        d_30_stepsUsed2_ = out20_
                                        generated = (d_14_stablePrefix_) + (d_28_symbolOut_)
                                        currentConstrainedOut = d_28_symbolOut_
                                        d_1_steps_ = (d_1_steps_) + (d_30_stepsUsed2_)
                                        if d_29_hitEos_:
                                            raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

