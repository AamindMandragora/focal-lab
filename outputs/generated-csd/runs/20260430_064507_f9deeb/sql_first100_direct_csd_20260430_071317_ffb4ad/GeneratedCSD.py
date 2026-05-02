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
        d_3_openSpanToken_: _dafny.Seq
        d_3_openSpanToken_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))
        d_4_fromKeyword_: _dafny.Seq
        d_4_fromKeyword_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "FROM"))
        d_5_selectKeyword_: _dafny.Seq
        d_5_selectKeyword_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SELECT"))
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_6_chunkBudget_: int
                        d_6_chunkBudget_ = (maxSteps) - (d_1_steps_)
                        d_7_chunkedGenerated_: _dafny.Seq
                        d_8_stoppedOnOpenSpan_: bool
                        d_9_stoppedOnEos_: bool
                        d_10_stepsUsed_: int
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: bool
                        out3_: int
                        out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_6_chunkBudget_, d_3_openSpanToken_, eosToken)
                        d_7_chunkedGenerated_ = out0_
                        d_8_stoppedOnOpenSpan_ = out1_
                        d_9_stoppedOnEos_ = out2_
                        d_10_stepsUsed_ = out3_
                        generated = d_7_chunkedGenerated_
                        d_1_steps_ = (d_1_steps_) + (d_10_stepsUsed_)
                        if d_9_stoppedOnEos_:
                            raise _dafny.Break("0")
                        elif d_8_stoppedOnOpenSpan_:
                            insideConstrainedOut = True
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        if (parser).IsCompletePrefix(currentConstrainedOut):
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
                            d_14_stablePrefix_: _dafny.Seq
                            d_14_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_15_constrainedPrompt_: _dafny.Seq
                            d_15_constrainedPrompt_ = (prompt) + (d_14_stablePrefix_)
                            d_16_validCount_: int
                            out7_: int
                            out7_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                            d_16_validCount_ = out7_
                            if (d_16_validCount_) <= (d_2_narrowThreshold_):
                                d_17_nextTight_: _dafny.Seq
                                out8_: _dafny.Seq
                                out8_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_15_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_17_nextTight_ = out8_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_17_nextTight_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_18_appendedGenerated1_: _dafny.Seq
                                    d_19_appendedInside1_: bool
                                    d_20_appendedCurrent1_: _dafny.Seq
                                    out9_: _dafny.Seq
                                    out10_: bool
                                    out11_: _dafny.Seq
                                    out9_, out10_, out11_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_17_nextTight_)
                                    d_18_appendedGenerated1_ = out9_
                                    d_19_appendedInside1_ = out10_
                                    d_20_appendedCurrent1_ = out11_
                                    generated = d_18_appendedGenerated1_
                                    insideConstrainedOut = d_19_appendedInside1_
                                    currentConstrainedOut = d_20_appendedCurrent1_
                            elif True:
                                d_21_fromContext_: _dafny.Seq
                                out12_: _dafny.Seq
                                out12_ = VerifiedDecoderAgent.CSDHelpers.ExtractAfterKeyword(currentConstrainedOut, d_4_fromKeyword_)
                                d_21_fromContext_ = out12_
                                d_22_selectContext_: _dafny.Seq
                                out13_: _dafny.Seq
                                out13_ = VerifiedDecoderAgent.CSDHelpers.ExtractAfterKeyword(currentConstrainedOut, d_5_selectKeyword_)
                                d_22_selectContext_ = out13_
                                d_23_flatPreferred_: _dafny.Seq
                                out14_: _dafny.Seq
                                out14_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
                                d_23_flatPreferred_ = out14_
                                (lm).GenerateLogits((d_15_constrainedPrompt_) + (currentConstrainedOut))
                                d_24_candidates_: _dafny.Seq
                                out15_: _dafny.Seq
                                out15_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_15_constrainedPrompt_, currentConstrainedOut, 24, eosToken)
                                d_24_candidates_ = out15_
                                if (len(d_21_fromContext_)) > (0):
                                    d_25_fromFocused_: _dafny.Seq
                                    out16_: _dafny.Seq
                                    out16_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_24_candidates_, d_21_fromContext_)
                                    d_25_fromFocused_ = out16_
                                    if (len(d_25_fromFocused_)) > (0):
                                        (d_0_helpers_).BoostTokenLogits(lm, d_25_fromFocused_, _dafny.BigRational('8e0'))
                                if (len(d_22_selectContext_)) > (0):
                                    d_26_selectFocused_: _dafny.Seq
                                    out17_: _dafny.Seq
                                    out17_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_24_candidates_, d_22_selectContext_)
                                    d_26_selectFocused_ = out17_
                                    if (len(d_26_selectFocused_)) > (0):
                                        (d_0_helpers_).BoostTokenLogits(lm, d_26_selectFocused_, _dafny.BigRational('3e0'))
                                if (len(d_23_flatPreferred_)) > (0):
                                    d_27_anyPreferredValid_: bool
                                    out18_: bool
                                    out18_ = (d_0_helpers_).GroupHasValidMember(parser, currentConstrainedOut, d_23_flatPreferred_)
                                    d_27_anyPreferredValid_ = out18_
                                    if d_27_anyPreferredValid_:
                                        d_28_preferredFocused_: _dafny.Seq
                                        out19_: _dafny.Seq
                                        out19_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_24_candidates_, d_23_flatPreferred_)
                                        d_28_preferredFocused_ = out19_
                                        if (len(d_28_preferredFocused_)) > (0):
                                            (d_0_helpers_).BoostTokenLogits(lm, d_28_preferredFocused_, _dafny.BigRational('5e0'))
                                (lm).MaskValidNextAndEos(parser, currentConstrainedOut, eosToken)
                                d_29_nextSoft_: _dafny.Seq
                                out20_: _dafny.Seq
                                out20_ = (lm).ChooseNextToken()
                                d_29_nextSoft_ = out20_
                                (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_29_nextSoft_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_30_appendedGenerated2_: _dafny.Seq
                                    d_31_appendedInside2_: bool
                                    d_32_appendedCurrent2_: _dafny.Seq
                                    out21_: _dafny.Seq
                                    out22_: bool
                                    out23_: _dafny.Seq
                                    out21_, out22_, out23_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_29_nextSoft_)
                                    d_30_appendedGenerated2_ = out21_
                                    d_31_appendedInside2_ = out22_
                                    d_32_appendedCurrent2_ = out23_
                                    generated = d_30_appendedGenerated2_
                                    insideConstrainedOut = d_31_appendedInside2_
                                    currentConstrainedOut = d_32_appendedCurrent2_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

