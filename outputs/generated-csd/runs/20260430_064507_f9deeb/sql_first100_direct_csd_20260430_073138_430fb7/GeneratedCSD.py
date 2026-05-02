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
        d_2_openSpanToken_: _dafny.Seq
        d_2_openSpanToken_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))
        d_3_narrowThreshold_: int
        d_3_narrowThreshold_ = 6
        d_4_mediumThreshold_: int
        d_4_mediumThreshold_ = 18
        d_5_flatGroups_: _dafny.Seq
        out0_: _dafny.Seq
        out0_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
        d_5_flatGroups_ = out0_
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_6_chunkBudget_: int
                        d_6_chunkBudget_ = (maxSteps) - (d_1_steps_)
                        d_7_generatedChunk_: _dafny.Seq
                        d_8_stoppedOnOpenSpan_: bool
                        d_9_stoppedOnEos_: bool
                        d_10_stepsUsed_: int
                        out1_: _dafny.Seq
                        out2_: bool
                        out3_: bool
                        out4_: int
                        out1_, out2_, out3_, out4_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_6_chunkBudget_, d_2_openSpanToken_, eosToken)
                        d_7_generatedChunk_ = out1_
                        d_8_stoppedOnOpenSpan_ = out2_
                        d_9_stoppedOnEos_ = out3_
                        d_10_stepsUsed_ = out4_
                        generated = d_7_generatedChunk_
                        d_1_steps_ = (d_1_steps_) + (d_10_stepsUsed_)
                        if d_9_stoppedOnEos_:
                            raise _dafny.Break("0")
                        elif True:
                            if d_8_stoppedOnOpenSpan_:
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            d_11_generatedClosed_: _dafny.Seq
                            d_12_insideClosed_: bool
                            d_13_currentClosed_: _dafny.Seq
                            out5_: _dafny.Seq
                            out6_: bool
                            out7_: _dafny.Seq
                            out5_, out6_, out7_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_11_generatedClosed_ = out5_
                            d_12_insideClosed_ = out6_
                            d_13_currentClosed_ = out7_
                            generated = d_11_generatedClosed_
                            insideConstrainedOut = d_12_insideClosed_
                            currentConstrainedOut = d_13_currentClosed_
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
                            if (d_16_validCount_) <= (d_3_narrowThreshold_):
                                d_17_nextNarrow_: _dafny.Seq
                                out9_: _dafny.Seq
                                out9_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_15_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_17_nextNarrow_ = out9_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_17_nextNarrow_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_18_generatedApp1_: _dafny.Seq
                                    d_19_insideApp1_: bool
                                    d_20_currentApp1_: _dafny.Seq
                                    out10_: _dafny.Seq
                                    out11_: bool
                                    out12_: _dafny.Seq
                                    out10_, out11_, out12_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_17_nextNarrow_)
                                    d_18_generatedApp1_ = out10_
                                    d_19_insideApp1_ = out11_
                                    d_20_currentApp1_ = out12_
                                    generated = d_18_generatedApp1_
                                    insideConstrainedOut = d_19_insideApp1_
                                    currentConstrainedOut = d_20_currentApp1_
                            elif True:
                                d_21_candidates_: _dafny.Seq
                                out13_: _dafny.Seq
                                out13_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_15_constrainedPrompt_, currentConstrainedOut, 16, eosToken)
                                d_21_candidates_ = out13_
                                if (((d_16_validCount_) <= (d_4_mediumThreshold_)) or ((len(d_21_candidates_)) == (0))) or ((len(d_5_flatGroups_)) == (0)):
                                    d_22_nextMedium_: _dafny.Seq
                                    out14_: _dafny.Seq
                                    out14_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_15_constrainedPrompt_, currentConstrainedOut, eosToken)
                                    d_22_nextMedium_ = out14_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    if (d_22_nextMedium_) == (eosToken):
                                        raise _dafny.Break("0")
                                    elif True:
                                        d_23_generatedApp2_: _dafny.Seq
                                        d_24_insideApp2_: bool
                                        d_25_currentApp2_: _dafny.Seq
                                        out15_: _dafny.Seq
                                        out16_: bool
                                        out17_: _dafny.Seq
                                        out15_, out16_, out17_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_22_nextMedium_)
                                        d_23_generatedApp2_ = out15_
                                        d_24_insideApp2_ = out16_
                                        d_25_currentApp2_ = out17_
                                        generated = d_23_generatedApp2_
                                        insideConstrainedOut = d_24_insideApp2_
                                        currentConstrainedOut = d_25_currentApp2_
                                elif True:
                                    (lm).GenerateLogits((d_15_constrainedPrompt_) + (currentConstrainedOut))
                                    d_26_i_: int
                                    d_26_i_ = 0
                                    while (d_26_i_) < (len(validTokenGroups)):
                                        d_27_group_: _dafny.Seq
                                        d_27_group_ = (validTokenGroups)[d_26_i_]
                                        d_28_anyValid_: bool
                                        out18_: bool
                                        out18_ = (d_0_helpers_).GroupHasValidMember(parser, currentConstrainedOut, d_27_group_)
                                        d_28_anyValid_ = out18_
                                        if d_28_anyValid_:
                                            d_29_overlap_: _dafny.Seq
                                            out19_: _dafny.Seq
                                            out19_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_21_candidates_, d_27_group_)
                                            d_29_overlap_ = out19_
                                            if (len(d_29_overlap_)) > (0):
                                                (d_0_helpers_).BoostTokenLogits(lm, d_29_overlap_, _dafny.BigRational('12e0'))
                                        d_26_i_ = (d_26_i_) + (1)
                                    d_30_outsideHints_: _dafny.Seq
                                    out20_: _dafny.Seq
                                    out20_ = VerifiedDecoderAgent.CSDHelpers.SubtractTokenSets(d_21_candidates_, d_5_flatGroups_)
                                    d_30_outsideHints_ = out20_
                                    if (len(d_30_outsideHints_)) > (0):
                                        (d_0_helpers_).PenalizeTokenLogits(lm, d_30_outsideHints_, _dafny.BigRational('2e0'))
                                    (lm).MaskValidNextAndEos(parser, currentConstrainedOut, eosToken)
                                    d_31_nextBiased_: _dafny.Seq
                                    out21_: _dafny.Seq
                                    out21_ = (lm).ChooseNextToken()
                                    d_31_nextBiased_ = out21_
                                    (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    if (d_31_nextBiased_) == (eosToken):
                                        raise _dafny.Break("0")
                                    elif True:
                                        d_32_generatedApp3_: _dafny.Seq
                                        d_33_insideApp3_: bool
                                        d_34_currentApp3_: _dafny.Seq
                                        out22_: _dafny.Seq
                                        out23_: bool
                                        out24_: _dafny.Seq
                                        out22_, out23_, out24_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_31_nextBiased_)
                                        d_32_generatedApp3_ = out22_
                                        d_33_insideApp3_ = out23_
                                        d_34_currentApp3_ = out24_
                                        generated = d_32_generatedApp3_
                                        insideConstrainedOut = d_33_insideApp3_
                                        currentConstrainedOut = d_34_currentApp3_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

