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
        d_3_sawSpan_: bool
        d_3_sawSpan_ = insideConstrained
        d_4_narrowThreshold_: int
        d_4_narrowThreshold_ = 8
        d_5_mediumThreshold_: int
        d_5_mediumThreshold_ = 24
        d_6_flatGroups_: _dafny.Seq
        out0_: _dafny.Seq
        out0_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
        d_6_flatGroups_ = out0_
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_7_chunkBudget_: int
                        d_7_chunkBudget_ = (maxSteps) - (d_1_steps_)
                        d_8_generatedChunk_: _dafny.Seq
                        d_9_stoppedOnOpenSpan_: bool
                        d_10_stoppedOnEos_: bool
                        d_11_stepsUsed_: int
                        out1_: _dafny.Seq
                        out2_: bool
                        out3_: bool
                        out4_: int
                        out1_, out2_, out3_, out4_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_7_chunkBudget_, d_2_openSpanToken_, eosToken)
                        d_8_generatedChunk_ = out1_
                        d_9_stoppedOnOpenSpan_ = out2_
                        d_10_stoppedOnEos_ = out3_
                        d_11_stepsUsed_ = out4_
                        generated = d_8_generatedChunk_
                        d_1_steps_ = (d_1_steps_) + (d_11_stepsUsed_)
                        if d_10_stoppedOnEos_:
                            raise _dafny.Break("0")
                        elif True:
                            if d_9_stoppedOnOpenSpan_:
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                d_3_sawSpan_ = True
                    elif True:
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            raise _dafny.Break("0")
                        elif True:
                            d_12_stablePrefix_: _dafny.Seq
                            d_12_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_13_constrainedPrompt_: _dafny.Seq
                            d_13_constrainedPrompt_ = (prompt) + (d_12_stablePrefix_)
                            d_14_validCount_: int
                            out5_: int
                            out5_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                            d_14_validCount_ = out5_
                            if (d_14_validCount_) <= (d_4_narrowThreshold_):
                                d_15_nextNarrow_: _dafny.Seq
                                out6_: _dafny.Seq
                                out6_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_13_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_15_nextNarrow_ = out6_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_15_nextNarrow_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_16_generatedApp1_: _dafny.Seq
                                    d_17_insideApp1_: bool
                                    d_18_currentApp1_: _dafny.Seq
                                    out7_: _dafny.Seq
                                    out8_: bool
                                    out9_: _dafny.Seq
                                    out7_, out8_, out9_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_15_nextNarrow_)
                                    d_16_generatedApp1_ = out7_
                                    d_17_insideApp1_ = out8_
                                    d_18_currentApp1_ = out9_
                                    generated = d_16_generatedApp1_
                                    insideConstrainedOut = d_17_insideApp1_
                                    currentConstrainedOut = d_18_currentApp1_
                            elif True:
                                d_19_candidates_: _dafny.Seq
                                out10_: _dafny.Seq
                                out10_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_13_constrainedPrompt_, currentConstrainedOut, 12, eosToken)
                                d_19_candidates_ = out10_
                                if ((d_14_validCount_) <= (d_5_mediumThreshold_)) or ((len(d_19_candidates_)) == (0)):
                                    d_20_nextMedium_: _dafny.Seq
                                    out11_: _dafny.Seq
                                    out11_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_13_constrainedPrompt_, currentConstrainedOut, eosToken)
                                    d_20_nextMedium_ = out11_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    if (d_20_nextMedium_) == (eosToken):
                                        raise _dafny.Break("0")
                                    elif True:
                                        d_21_generatedApp2_: _dafny.Seq
                                        d_22_insideApp2_: bool
                                        d_23_currentApp2_: _dafny.Seq
                                        out12_: _dafny.Seq
                                        out13_: bool
                                        out14_: _dafny.Seq
                                        out12_, out13_, out14_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_20_nextMedium_)
                                        d_21_generatedApp2_ = out12_
                                        d_22_insideApp2_ = out13_
                                        d_23_currentApp2_ = out14_
                                        generated = d_21_generatedApp2_
                                        insideConstrainedOut = d_22_insideApp2_
                                        currentConstrainedOut = d_23_currentApp2_
                                elif True:
                                    (lm).GenerateLogits((d_13_constrainedPrompt_) + (currentConstrainedOut))
                                    d_24_i_: int
                                    d_24_i_ = 0
                                    while (d_24_i_) < (len(validTokenGroups)):
                                        d_25_group_: _dafny.Seq
                                        d_25_group_ = (validTokenGroups)[d_24_i_]
                                        d_26_anyValid_: bool
                                        out15_: bool
                                        out15_ = (d_0_helpers_).GroupHasValidMember(parser, currentConstrainedOut, d_25_group_)
                                        d_26_anyValid_ = out15_
                                        if d_26_anyValid_:
                                            d_27_overlap_: _dafny.Seq
                                            out16_: _dafny.Seq
                                            out16_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_19_candidates_, d_25_group_)
                                            d_27_overlap_ = out16_
                                            if (len(d_27_overlap_)) > (0):
                                                (d_0_helpers_).BoostTokenLogits(lm, d_27_overlap_, _dafny.BigRational('8e0'))
                                        d_24_i_ = (d_24_i_) + (1)
                                    if (len(d_6_flatGroups_)) > (0):
                                        d_28_outsideHints_: _dafny.Seq
                                        out17_: _dafny.Seq
                                        out17_ = VerifiedDecoderAgent.CSDHelpers.SubtractTokenSets(d_19_candidates_, d_6_flatGroups_)
                                        d_28_outsideHints_ = out17_
                                        if (len(d_28_outsideHints_)) > (0):
                                            (d_0_helpers_).PenalizeTokenLogits(lm, d_28_outsideHints_, _dafny.BigRational('15e-1'))
                                    (lm).MaskValidNextAndEos(parser, currentConstrainedOut, eosToken)
                                    d_29_nextBiased_: _dafny.Seq
                                    out18_: _dafny.Seq
                                    out18_ = (lm).ChooseNextToken()
                                    d_29_nextBiased_ = out18_
                                    (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    if (d_29_nextBiased_) == (eosToken):
                                        raise _dafny.Break("0")
                                    elif True:
                                        d_30_generatedApp3_: _dafny.Seq
                                        d_31_insideApp3_: bool
                                        d_32_currentApp3_: _dafny.Seq
                                        out19_: _dafny.Seq
                                        out20_: bool
                                        out21_: _dafny.Seq
                                        out19_, out20_, out21_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_29_nextBiased_)
                                        d_30_generatedApp3_ = out19_
                                        d_31_insideApp3_ = out20_
                                        d_32_currentApp3_ = out21_
                                        generated = d_30_generatedApp3_
                                        insideConstrainedOut = d_31_insideApp3_
                                        currentConstrainedOut = d_32_currentApp3_
                    pass
            pass
        if d_3_sawSpan_:
            generated = currentConstrainedOut
            insideConstrainedOut = False
            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

