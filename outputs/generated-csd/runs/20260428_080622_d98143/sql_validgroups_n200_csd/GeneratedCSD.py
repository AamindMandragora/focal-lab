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
        d_3_fromKeyword_: _dafny.Seq
        d_3_fromKeyword_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "FROM"))
        d_4_selectKeyword_: _dafny.Seq
        d_4_selectKeyword_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SELECT"))
        d_5_preferredFlat_: _dafny.Seq
        out0_: _dafny.Seq
        out0_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
        d_5_preferredFlat_ = out0_
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_6_chunkBudget_: int
                        d_6_chunkBudget_ = (maxSteps) - (d_1_steps_)
                        d_7_chunkedGenerated_: _dafny.Seq
                        d_8_stoppedOpen_: bool
                        d_9_stoppedEos_: bool
                        d_10_stepsUsed_: int
                        out1_: _dafny.Seq
                        out2_: bool
                        out3_: bool
                        out4_: int
                        out1_, out2_, out3_, out4_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_6_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_7_chunkedGenerated_ = out1_
                        d_8_stoppedOpen_ = out2_
                        d_9_stoppedEos_ = out3_
                        d_10_stepsUsed_ = out4_
                        generated = d_7_chunkedGenerated_
                        d_1_steps_ = (d_1_steps_) + (d_10_stepsUsed_)
                        if d_9_stoppedEos_:
                            raise _dafny.Break("0")
                        elif d_8_stoppedOpen_:
                            insideConstrainedOut = True
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        d_11_complete_: bool
                        d_11_complete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_11_complete_:
                            d_12_closedGenerated_: _dafny.Seq
                            d_13_closedInside_: bool
                            d_14_closedCurrent_: _dafny.Seq
                            out5_: _dafny.Seq
                            out6_: bool
                            out7_: _dafny.Seq
                            out5_, out6_, out7_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_12_closedGenerated_ = out5_
                            d_13_closedInside_ = out6_
                            d_14_closedCurrent_ = out7_
                            generated = d_12_closedGenerated_
                            insideConstrainedOut = d_13_closedInside_
                            currentConstrainedOut = d_14_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_15_stablePrefix_: _dafny.Seq
                            d_15_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_16_constrainedPrompt_: _dafny.Seq
                            d_16_constrainedPrompt_ = (prompt) + (d_15_stablePrefix_)
                            d_17_validCount_: int
                            out8_: int
                            out8_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                            d_17_validCount_ = out8_
                            if (d_17_validCount_) <= (d_2_narrowThreshold_):
                                d_18_next_: _dafny.Seq
                                out9_: _dafny.Seq
                                out9_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_16_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_18_next_ = out9_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_18_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_19_appendedGenerated_: _dafny.Seq
                                    d_20_appendedInside_: bool
                                    d_21_appendedCurrent_: _dafny.Seq
                                    out10_: _dafny.Seq
                                    out11_: bool
                                    out12_: _dafny.Seq
                                    out10_, out11_, out12_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_18_next_)
                                    d_19_appendedGenerated_ = out10_
                                    d_20_appendedInside_ = out11_
                                    d_21_appendedCurrent_ = out12_
                                    generated = d_19_appendedGenerated_
                                    insideConstrainedOut = d_20_appendedInside_
                                    currentConstrainedOut = d_21_appendedCurrent_
                            elif True:
                                d_22_fromContext_: _dafny.Seq
                                out13_: _dafny.Seq
                                out13_ = VerifiedDecoderAgent.CSDHelpers.ExtractAfterKeyword(currentConstrainedOut, d_3_fromKeyword_)
                                d_22_fromContext_ = out13_
                                d_23_selectContext_: _dafny.Seq
                                out14_: _dafny.Seq
                                out14_ = VerifiedDecoderAgent.CSDHelpers.ExtractAfterKeyword(currentConstrainedOut, d_4_selectKeyword_)
                                d_23_selectContext_ = out14_
                                d_24_semanticContext_: _dafny.Seq
                                d_24_semanticContext_ = (d_22_fromContext_) + (d_23_selectContext_)
                                (lm).GenerateLogits((d_16_constrainedPrompt_) + (currentConstrainedOut))
                                d_25_candidates_: _dafny.Seq
                                out15_: _dafny.Seq
                                out15_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_16_constrainedPrompt_, currentConstrainedOut, 24, eosToken)
                                d_25_candidates_ = out15_
                                if (len(d_5_preferredFlat_)) > (0):
                                    d_26_preferredCandidates_: _dafny.Seq
                                    out16_: _dafny.Seq
                                    out16_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_25_candidates_, d_5_preferredFlat_)
                                    d_26_preferredCandidates_ = out16_
                                    if (len(d_26_preferredCandidates_)) > (0):
                                        (d_0_helpers_).BoostTokenLogits(lm, d_26_preferredCandidates_, _dafny.BigRational('5e0'))
                                if (len(d_24_semanticContext_)) > (0):
                                    d_27_semanticCandidates_: _dafny.Seq
                                    out17_: _dafny.Seq
                                    out17_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_25_candidates_, d_24_semanticContext_)
                                    d_27_semanticCandidates_ = out17_
                                    if (len(d_27_semanticCandidates_)) > (0):
                                        (d_0_helpers_).BoostTokenLogits(lm, d_27_semanticCandidates_, _dafny.BigRational('6e0'))
                                (lm).MaskValidNextAndEos(parser, currentConstrainedOut, eosToken)
                                d_28_next_: _dafny.Seq
                                out18_: _dafny.Seq
                                out18_ = (lm).ChooseNextToken()
                                d_28_next_ = out18_
                                (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_28_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_29_appendedGenerated_: _dafny.Seq
                                    d_30_appendedInside_: bool
                                    d_31_appendedCurrent_: _dafny.Seq
                                    out19_: _dafny.Seq
                                    out20_: bool
                                    out21_: _dafny.Seq
                                    out19_, out20_, out21_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_28_next_)
                                    d_29_appendedGenerated_ = out19_
                                    d_30_appendedInside_ = out20_
                                    d_31_appendedCurrent_ = out21_
                                    generated = d_29_appendedGenerated_
                                    insideConstrainedOut = d_30_appendedInside_
                                    currentConstrainedOut = d_31_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

