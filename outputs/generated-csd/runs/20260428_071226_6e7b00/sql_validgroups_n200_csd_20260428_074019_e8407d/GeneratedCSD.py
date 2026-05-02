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
        d_4_fromContext_: _dafny.Seq
        d_4_fromContext_ = _dafny.SeqWithoutIsStrInference([])
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
                        elif d_7_stoppedOnOpenSpan_:
                            insideConstrainedOut = True
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                            d_4_fromContext_ = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        out4_: _dafny.Seq
                        out4_ = VerifiedDecoderAgent.CSDHelpers.ExtractAfterKeyword(currentConstrainedOut, d_3_fromKeyword_)
                        d_4_fromContext_ = out4_
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            d_10_closedGenerated_: _dafny.Seq
                            d_11_closedInside_: bool
                            d_12_closedCurrent_: _dafny.Seq
                            out5_: _dafny.Seq
                            out6_: bool
                            out7_: _dafny.Seq
                            out5_, out6_, out7_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_10_closedGenerated_ = out5_
                            d_11_closedInside_ = out6_
                            d_12_closedCurrent_ = out7_
                            generated = d_10_closedGenerated_
                            insideConstrainedOut = d_11_closedInside_
                            currentConstrainedOut = d_12_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_4_fromContext_ = _dafny.SeqWithoutIsStrInference([])
                        elif True:
                            d_13_validCount_: int
                            out8_: int
                            out8_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                            d_13_validCount_ = out8_
                            d_14_stablePrefix_: _dafny.Seq
                            d_14_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_15_constrainedPrompt_: _dafny.Seq
                            d_15_constrainedPrompt_ = (prompt) + (d_14_stablePrefix_)
                            if (d_13_validCount_) > (d_2_narrowThreshold_):
                                d_16_budget_: int
                                d_16_budget_ = stepTokenBudget
                                if (d_16_budget_) > ((maxSteps) - (d_1_steps_)):
                                    d_16_budget_ = (maxSteps) - (d_1_steps_)
                                if (d_16_budget_) == (0):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_17_symbolOut_: _dafny.Seq
                                    d_18_hitEos_: bool
                                    d_19_stepsUsed_: int
                                    out9_: _dafny.Seq
                                    out10_: bool
                                    out11_: int
                                    out9_, out10_, out11_ = (d_0_helpers_).ConstrainedSymbol(lm, parser, d_15_constrainedPrompt_, currentConstrainedOut, d_16_budget_, eosToken)
                                    d_17_symbolOut_ = out9_
                                    d_18_hitEos_ = out10_
                                    d_19_stepsUsed_ = out11_
                                    generated = (d_14_stablePrefix_) + (d_17_symbolOut_)
                                    currentConstrainedOut = d_17_symbolOut_
                                    d_1_steps_ = (d_1_steps_) + (d_19_stepsUsed_)
                                    out12_: _dafny.Seq
                                    out12_ = VerifiedDecoderAgent.CSDHelpers.ExtractAfterKeyword(currentConstrainedOut, d_3_fromKeyword_)
                                    d_4_fromContext_ = out12_
                                    if d_18_hitEos_:
                                        raise _dafny.Break("0")
                            elif True:
                                (lm).GenerateLogits((d_15_constrainedPrompt_) + (currentConstrainedOut))
                                if (len(validTokenGroups)) > (0):
                                    d_20_flatPreferred_: _dafny.Seq
                                    out13_: _dafny.Seq
                                    out13_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
                                    d_20_flatPreferred_ = out13_
                                    d_21_candidates_: _dafny.Seq
                                    out14_: _dafny.Seq
                                    out14_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_15_constrainedPrompt_, currentConstrainedOut, 20, eosToken)
                                    d_21_candidates_ = out14_
                                    d_22_preferred_: _dafny.Seq
                                    out15_: _dafny.Seq
                                    out15_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_21_candidates_, d_20_flatPreferred_)
                                    d_22_preferred_ = out15_
                                    if (len(d_22_preferred_)) > (0):
                                        (d_0_helpers_).BoostTokenLogits(lm, d_22_preferred_, _dafny.BigRational('5e0'))
                                if (len(d_4_fromContext_)) > (0):
                                    d_23_contextCandidates_: _dafny.Seq
                                    out16_: _dafny.Seq
                                    out16_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_15_constrainedPrompt_, currentConstrainedOut, 20, eosToken)
                                    d_23_contextCandidates_ = out16_
                                    d_24_focused_: _dafny.Seq
                                    out17_: _dafny.Seq
                                    out17_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_23_contextCandidates_, d_4_fromContext_)
                                    d_24_focused_ = out17_
                                    if (len(d_24_focused_)) > (0):
                                        (d_0_helpers_).BoostTokenLogits(lm, d_24_focused_, _dafny.BigRational('4e0'))
                                (lm).MaskValidNextAndEos(parser, currentConstrainedOut, eosToken)
                                d_25_next_: _dafny.Seq
                                out18_: _dafny.Seq
                                out18_ = (lm).ChooseNextToken()
                                d_25_next_ = out18_
                                (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_25_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_26_appendedGenerated_: _dafny.Seq
                                    d_27_appendedInside_: bool
                                    d_28_appendedCurrent_: _dafny.Seq
                                    out19_: _dafny.Seq
                                    out20_: bool
                                    out21_: _dafny.Seq
                                    out19_, out20_, out21_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_25_next_)
                                    d_26_appendedGenerated_ = out19_
                                    d_27_appendedInside_ = out20_
                                    d_28_appendedCurrent_ = out21_
                                    generated = d_26_appendedGenerated_
                                    insideConstrainedOut = d_27_appendedInside_
                                    currentConstrainedOut = d_28_appendedCurrent_
                                    out22_: _dafny.Seq
                                    out22_ = VerifiedDecoderAgent.CSDHelpers.ExtractAfterKeyword(currentConstrainedOut, d_3_fromKeyword_)
                                    d_4_fromContext_ = out22_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

