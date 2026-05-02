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
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_4_chunkBudget_: int
                        d_4_chunkBudget_ = (maxSteps) - (d_1_steps_)
                        d_5_chunkedGenerated_: _dafny.Seq
                        d_6_stoppedOpen_: bool
                        d_7_stoppedEos_: bool
                        d_8_stepsUsed_: int
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: bool
                        out3_: int
                        out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_4_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_5_chunkedGenerated_ = out0_
                        d_6_stoppedOpen_ = out1_
                        d_7_stoppedEos_ = out2_
                        d_8_stepsUsed_ = out3_
                        generated = d_5_chunkedGenerated_
                        d_1_steps_ = (d_1_steps_) + (d_8_stepsUsed_)
                        if d_7_stoppedEos_:
                            raise _dafny.Break("0")
                        elif d_6_stoppedOpen_:
                            insideConstrainedOut = True
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        d_9_complete_: bool
                        d_9_complete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_9_complete_:
                            d_10_closedGenerated_: _dafny.Seq
                            d_11_closedInside_: bool
                            d_12_closedCurrent_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_10_closedGenerated_ = out4_
                            d_11_closedInside_ = out5_
                            d_12_closedCurrent_ = out6_
                            generated = d_10_closedGenerated_
                            insideConstrainedOut = d_11_closedInside_
                            currentConstrainedOut = d_12_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_13_constrainedPrompt_: _dafny.Seq
                            d_13_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_14_validCount_: int
                            out7_: int
                            out7_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                            d_14_validCount_ = out7_
                            if ((d_14_validCount_) <= (d_2_narrowThreshold_)) or ((stepTokenBudget) == (0)):
                                (lm).GenerateLogits((d_13_constrainedPrompt_) + (currentConstrainedOut))
                                if (len(validTokenGroups)) > (0):
                                    d_15_flatPreferred_: _dafny.Seq
                                    out8_: _dafny.Seq
                                    out8_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
                                    d_15_flatPreferred_ = out8_
                                    d_16_topCandidates_: _dafny.Seq
                                    out9_: _dafny.Seq
                                    out9_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_13_constrainedPrompt_, currentConstrainedOut, 24, eosToken)
                                    d_16_topCandidates_ = out9_
                                    d_17_preferred_: _dafny.Seq
                                    out10_: _dafny.Seq
                                    out10_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_16_topCandidates_, d_15_flatPreferred_)
                                    d_17_preferred_ = out10_
                                    if (len(d_17_preferred_)) > (0):
                                        (d_0_helpers_).BoostTokenLogits(lm, d_17_preferred_, _dafny.BigRational('5e0'))
                                d_18_semanticContext_: _dafny.Seq
                                out11_: _dafny.Seq
                                out11_ = VerifiedDecoderAgent.CSDHelpers.ExtractAfterKeyword(currentConstrainedOut, d_3_scopeKeyword_)
                                d_18_semanticContext_ = out11_
                                if (len(d_18_semanticContext_)) > (0):
                                    d_19_topCandidates2_: _dafny.Seq
                                    out12_: _dafny.Seq
                                    out12_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_13_constrainedPrompt_, currentConstrainedOut, 24, eosToken)
                                    d_19_topCandidates2_ = out12_
                                    d_20_focused_: _dafny.Seq
                                    out13_: _dafny.Seq
                                    out13_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_19_topCandidates2_, d_18_semanticContext_)
                                    d_20_focused_ = out13_
                                    if (len(d_20_focused_)) > (0):
                                        (d_0_helpers_).BoostTokenLogits(lm, d_20_focused_, _dafny.BigRational('4e0'))
                                (lm).MaskValidNextAndEos(parser, currentConstrainedOut, eosToken)
                                d_21_next_: _dafny.Seq
                                out14_: _dafny.Seq
                                out14_ = (lm).ChooseNextToken()
                                d_21_next_ = out14_
                                (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_21_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_22_appendedGenerated_: _dafny.Seq
                                    d_23_appendedInside_: bool
                                    d_24_appendedCurrent_: _dafny.Seq
                                    out15_: _dafny.Seq
                                    out16_: bool
                                    out17_: _dafny.Seq
                                    out15_, out16_, out17_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_21_next_)
                                    d_22_appendedGenerated_ = out15_
                                    d_23_appendedInside_ = out16_
                                    d_24_appendedCurrent_ = out17_
                                    generated = d_22_appendedGenerated_
                                    insideConstrainedOut = d_23_appendedInside_
                                    currentConstrainedOut = d_24_appendedCurrent_
                            elif True:
                                d_25_stablePrefix_: _dafny.Seq
                                d_25_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                d_26_symbolBudget_: int
                                d_26_symbolBudget_ = (maxSteps) - (d_1_steps_)
                                if (stepTokenBudget) < (d_26_symbolBudget_):
                                    d_26_symbolBudget_ = stepTokenBudget
                                d_27_symbolOut_: _dafny.Seq
                                d_28_hitEos_: bool
                                d_29_stepsUsed2_: int
                                out18_: _dafny.Seq
                                out19_: bool
                                out20_: int
                                out18_, out19_, out20_ = (d_0_helpers_).ConstrainedSymbol(lm, parser, d_13_constrainedPrompt_, currentConstrainedOut, d_26_symbolBudget_, eosToken)
                                d_27_symbolOut_ = out18_
                                d_28_hitEos_ = out19_
                                d_29_stepsUsed2_ = out20_
                                generated = (d_25_stablePrefix_) + (d_27_symbolOut_)
                                currentConstrainedOut = d_27_symbolOut_
                                d_1_steps_ = (d_1_steps_) + (d_29_stepsUsed2_)
                                if d_28_hitEos_:
                                    raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

