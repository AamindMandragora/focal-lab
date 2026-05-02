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
        d_1_fromKeyword_: _dafny.Seq
        d_1_fromKeyword_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "FROM"))
        d_2_narrowThreshold_: int
        d_2_narrowThreshold_ = 12
        d_3_steps_: int
        d_3_steps_ = 0
        with _dafny.label("0"):
            while (d_3_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_4_chunkBudget_: int
                        d_4_chunkBudget_ = (maxSteps) - (d_3_steps_)
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
                        d_3_steps_ = (d_3_steps_) + (d_8_stepsUsed_)
                        if d_7_stoppedEos_:
                            raise _dafny.Break("0")
                        elif d_6_stoppedOpen_:
                            insideConstrainedOut = True
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        d_9_isComplete_: bool
                        d_9_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_9_isComplete_:
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
                            d_3_steps_ = (d_3_steps_) + (1)
                        elif True:
                            d_13_constrainedPrompt_: _dafny.Seq
                            d_13_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_14_validCount_: int
                            out7_: int
                            out7_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                            d_14_validCount_ = out7_
                            d_15_remaining_: int
                            d_15_remaining_ = (maxSteps) - (d_3_steps_)
                            if (((d_14_validCount_) > (d_2_narrowThreshold_)) and ((stepTokenBudget) > (0))) and ((d_15_remaining_) > (0)):
                                d_16_stablePrefix_: _dafny.Seq
                                d_16_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                d_17_symbolBudget_: int
                                d_17_symbolBudget_ = stepTokenBudget
                                if (d_15_remaining_) < (d_17_symbolBudget_):
                                    d_17_symbolBudget_ = d_15_remaining_
                                d_18_symbolOut_: _dafny.Seq
                                d_19_hitEos_: bool
                                d_20_stepsUsed2_: int
                                out8_: _dafny.Seq
                                out9_: bool
                                out10_: int
                                out8_, out9_, out10_ = (d_0_helpers_).ConstrainedSymbol(lm, parser, d_13_constrainedPrompt_, currentConstrainedOut, d_17_symbolBudget_, eosToken)
                                d_18_symbolOut_ = out8_
                                d_19_hitEos_ = out9_
                                d_20_stepsUsed2_ = out10_
                                generated = (d_16_stablePrefix_) + (d_18_symbolOut_)
                                currentConstrainedOut = d_18_symbolOut_
                                d_3_steps_ = (d_3_steps_) + (d_20_stepsUsed2_)
                                if d_19_hitEos_:
                                    raise _dafny.Break("0")
                            elif True:
                                d_21_semanticContext_: _dafny.Seq
                                out11_: _dafny.Seq
                                out11_ = VerifiedDecoderAgent.CSDHelpers.ExtractAfterKeyword(currentConstrainedOut, d_1_fromKeyword_)
                                d_21_semanticContext_ = out11_
                                (lm).GenerateLogits((d_13_constrainedPrompt_) + (currentConstrainedOut))
                                if (len(d_21_semanticContext_)) > (0):
                                    d_22_candidates1_: _dafny.Seq
                                    out12_: _dafny.Seq
                                    out12_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_13_constrainedPrompt_, currentConstrainedOut, 20, eosToken)
                                    d_22_candidates1_ = out12_
                                    d_23_focused_: _dafny.Seq
                                    out13_: _dafny.Seq
                                    out13_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_22_candidates1_, d_21_semanticContext_)
                                    d_23_focused_ = out13_
                                    if (len(d_23_focused_)) > (0):
                                        (d_0_helpers_).BoostTokenLogits(lm, d_23_focused_, _dafny.BigRational('6e0'))
                                if (len(validTokenGroups)) > (0):
                                    d_24_flatPreferred_: _dafny.Seq
                                    out14_: _dafny.Seq
                                    out14_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
                                    d_24_flatPreferred_ = out14_
                                    if (len(d_24_flatPreferred_)) > (0):
                                        d_25_anyPreferredValid_: bool
                                        out15_: bool
                                        out15_ = VerifiedDecoderAgent.CSDHelpers.GroupHasValidMember(parser, currentConstrainedOut, d_24_flatPreferred_)
                                        d_25_anyPreferredValid_ = out15_
                                        if d_25_anyPreferredValid_:
                                            d_26_candidates2_: _dafny.Seq
                                            out16_: _dafny.Seq
                                            out16_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_13_constrainedPrompt_, currentConstrainedOut, 30, eosToken)
                                            d_26_candidates2_ = out16_
                                            d_27_preferred_: _dafny.Seq
                                            out17_: _dafny.Seq
                                            out17_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_26_candidates2_, d_24_flatPreferred_)
                                            d_27_preferred_ = out17_
                                            if (len(d_27_preferred_)) > (0):
                                                (d_0_helpers_).BoostTokenLogits(lm, d_27_preferred_, _dafny.BigRational('4e0'))
                                (lm).MaskValidNextAndEos(parser, currentConstrainedOut, eosToken)
                                d_28_next_: _dafny.Seq
                                out18_: _dafny.Seq
                                out18_ = (lm).ChooseNextToken()
                                d_28_next_ = out18_
                                (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                                d_3_steps_ = (d_3_steps_) + (1)
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
        cost = d_3_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

