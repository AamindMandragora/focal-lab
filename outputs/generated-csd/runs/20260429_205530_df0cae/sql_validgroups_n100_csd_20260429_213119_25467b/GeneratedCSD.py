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
        d_2_activeGroup_: int
        d_2_activeGroup_ = -1
        d_3_schemaKeywords_: _dafny.Seq
        d_3_schemaKeywords_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "FROM")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "JOIN")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "UPDATE")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "INTO"))])
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_4_chunkBudget_: int
                        d_4_chunkBudget_ = (maxSteps) - (d_1_steps_)
                        d_5_chunkedGenerated_: _dafny.Seq
                        d_6_stoppedOnOpenSpan_: bool
                        d_7_stoppedOnEos_: bool
                        d_8_stepsUsed_: int
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: bool
                        out3_: int
                        out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_4_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_5_chunkedGenerated_ = out0_
                        d_6_stoppedOnOpenSpan_ = out1_
                        d_7_stoppedOnEos_ = out2_
                        d_8_stepsUsed_ = out3_
                        generated = d_5_chunkedGenerated_
                        d_1_steps_ = (d_1_steps_) + (d_8_stepsUsed_)
                        if d_7_stoppedOnEos_:
                            raise _dafny.Break("0")
                        elif d_6_stoppedOnOpenSpan_:
                            insideConstrainedOut = True
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                            d_2_activeGroup_ = -1
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
                            d_2_activeGroup_ = -1
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_13_deadEnd_: bool
                            out7_: bool
                            out7_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                            d_13_deadEnd_ = out7_
                            if d_13_deadEnd_:
                                d_14_repaired_: _dafny.Seq
                                out8_: _dafny.Seq
                                out8_ = VerifiedDecoderAgent.CSDHelpers.RollbackToBoundary(parser, currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ",")))
                                d_14_repaired_ = out8_
                                d_15_dropped_: int
                                d_15_dropped_ = (len(currentConstrainedOut)) - (len(d_14_repaired_))
                                generated = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (d_15_dropped_):])
                                currentConstrainedOut = d_14_repaired_
                                if (len(d_14_repaired_)) == (0):
                                    d_2_activeGroup_ = -1
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                d_16_stablePrefix_: _dafny.Seq
                                d_16_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                d_17_constrainedPrompt_: _dafny.Seq
                                d_17_constrainedPrompt_ = (prompt) + (d_16_stablePrefix_)
                                (lm).GenerateLogits((d_17_constrainedPrompt_) + (currentConstrainedOut))
                                d_18_lastBeforeComma_: _dafny.Seq
                                d_19_foundBeforeComma_: bool
                                out9_: _dafny.Seq
                                out10_: bool
                                out9_, out10_ = (d_0_helpers_).LastTokenBefore(currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ",")))
                                d_18_lastBeforeComma_ = out9_
                                d_19_foundBeforeComma_ = out10_
                                d_20_focusSchema_: bool
                                d_20_focusSchema_ = False
                                if d_19_foundBeforeComma_:
                                    if (d_18_lastBeforeComma_) in (d_3_schemaKeywords_):
                                        d_20_focusSchema_ = True
                                if (len(validTokenGroups)) > (0):
                                    if ((d_2_activeGroup_) >= (0)) and ((d_2_activeGroup_) < (len(validTokenGroups))):
                                        d_21_activeHasValid_: bool
                                        out11_: bool
                                        out11_ = (d_0_helpers_).GroupHasValidMember(parser, currentConstrainedOut, (validTokenGroups)[d_2_activeGroup_])
                                        d_21_activeHasValid_ = out11_
                                        if d_21_activeHasValid_:
                                            d_22_activeCandidates_: _dafny.Seq
                                            out12_: _dafny.Seq
                                            out12_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_17_constrainedPrompt_, currentConstrainedOut, 25, eosToken)
                                            d_22_activeCandidates_ = out12_
                                            d_23_activePreferred_: _dafny.Seq
                                            out13_: _dafny.Seq
                                            out13_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_22_activeCandidates_, (validTokenGroups)[d_2_activeGroup_])
                                            d_23_activePreferred_ = out13_
                                            if (len(d_23_activePreferred_)) > (0):
                                                if d_20_focusSchema_:
                                                    (d_0_helpers_).BoostTokenLogits(lm, d_23_activePreferred_, _dafny.BigRational('8e0'))
                                                elif True:
                                                    (d_0_helpers_).BoostTokenLogits(lm, d_23_activePreferred_, _dafny.BigRational('4e0'))
                                    elif True:
                                        d_24_flatPreferred_: _dafny.Seq
                                        out14_: _dafny.Seq
                                        out14_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
                                        d_24_flatPreferred_ = out14_
                                        if (len(d_24_flatPreferred_)) > (0):
                                            d_25_anyValidPreferred_: bool
                                            out15_: bool
                                            out15_ = (d_0_helpers_).GroupHasValidMember(parser, currentConstrainedOut, d_24_flatPreferred_)
                                            d_25_anyValidPreferred_ = out15_
                                            if d_25_anyValidPreferred_:
                                                d_26_candidates_: _dafny.Seq
                                                out16_: _dafny.Seq
                                                out16_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_17_constrainedPrompt_, currentConstrainedOut, 25, eosToken)
                                                d_26_candidates_ = out16_
                                                d_27_preferred_: _dafny.Seq
                                                out17_: _dafny.Seq
                                                out17_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_26_candidates_, d_24_flatPreferred_)
                                                d_27_preferred_ = out17_
                                                if (len(d_27_preferred_)) > (0):
                                                    if d_20_focusSchema_:
                                                        (d_0_helpers_).BoostTokenLogits(lm, d_27_preferred_, _dafny.BigRational('6e0'))
                                                    elif True:
                                                        (d_0_helpers_).BoostTokenLogits(lm, d_27_preferred_, _dafny.BigRational('3e0'))
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
                                    if (len(validTokenGroups)) > (0):
                                        d_32_idx_: int
                                        out22_: int
                                        out22_ = VerifiedDecoderAgent.CSDHelpers.GroupContaining(validTokenGroups, d_28_next_)
                                        d_32_idx_ = out22_
                                        if (d_32_idx_) >= (0):
                                            d_2_activeGroup_ = d_32_idx_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

