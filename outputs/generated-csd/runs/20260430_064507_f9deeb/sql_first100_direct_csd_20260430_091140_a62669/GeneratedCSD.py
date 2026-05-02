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
        d_2_boundaryToken_: _dafny.Seq
        d_2_boundaryToken_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ","))
        d_3_focusKeyword_: _dafny.Seq
        d_3_focusKeyword_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "FROM"))
        d_4_minValidCount_: int
        d_4_minValidCount_ = 1
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
                    elif True:
                        if (parser).IsCompletePrefix(currentConstrainedOut):
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
                            d_13_deadEnd_: bool
                            out7_: bool
                            out7_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, d_4_minValidCount_)
                            d_13_deadEnd_ = out7_
                            if d_13_deadEnd_:
                                d_14_rewound_: _dafny.Seq
                                out8_: _dafny.Seq
                                out8_ = VerifiedDecoderAgent.CSDHelpers.RollbackToBoundary(parser, currentConstrainedOut, d_2_boundaryToken_)
                                d_14_rewound_ = out8_
                                d_15_dropped_: int
                                d_15_dropped_ = (len(currentConstrainedOut)) - (len(d_14_rewound_))
                                generated = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (d_15_dropped_):])
                                currentConstrainedOut = d_14_rewound_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                d_16_stablePrefix_: _dafny.Seq
                                d_16_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                d_17_constrainedPrompt_: _dafny.Seq
                                d_17_constrainedPrompt_ = (prompt) + (d_16_stablePrefix_)
                                d_18_schemaFocus_: _dafny.Seq
                                out9_: _dafny.Seq
                                out9_ = VerifiedDecoderAgent.CSDHelpers.ExtractAfterKeyword(currentConstrainedOut, d_3_focusKeyword_)
                                d_18_schemaFocus_ = out9_
                                (lm).GenerateLogits((d_17_constrainedPrompt_) + (currentConstrainedOut))
                                if (len(d_18_schemaFocus_)) > (0):
                                    d_19_focusCandidates_: _dafny.Seq
                                    out10_: _dafny.Seq
                                    out10_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_17_constrainedPrompt_, currentConstrainedOut, 24, eosToken)
                                    d_19_focusCandidates_ = out10_
                                    d_20_focused_: _dafny.Seq
                                    out11_: _dafny.Seq
                                    out11_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_19_focusCandidates_, d_18_schemaFocus_)
                                    d_20_focused_ = out11_
                                    if (len(d_20_focused_)) > (0):
                                        (d_0_helpers_).BoostTokenLogits(lm, d_20_focused_, _dafny.BigRational('8e0'))
                                if (len(validTokenGroups)) > (0):
                                    d_21_flatPreferred_: _dafny.Seq
                                    out12_: _dafny.Seq
                                    out12_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
                                    d_21_flatPreferred_ = out12_
                                    if (len(d_21_flatPreferred_)) > (0):
                                        d_22_anyPreferredValid_: bool
                                        out13_: bool
                                        out13_ = (d_0_helpers_).GroupHasValidMember(parser, currentConstrainedOut, d_21_flatPreferred_)
                                        d_22_anyPreferredValid_ = out13_
                                        if d_22_anyPreferredValid_:
                                            d_23_groupCandidates_: _dafny.Seq
                                            out14_: _dafny.Seq
                                            out14_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_17_constrainedPrompt_, currentConstrainedOut, 24, eosToken)
                                            d_23_groupCandidates_ = out14_
                                            d_24_preferred_: _dafny.Seq
                                            out15_: _dafny.Seq
                                            out15_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_23_groupCandidates_, d_21_flatPreferred_)
                                            d_24_preferred_ = out15_
                                            if (len(d_24_preferred_)) > (0):
                                                (d_0_helpers_).BoostTokenLogits(lm, d_24_preferred_, _dafny.BigRational('5e0'))
                                (lm).MaskValidNextAndEos(parser, currentConstrainedOut, eosToken)
                                d_25_next_: _dafny.Seq
                                out16_: _dafny.Seq
                                out16_ = (lm).ChooseNextToken()
                                d_25_next_ = out16_
                                (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_25_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_26_appendedGenerated_: _dafny.Seq
                                    d_27_appendedInside_: bool
                                    d_28_appendedCurrent_: _dafny.Seq
                                    out17_: _dafny.Seq
                                    out18_: bool
                                    out19_: _dafny.Seq
                                    out17_, out18_, out19_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_25_next_)
                                    d_26_appendedGenerated_ = out17_
                                    d_27_appendedInside_ = out18_
                                    d_28_appendedCurrent_ = out19_
                                    generated = d_26_appendedGenerated_
                                    insideConstrainedOut = d_27_appendedInside_
                                    currentConstrainedOut = d_28_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

