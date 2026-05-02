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
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_fromKeyword_: _dafny.Seq
        d_2_fromKeyword_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "FROM"))
        d_3_fromContext_: _dafny.Seq
        d_3_fromContext_ = _dafny.SeqWithoutIsStrInference([])
        d_4_activeGroup_: int
        d_4_activeGroup_ = -1
        d_5_narrowThreshold_: int
        d_5_narrowThreshold_ = 12
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
                        out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_6_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
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
                            d_3_fromContext_ = _dafny.SeqWithoutIsStrInference([])
                            d_4_activeGroup_ = -1
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
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
                        d_3_fromContext_ = _dafny.SeqWithoutIsStrInference([])
                        d_4_activeGroup_ = -1
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        out7_: _dafny.Seq
                        out7_ = VerifiedDecoderAgent.CSDHelpers.ExtractAfterKeyword(currentConstrainedOut, d_2_fromKeyword_)
                        d_3_fromContext_ = out7_
                        d_14_stablePrefix_: _dafny.Seq
                        d_14_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_15_constrainedPrompt_: _dafny.Seq
                        d_15_constrainedPrompt_ = (prompt) + (d_14_stablePrefix_)
                        d_16_validCount_: int
                        out8_: int
                        out8_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                        d_16_validCount_ = out8_
                        if (d_16_validCount_) > (d_5_narrowThreshold_):
                            d_17_next_: _dafny.Seq
                            out9_: _dafny.Seq
                            out9_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_15_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_5_narrowThreshold_, eosToken)
                            d_17_next_ = out9_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_17_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_18_appendedGenerated_: _dafny.Seq
                                d_19_appendedInside_: bool
                                d_20_appendedCurrent_: _dafny.Seq
                                out10_: _dafny.Seq
                                out11_: bool
                                out12_: _dafny.Seq
                                out10_, out11_, out12_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_17_next_)
                                d_18_appendedGenerated_ = out10_
                                d_19_appendedInside_ = out11_
                                d_20_appendedCurrent_ = out12_
                                generated = d_18_appendedGenerated_
                                insideConstrainedOut = d_19_appendedInside_
                                currentConstrainedOut = d_20_appendedCurrent_
                                out13_: int
                                out13_ = VerifiedDecoderAgent.CSDHelpers.GroupContaining(validTokenGroups, d_17_next_)
                                d_4_activeGroup_ = out13_
                        elif True:
                            (lm).GenerateLogits((d_15_constrainedPrompt_) + (currentConstrainedOut))
                            if ((0) <= (d_4_activeGroup_)) and ((d_4_activeGroup_) < (len(validTokenGroups))):
                                d_21_groupValid_: bool
                                out14_: bool
                                out14_ = (d_0_helpers_).GroupHasValidMember(parser, currentConstrainedOut, (validTokenGroups)[d_4_activeGroup_])
                                d_21_groupValid_ = out14_
                                if d_21_groupValid_:
                                    (d_0_helpers_).BoostTokenLogits(lm, (validTokenGroups)[d_4_activeGroup_], _dafny.BigRational('8e0'))
                            if (len(d_3_fromContext_)) > (0):
                                d_22_flatGroups_: _dafny.Seq
                                out15_: _dafny.Seq
                                out15_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
                                d_22_flatGroups_ = out15_
                                d_23_focused_: _dafny.Seq
                                out16_: _dafny.Seq
                                out16_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_3_fromContext_, d_22_flatGroups_)
                                d_23_focused_ = out16_
                                if (len(d_23_focused_)) > (0):
                                    d_24_anyFocusedValid_: bool
                                    out17_: bool
                                    out17_ = (d_0_helpers_).GroupHasValidMember(parser, currentConstrainedOut, d_23_focused_)
                                    d_24_anyFocusedValid_ = out17_
                                    if d_24_anyFocusedValid_:
                                        (d_0_helpers_).BoostTokenLogits(lm, d_23_focused_, _dafny.BigRational('6e0'))
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
                                out22_: int
                                out22_ = VerifiedDecoderAgent.CSDHelpers.GroupContaining(validTokenGroups, d_25_next_)
                                d_4_activeGroup_ = out22_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

