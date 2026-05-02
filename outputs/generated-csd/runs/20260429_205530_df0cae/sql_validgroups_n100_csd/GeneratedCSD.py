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
        d_3_activeGroup_: int
        d_3_activeGroup_ = -1
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
                            d_3_activeGroup_ = -1
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
                            d_3_activeGroup_ = -1
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_13_stablePrefix_: _dafny.Seq
                            d_13_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_14_constrainedPrompt_: _dafny.Seq
                            d_14_constrainedPrompt_ = (prompt) + (d_13_stablePrefix_)
                            d_15_validCount_: int
                            out7_: int
                            out7_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                            d_15_validCount_ = out7_
                            if (d_15_validCount_) <= (d_2_narrowThreshold_):
                                d_16_next_: _dafny.Seq
                                out8_: _dafny.Seq
                                out8_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_14_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_16_next_ = out8_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_16_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_17_appendedGenerated_: _dafny.Seq
                                    d_18_appendedInside_: bool
                                    d_19_appendedCurrent_: _dafny.Seq
                                    out9_: _dafny.Seq
                                    out10_: bool
                                    out11_: _dafny.Seq
                                    out9_, out10_, out11_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_16_next_)
                                    d_17_appendedGenerated_ = out9_
                                    d_18_appendedInside_ = out10_
                                    d_19_appendedCurrent_ = out11_
                                    generated = d_17_appendedGenerated_
                                    insideConstrainedOut = d_18_appendedInside_
                                    currentConstrainedOut = d_19_appendedCurrent_
                                    out12_: int
                                    out12_ = VerifiedDecoderAgent.CSDHelpers.GroupContaining(validTokenGroups, d_16_next_)
                                    d_3_activeGroup_ = out12_
                            elif True:
                                (lm).GenerateLogits((d_14_constrainedPrompt_) + (currentConstrainedOut))
                                if (len(validTokenGroups)) > (0):
                                    d_20_topCandidates_: _dafny.Seq
                                    out13_: _dafny.Seq
                                    out13_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_14_constrainedPrompt_, currentConstrainedOut, 24, eosToken)
                                    d_20_topCandidates_ = out13_
                                    if ((d_3_activeGroup_) >= (0)) and ((d_3_activeGroup_) < (len(validTokenGroups))):
                                        d_21_focused_: _dafny.Seq
                                        out14_: _dafny.Seq
                                        out14_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_20_topCandidates_, (validTokenGroups)[d_3_activeGroup_])
                                        d_21_focused_ = out14_
                                        if (len(d_21_focused_)) > (0):
                                            (d_0_helpers_).BoostTokenLogits(lm, d_21_focused_, _dafny.BigRational('8e0'))
                                    elif True:
                                        d_22_flatPreferred_: _dafny.Seq
                                        out15_: _dafny.Seq
                                        out15_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
                                        d_22_flatPreferred_ = out15_
                                        d_23_preferred_: _dafny.Seq
                                        out16_: _dafny.Seq
                                        out16_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_20_topCandidates_, d_22_flatPreferred_)
                                        d_23_preferred_ = out16_
                                        if (len(d_23_preferred_)) > (0):
                                            (d_0_helpers_).BoostTokenLogits(lm, d_23_preferred_, _dafny.BigRational('4e0'))
                                (lm).MaskValidNextAndEos(parser, currentConstrainedOut, eosToken)
                                d_24_next_: _dafny.Seq
                                out17_: _dafny.Seq
                                out17_ = (lm).ChooseNextToken()
                                d_24_next_ = out17_
                                (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_24_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_25_appendedGenerated_: _dafny.Seq
                                    d_26_appendedInside_: bool
                                    d_27_appendedCurrent_: _dafny.Seq
                                    out18_: _dafny.Seq
                                    out19_: bool
                                    out20_: _dafny.Seq
                                    out18_, out19_, out20_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_24_next_)
                                    d_25_appendedGenerated_ = out18_
                                    d_26_appendedInside_ = out19_
                                    d_27_appendedCurrent_ = out20_
                                    generated = d_25_appendedGenerated_
                                    insideConstrainedOut = d_26_appendedInside_
                                    currentConstrainedOut = d_27_appendedCurrent_
                                    out21_: int
                                    out21_ = VerifiedDecoderAgent.CSDHelpers.GroupContaining(validTokenGroups, d_24_next_)
                                    d_3_activeGroup_ = out21_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

