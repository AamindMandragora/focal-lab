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
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_3_chunkBudget_: int
                        d_3_chunkBudget_ = (maxSteps) - (d_1_steps_)
                        d_4_chunkedGenerated_: _dafny.Seq
                        d_5_stoppedOnOpenSpan_: bool
                        d_6_stoppedOnEos_: bool
                        d_7_stepsUsed_: int
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: bool
                        out3_: int
                        out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_3_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_4_chunkedGenerated_ = out0_
                        d_5_stoppedOnOpenSpan_ = out1_
                        d_6_stoppedOnEos_ = out2_
                        d_7_stepsUsed_ = out3_
                        generated = d_4_chunkedGenerated_
                        d_1_steps_ = (d_1_steps_) + (d_7_stepsUsed_)
                        if d_6_stoppedOnEos_:
                            raise _dafny.Break("0")
                        elif d_5_stoppedOnOpenSpan_:
                            insideConstrainedOut = True
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                            d_8_prevTok_: _dafny.Seq
                            d_9_foundPrev_: bool
                            out4_: _dafny.Seq
                            out5_: bool
                            out4_, out5_ = (d_0_helpers_).LastTokenBefore(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))
                            d_8_prevTok_ = out4_
                            d_9_foundPrev_ = out5_
                            if d_9_foundPrev_:
                                out6_: int
                                out6_ = VerifiedDecoderAgent.CSDHelpers.GroupContaining(validTokenGroups, d_8_prevTok_)
                                d_2_activeGroup_ = out6_
                            elif True:
                                d_2_activeGroup_ = -1
                    elif True:
                        d_10_completeNow_: bool
                        d_10_completeNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_10_completeNow_:
                            d_11_closedGenerated_: _dafny.Seq
                            d_12_closedInside_: bool
                            d_13_closedCurrent_: _dafny.Seq
                            out7_: _dafny.Seq
                            out8_: bool
                            out9_: _dafny.Seq
                            out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_11_closedGenerated_ = out7_
                            d_12_closedInside_ = out8_
                            d_13_closedCurrent_ = out9_
                            generated = d_11_closedGenerated_
                            insideConstrainedOut = d_12_closedInside_
                            currentConstrainedOut = d_13_closedCurrent_
                            d_2_activeGroup_ = -1
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_14_stablePrefix_: _dafny.Seq
                            d_14_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_15_constrainedPrompt_: _dafny.Seq
                            d_15_constrainedPrompt_ = (prompt) + (d_14_stablePrefix_)
                            (lm).GenerateLogits((d_15_constrainedPrompt_) + (currentConstrainedOut))
                            d_16_candidates_: _dafny.Seq
                            out10_: _dafny.Seq
                            out10_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_15_constrainedPrompt_, currentConstrainedOut, 24, eosToken)
                            d_16_candidates_ = out10_
                            if ((0) <= (d_2_activeGroup_)) and ((d_2_activeGroup_) < (len(validTokenGroups))):
                                d_17_focused_: _dafny.Seq
                                out11_: _dafny.Seq
                                out11_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_16_candidates_, (validTokenGroups)[d_2_activeGroup_])
                                d_17_focused_ = out11_
                                if (len(d_17_focused_)) > (0):
                                    (d_0_helpers_).BoostTokenLogits(lm, d_17_focused_, _dafny.BigRational('8e0'))
                                elif True:
                                    d_18_flatPreferred_: _dafny.Seq
                                    out12_: _dafny.Seq
                                    out12_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
                                    d_18_flatPreferred_ = out12_
                                    d_19_fallbackPreferred_: _dafny.Seq
                                    out13_: _dafny.Seq
                                    out13_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_16_candidates_, d_18_flatPreferred_)
                                    d_19_fallbackPreferred_ = out13_
                                    if (len(d_19_fallbackPreferred_)) > (0):
                                        (d_0_helpers_).BoostTokenLogits(lm, d_19_fallbackPreferred_, _dafny.BigRational('3e0'))
                            elif True:
                                d_20_flatPreferred2_: _dafny.Seq
                                out14_: _dafny.Seq
                                out14_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
                                d_20_flatPreferred2_ = out14_
                                d_21_preferred2_: _dafny.Seq
                                out15_: _dafny.Seq
                                out15_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_16_candidates_, d_20_flatPreferred2_)
                                d_21_preferred2_ = out15_
                                if (len(d_21_preferred2_)) > (0):
                                    (d_0_helpers_).BoostTokenLogits(lm, d_21_preferred2_, _dafny.BigRational('3e0'))
                            (lm).MaskValidNextAndEos(parser, currentConstrainedOut, eosToken)
                            d_22_next_: _dafny.Seq
                            out16_: _dafny.Seq
                            out16_ = (lm).ChooseNextToken()
                            d_22_next_ = out16_
                            (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_22_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_23_appendedGenerated_: _dafny.Seq
                                d_24_appendedInside_: bool
                                d_25_appendedCurrent_: _dafny.Seq
                                out17_: _dafny.Seq
                                out18_: bool
                                out19_: _dafny.Seq
                                out17_, out18_, out19_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_22_next_)
                                d_23_appendedGenerated_ = out17_
                                d_24_appendedInside_ = out18_
                                d_25_appendedCurrent_ = out19_
                                generated = d_23_appendedGenerated_
                                insideConstrainedOut = d_24_appendedInside_
                                currentConstrainedOut = d_25_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

