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
        d_2_flatPreferred_: _dafny.Seq
        out0_: _dafny.Seq
        out0_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
        d_2_flatPreferred_ = out0_
        d_3_repeatThreshold_: int
        d_3_repeatThreshold_ = 2
        d_4_lastConstrainedToken_: _dafny.Seq
        d_4_lastConstrainedToken_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
        d_5_repeatStreak_: int
        d_5_repeatStreak_ = 0
        d_6_haveLastConstrainedToken_: bool
        d_6_haveLastConstrainedToken_ = False
        d_7_narrowThreshold_: int
        d_7_narrowThreshold_ = 4
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_8_chunkBudget_: int
                        d_8_chunkBudget_ = (maxSteps) - (d_1_steps_)
                        d_9_chunkedGenerated_: _dafny.Seq
                        d_10_stoppedOnOpenSpan_: bool
                        d_11_stoppedOnEos_: bool
                        d_12_stepsUsed_: int
                        out1_: _dafny.Seq
                        out2_: bool
                        out3_: bool
                        out4_: int
                        out1_, out2_, out3_, out4_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_8_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_9_chunkedGenerated_ = out1_
                        d_10_stoppedOnOpenSpan_ = out2_
                        d_11_stoppedOnEos_ = out3_
                        d_12_stepsUsed_ = out4_
                        generated = d_9_chunkedGenerated_
                        d_1_steps_ = (d_1_steps_) + (d_12_stepsUsed_)
                        if d_11_stoppedOnEos_:
                            raise _dafny.Break("0")
                        elif True:
                            if d_10_stoppedOnOpenSpan_:
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                d_6_haveLastConstrainedToken_ = False
                                d_5_repeatStreak_ = 0
                                d_4_lastConstrainedToken_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                    elif True:
                        d_13_complete_: bool
                        d_13_complete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_13_complete_:
                            d_14_closedGenerated_: _dafny.Seq
                            d_15_closedInside_: bool
                            d_16_closedCurrent_: _dafny.Seq
                            out5_: _dafny.Seq
                            out6_: bool
                            out7_: _dafny.Seq
                            out5_, out6_, out7_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_14_closedGenerated_ = out5_
                            d_15_closedInside_ = out6_
                            d_16_closedCurrent_ = out7_
                            generated = d_14_closedGenerated_
                            insideConstrainedOut = d_15_closedInside_
                            currentConstrainedOut = d_16_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_6_haveLastConstrainedToken_ = False
                            d_5_repeatStreak_ = 0
                            d_4_lastConstrainedToken_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                        elif True:
                            d_17_validCount_: int
                            out8_: int
                            out8_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                            d_17_validCount_ = out8_
                            d_18_constrainedPrompt_: _dafny.Seq
                            d_18_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            (lm).GenerateLogits((d_18_constrainedPrompt_) + (currentConstrainedOut))
                            if (len(d_2_flatPreferred_)) > (0):
                                d_19_candidates_: _dafny.Seq
                                out9_: _dafny.Seq
                                out9_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_18_constrainedPrompt_, currentConstrainedOut, 32, eosToken)
                                d_19_candidates_ = out9_
                                d_20_preferredNow_: _dafny.Seq
                                out10_: _dafny.Seq
                                out10_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_19_candidates_, d_2_flatPreferred_)
                                d_20_preferredNow_ = out10_
                                if (len(d_20_preferredNow_)) > (0):
                                    (d_0_helpers_).BoostTokenLogits(lm, d_20_preferredNow_, _dafny.BigRational('6e0'))
                            if (d_17_validCount_) <= (d_7_narrowThreshold_):
                                d_21_topValid_: _dafny.Seq
                                out11_: _dafny.Seq
                                out11_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_18_constrainedPrompt_, currentConstrainedOut, 1, eosToken)
                                d_21_topValid_ = out11_
                                if (len(d_21_topValid_)) > (0):
                                    (d_0_helpers_).BoostTokenLogits(lm, d_21_topValid_, _dafny.BigRational('4e0'))
                            if ((d_6_haveLastConstrainedToken_) and ((d_5_repeatStreak_) >= (d_3_repeatThreshold_))) and ((d_17_validCount_) > (1)):
                                d_22_repeatedSeq_: _dafny.Seq
                                d_22_repeatedSeq_ = _dafny.SeqWithoutIsStrInference([d_4_lastConstrainedToken_])
                                (d_0_helpers_).PenalizeTokenLogits(lm, d_22_repeatedSeq_, _dafny.BigRational('8e0'))
                            (lm).MaskValidNextAndEos(parser, currentConstrainedOut, eosToken)
                            d_23_next_: _dafny.Seq
                            out12_: _dafny.Seq
                            out12_ = (lm).ChooseNextToken()
                            d_23_next_ = out12_
                            (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_23_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_24_appendedGenerated_: _dafny.Seq
                                d_25_appendedInside_: bool
                                d_26_appendedCurrent_: _dafny.Seq
                                out13_: _dafny.Seq
                                out14_: bool
                                out15_: _dafny.Seq
                                out13_, out14_, out15_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_23_next_)
                                d_24_appendedGenerated_ = out13_
                                d_25_appendedInside_ = out14_
                                d_26_appendedCurrent_ = out15_
                                generated = d_24_appendedGenerated_
                                insideConstrainedOut = d_25_appendedInside_
                                currentConstrainedOut = d_26_appendedCurrent_
                                if d_6_haveLastConstrainedToken_:
                                    if (d_23_next_) == (d_4_lastConstrainedToken_):
                                        d_5_repeatStreak_ = (d_5_repeatStreak_) + (1)
                                    elif True:
                                        d_4_lastConstrainedToken_ = d_23_next_
                                        d_5_repeatStreak_ = 1
                                elif True:
                                    d_6_haveLastConstrainedToken_ = True
                                    d_4_lastConstrainedToken_ = d_23_next_
                                    d_5_repeatStreak_ = 1
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

