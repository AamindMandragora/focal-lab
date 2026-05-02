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
        d_2_flatGroups_: _dafny.Seq
        out0_: _dafny.Seq
        out0_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
        d_2_flatGroups_ = out0_
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_3_openedGenerated_: _dafny.Seq
                        d_4_openedInside_: bool
                        d_5_openedCurrent_: _dafny.Seq
                        out1_: _dafny.Seq
                        out2_: bool
                        out3_: _dafny.Seq
                        out1_, out2_, out3_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                        d_3_openedGenerated_ = out1_
                        d_4_openedInside_ = out2_
                        d_5_openedCurrent_ = out3_
                        generated = d_3_openedGenerated_
                        insideConstrainedOut = d_4_openedInside_
                        currentConstrainedOut = d_5_openedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_6_completeNow_: bool
                        d_6_completeNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_6_completeNow_:
                            d_7_closedGenerated_: _dafny.Seq
                            d_8_closedInside_: bool
                            d_9_closedCurrent_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_7_closedGenerated_ = out4_
                            d_8_closedInside_ = out5_
                            d_9_closedCurrent_ = out6_
                            generated = d_7_closedGenerated_
                            insideConstrainedOut = d_8_closedInside_
                            currentConstrainedOut = d_9_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                            raise _dafny.Break("0")
                        elif True:
                            d_10_deadEnd_: bool
                            out7_: bool
                            out7_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                            d_10_deadEnd_ = out7_
                            if d_10_deadEnd_:
                                d_11_repaired_: _dafny.Seq
                                out8_: _dafny.Seq
                                out8_ = VerifiedDecoderAgent.CSDHelpers.RollbackToValidPrefix(parser, currentConstrainedOut)
                                d_11_repaired_ = out8_
                                if (len(d_11_repaired_)) == (len(currentConstrainedOut)):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_12_stablePrefix_: _dafny.Seq
                                    d_12_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                    generated = (d_12_stablePrefix_) + (d_11_repaired_)
                                    currentConstrainedOut = d_11_repaired_
                            elif True:
                                d_13_stablePrefix2_: _dafny.Seq
                                d_13_stablePrefix2_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                d_14_constrainedPrompt_: _dafny.Seq
                                d_14_constrainedPrompt_ = (prompt) + (d_13_stablePrefix2_)
                                (lm).GenerateLogits((d_14_constrainedPrompt_) + (currentConstrainedOut))
                                d_15_candidates_: _dafny.Seq
                                out9_: _dafny.Seq
                                out9_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_14_constrainedPrompt_, currentConstrainedOut, 16, eosToken)
                                d_15_candidates_ = out9_
                                d_16_hinted_: _dafny.Seq
                                out10_: _dafny.Seq
                                out10_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_15_candidates_, d_2_flatGroups_)
                                d_16_hinted_ = out10_
                                if (len(d_16_hinted_)) > (0):
                                    (d_0_helpers_).BoostTokenLogits(lm, d_16_hinted_, _dafny.BigRational('1e1'))
                                    d_17_nonHinted_: _dafny.Seq
                                    out11_: _dafny.Seq
                                    out11_ = VerifiedDecoderAgent.CSDHelpers.SubtractTokenSets(d_15_candidates_, d_16_hinted_)
                                    d_17_nonHinted_ = out11_
                                    if (len(d_17_nonHinted_)) > (0):
                                        (d_0_helpers_).PenalizeTokenLogits(lm, d_17_nonHinted_, _dafny.BigRational('4e0'))
                                d_18_prevTok_: _dafny.Seq
                                d_19_foundPrev_: bool
                                out12_: _dafny.Seq
                                out13_: bool
                                out12_, out13_ = (d_0_helpers_).LastTokenBefore(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))
                                d_18_prevTok_ = out12_
                                d_19_foundPrev_ = out13_
                                if d_19_foundPrev_:
                                    d_20_prevIdx_: int
                                    out14_: int
                                    out14_ = VerifiedDecoderAgent.CSDHelpers.GroupContaining(validTokenGroups, d_18_prevTok_)
                                    d_20_prevIdx_ = out14_
                                    if (d_20_prevIdx_) >= (0):
                                        d_21_safeGroup_: _dafny.Seq
                                        out15_: _dafny.Seq
                                        out15_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets((validTokenGroups)[d_20_prevIdx_], (lm).Tokens)
                                        d_21_safeGroup_ = out15_
                                        if (len(d_21_safeGroup_)) > (0):
                                            (d_0_helpers_).BoostTokenLogits(lm, d_21_safeGroup_, _dafny.BigRational('3e0'))
                                d_22_next_: _dafny.Seq
                                out16_: _dafny.Seq
                                out16_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_14_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_22_next_ = out16_
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

