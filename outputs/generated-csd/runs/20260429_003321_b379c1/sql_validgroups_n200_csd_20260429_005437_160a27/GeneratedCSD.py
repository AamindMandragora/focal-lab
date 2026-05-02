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
                        d_4_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_4_next_ = out0_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_4_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_4_next_]))
                            if (d_4_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                d_3_activeGroup_ = -1
                    elif True:
                        d_5_completeNow_: bool
                        d_5_completeNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_5_completeNow_:
                            d_6_closedGenerated_: _dafny.Seq
                            d_7_closedInside_: bool
                            d_8_closedCurrent_: _dafny.Seq
                            out1_: _dafny.Seq
                            out2_: bool
                            out3_: _dafny.Seq
                            out1_, out2_, out3_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_6_closedGenerated_ = out1_
                            d_7_closedInside_ = out2_
                            d_8_closedCurrent_ = out3_
                            generated = d_6_closedGenerated_
                            insideConstrainedOut = d_7_closedInside_
                            currentConstrainedOut = d_8_closedCurrent_
                            d_3_activeGroup_ = -1
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_9_stablePrefix_: _dafny.Seq
                            d_9_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_10_constrainedPrompt_: _dafny.Seq
                            d_10_constrainedPrompt_ = (prompt) + (d_9_stablePrefix_)
                            d_11_validCount_: int
                            out4_: int
                            out4_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                            d_11_validCount_ = out4_
                            if (d_11_validCount_) <= (d_2_narrowThreshold_):
                                d_12_next_: _dafny.Seq
                                out5_: _dafny.Seq
                                out5_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_10_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_12_next_ = out5_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_12_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_13_appendedGenerated_: _dafny.Seq
                                    d_14_appendedInside_: bool
                                    d_15_appendedCurrent_: _dafny.Seq
                                    out6_: _dafny.Seq
                                    out7_: bool
                                    out8_: _dafny.Seq
                                    out6_, out7_, out8_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_12_next_)
                                    d_13_appendedGenerated_ = out6_
                                    d_14_appendedInside_ = out7_
                                    d_15_appendedCurrent_ = out8_
                                    generated = d_13_appendedGenerated_
                                    insideConstrainedOut = d_14_appendedInside_
                                    currentConstrainedOut = d_15_appendedCurrent_
                                    out9_: int
                                    out9_ = VerifiedDecoderAgent.CSDHelpers.GroupContaining(validTokenGroups, d_12_next_)
                                    d_3_activeGroup_ = out9_
                            elif True:
                                (lm).GenerateLogits((d_10_constrainedPrompt_) + (currentConstrainedOut))
                                if (len(validTokenGroups)) > (0):
                                    d_16_flatPreferred_: _dafny.Seq
                                    out10_: _dafny.Seq
                                    out10_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
                                    d_16_flatPreferred_ = out10_
                                    if (len(d_16_flatPreferred_)) > (0):
                                        d_17_anyValidPreferred_: bool
                                        out11_: bool
                                        out11_ = (d_0_helpers_).GroupHasValidMember(parser, currentConstrainedOut, d_16_flatPreferred_)
                                        d_17_anyValidPreferred_ = out11_
                                        if d_17_anyValidPreferred_:
                                            d_18_candidates_: _dafny.Seq
                                            out12_: _dafny.Seq
                                            out12_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_10_constrainedPrompt_, currentConstrainedOut, 24, eosToken)
                                            d_18_candidates_ = out12_
                                            d_19_preferred_: _dafny.Seq
                                            out13_: _dafny.Seq
                                            out13_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_18_candidates_, d_16_flatPreferred_)
                                            d_19_preferred_ = out13_
                                            if (len(d_19_preferred_)) > (0):
                                                (d_0_helpers_).BoostTokenLogits(lm, d_19_preferred_, _dafny.BigRational('4e0'))
                                            if (0) <= (d_3_activeGroup_):
                                                d_20_focused_: _dafny.Seq
                                                out14_: _dafny.Seq
                                                out14_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_18_candidates_, (validTokenGroups)[d_3_activeGroup_])
                                                d_20_focused_ = out14_
                                                if (len(d_20_focused_)) > (0):
                                                    (d_0_helpers_).BoostTokenLogits(lm, d_20_focused_, _dafny.BigRational('7e0'))
                                (lm).MaskValidNextAndEos(parser, currentConstrainedOut, eosToken)
                                d_21_next_: _dafny.Seq
                                out15_: _dafny.Seq
                                out15_ = (lm).ChooseNextToken()
                                d_21_next_ = out15_
                                (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_21_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_22_appendedGenerated_: _dafny.Seq
                                    d_23_appendedInside_: bool
                                    d_24_appendedCurrent_: _dafny.Seq
                                    out16_: _dafny.Seq
                                    out17_: bool
                                    out18_: _dafny.Seq
                                    out16_, out17_, out18_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_21_next_)
                                    d_22_appendedGenerated_ = out16_
                                    d_23_appendedInside_ = out17_
                                    d_24_appendedCurrent_ = out18_
                                    generated = d_22_appendedGenerated_
                                    insideConstrainedOut = d_23_appendedInside_
                                    currentConstrainedOut = d_24_appendedCurrent_
                                    out19_: int
                                    out19_ = VerifiedDecoderAgent.CSDHelpers.GroupContaining(validTokenGroups, d_21_next_)
                                    d_3_activeGroup_ = out19_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

