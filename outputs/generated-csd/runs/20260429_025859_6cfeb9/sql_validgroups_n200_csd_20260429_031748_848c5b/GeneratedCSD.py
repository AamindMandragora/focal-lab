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
        d_2_preferredFlat_: _dafny.Seq
        out0_: _dafny.Seq
        out0_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
        d_2_preferredFlat_ = out0_
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_3_next_: _dafny.Seq
                        out1_: _dafny.Seq
                        out1_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_3_next_ = out1_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_3_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_3_next_]))
                            if (d_3_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        d_4_completeNow_: bool
                        d_4_completeNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_4_completeNow_:
                            d_5_closedGenerated_: _dafny.Seq
                            d_6_closedInside_: bool
                            d_7_closedCurrent_: _dafny.Seq
                            out2_: _dafny.Seq
                            out3_: bool
                            out4_: _dafny.Seq
                            out2_, out3_, out4_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_5_closedGenerated_ = out2_
                            d_6_closedInside_ = out3_
                            d_7_closedCurrent_ = out4_
                            generated = d_5_closedGenerated_
                            insideConstrainedOut = d_6_closedInside_
                            currentConstrainedOut = d_7_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_8_stablePrefix_: _dafny.Seq
                            d_8_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_9_constrainedPrompt_: _dafny.Seq
                            d_9_constrainedPrompt_ = (prompt) + (d_8_stablePrefix_)
                            (lm).GenerateLogits((d_9_constrainedPrompt_) + (currentConstrainedOut))
                            d_10_candidates_: _dafny.Seq
                            out5_: _dafny.Seq
                            out5_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_9_constrainedPrompt_, currentConstrainedOut, 48, eosToken)
                            d_10_candidates_ = out5_
                            if (len(d_2_preferredFlat_)) > (0):
                                d_11_preferredCandidates_: _dafny.Seq
                                out6_: _dafny.Seq
                                out6_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_10_candidates_, d_2_preferredFlat_)
                                d_11_preferredCandidates_ = out6_
                                if (len(d_11_preferredCandidates_)) > (0):
                                    (d_0_helpers_).BoostTokenLogits(lm, d_11_preferredCandidates_, _dafny.BigRational('8e0'))
                            if (len(currentConstrainedOut)) > (0):
                                d_12_lastTok_: _dafny.Seq
                                d_12_lastTok_ = (currentConstrainedOut)[(len(currentConstrainedOut)) - (1)]
                                if (d_12_lastTok_) in ((lm).Tokens):
                                    (d_0_helpers_).PenalizeTokenLogits(lm, _dafny.SeqWithoutIsStrInference([d_12_lastTok_]), _dafny.BigRational('2e0'))
                                d_13_gidx_: int
                                out7_: int
                                out7_ = VerifiedDecoderAgent.CSDHelpers.GroupContaining(validTokenGroups, d_12_lastTok_)
                                d_13_gidx_ = out7_
                                if (d_13_gidx_) >= (0):
                                    d_14_group_: _dafny.Seq
                                    d_14_group_ = (validTokenGroups)[d_13_gidx_]
                                    d_15_anyValid_: bool
                                    out8_: bool
                                    out8_ = (d_0_helpers_).GroupHasValidMember(parser, currentConstrainedOut, d_14_group_)
                                    d_15_anyValid_ = out8_
                                    if d_15_anyValid_:
                                        d_16_sameGroupRaw_: _dafny.Seq
                                        out9_: _dafny.Seq
                                        out9_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_14_group_, d_2_preferredFlat_)
                                        d_16_sameGroupRaw_ = out9_
                                        d_17_sameGroupCandidates_: _dafny.Seq
                                        out10_: _dafny.Seq
                                        out10_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_16_sameGroupRaw_, d_10_candidates_)
                                        d_17_sameGroupCandidates_ = out10_
                                        if (len(d_17_sameGroupCandidates_)) > (0):
                                            (d_0_helpers_).BoostTokenLogits(lm, d_17_sameGroupCandidates_, _dafny.BigRational('3e0'))
                            (lm).MaskValidNextAndEos(parser, currentConstrainedOut, eosToken)
                            d_18_next2_: _dafny.Seq
                            out11_: _dafny.Seq
                            out11_ = (lm).ChooseNextToken()
                            d_18_next2_ = out11_
                            (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_18_next2_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_19_appendedGenerated_: _dafny.Seq
                                d_20_appendedInside_: bool
                                d_21_appendedCurrent_: _dafny.Seq
                                out12_: _dafny.Seq
                                out13_: bool
                                out14_: _dafny.Seq
                                out12_, out13_, out14_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_18_next2_)
                                d_19_appendedGenerated_ = out12_
                                d_20_appendedInside_ = out13_
                                d_21_appendedCurrent_ = out14_
                                generated = d_19_appendedGenerated_
                                insideConstrainedOut = d_20_appendedInside_
                                currentConstrainedOut = d_21_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

