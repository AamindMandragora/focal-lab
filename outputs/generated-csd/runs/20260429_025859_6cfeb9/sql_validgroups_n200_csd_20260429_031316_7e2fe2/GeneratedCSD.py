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
                        d_4_shouldClose_: bool
                        d_4_shouldClose_ = False
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            d_4_shouldClose_ = True
                            if ((len(currentConstrainedOut)) > (0)) and ((len(validTokenGroups)) > (0)):
                                d_5_lastTok_: _dafny.Seq
                                d_5_lastTok_ = (currentConstrainedOut)[(len(currentConstrainedOut)) - (1)]
                                d_6_gidx_: int
                                out2_: int
                                out2_ = VerifiedDecoderAgent.CSDHelpers.GroupContaining(validTokenGroups, d_5_lastTok_)
                                d_6_gidx_ = out2_
                                if (d_6_gidx_) >= (0):
                                    d_7_group_: _dafny.Seq
                                    d_7_group_ = (validTokenGroups)[d_6_gidx_]
                                    d_8_anyGroupValid_: bool
                                    out3_: bool
                                    out3_ = (d_0_helpers_).GroupHasValidMember(parser, currentConstrainedOut, d_7_group_)
                                    d_8_anyGroupValid_ = out3_
                                    if d_8_anyGroupValid_:
                                        d_9_validLast_: bool
                                        out4_: bool
                                        out4_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_5_lastTok_)
                                        d_9_validLast_ = out4_
                                        if not(d_9_validLast_):
                                            d_4_shouldClose_ = False
                        if d_4_shouldClose_:
                            d_10_closedGenerated_: _dafny.Seq
                            d_11_closedInside_: bool
                            d_12_closedCurrent_: _dafny.Seq
                            out5_: _dafny.Seq
                            out6_: bool
                            out7_: _dafny.Seq
                            out5_, out6_, out7_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_10_closedGenerated_ = out5_
                            d_11_closedInside_ = out6_
                            d_12_closedCurrent_ = out7_
                            generated = d_10_closedGenerated_
                            insideConstrainedOut = d_11_closedInside_
                            currentConstrainedOut = d_12_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_13_stablePrefix_: _dafny.Seq
                            d_13_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_14_constrainedPrompt_: _dafny.Seq
                            d_14_constrainedPrompt_ = (prompt) + (d_13_stablePrefix_)
                            (lm).GenerateLogits((d_14_constrainedPrompt_) + (currentConstrainedOut))
                            if (len(d_2_preferredFlat_)) > (0):
                                d_15_candidates_: _dafny.Seq
                                out8_: _dafny.Seq
                                out8_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_14_constrainedPrompt_, currentConstrainedOut, 32, eosToken)
                                d_15_candidates_ = out8_
                                d_16_preferredCandidates_: _dafny.Seq
                                out9_: _dafny.Seq
                                out9_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_15_candidates_, d_2_preferredFlat_)
                                d_16_preferredCandidates_ = out9_
                                if (len(d_16_preferredCandidates_)) > (0):
                                    (d_0_helpers_).BoostTokenLogits(lm, d_16_preferredCandidates_, _dafny.BigRational('8e0'))
                            if (len(currentConstrainedOut)) > (0):
                                d_17_lastTok2_: _dafny.Seq
                                d_17_lastTok2_ = (currentConstrainedOut)[(len(currentConstrainedOut)) - (1)]
                                if (d_17_lastTok2_) in ((lm).Tokens):
                                    (d_0_helpers_).PenalizeTokenLogits(lm, _dafny.SeqWithoutIsStrInference([d_17_lastTok2_]), _dafny.BigRational('3e0'))
                            d_18_wasComplete_: bool
                            d_18_wasComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                            (lm).MaskValidNextAndEos(parser, currentConstrainedOut, eosToken)
                            d_19_next2_: _dafny.Seq
                            out10_: _dafny.Seq
                            out10_ = (lm).ChooseNextToken()
                            d_19_next2_ = out10_
                            (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_19_next2_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                if not(d_18_wasComplete_):
                                    d_20_appendedGenerated_: _dafny.Seq
                                    d_21_appendedInside_: bool
                                    d_22_appendedCurrent_: _dafny.Seq
                                    out11_: _dafny.Seq
                                    out12_: bool
                                    out13_: _dafny.Seq
                                    out11_, out12_, out13_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_19_next2_)
                                    d_20_appendedGenerated_ = out11_
                                    d_21_appendedInside_ = out12_
                                    d_22_appendedCurrent_ = out13_
                                    generated = d_20_appendedGenerated_
                                    insideConstrainedOut = d_21_appendedInside_
                                    currentConstrainedOut = d_22_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

