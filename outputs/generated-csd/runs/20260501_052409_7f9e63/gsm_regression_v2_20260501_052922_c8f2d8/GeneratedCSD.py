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
        d_2_narrowThreshold_: int
        d_2_narrowThreshold_ = 12
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_3_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_3_next_ = out0_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_3_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_3_next_]))
                            if (d_3_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        d_4_complete_: bool
                        d_4_complete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_4_complete_:
                            d_5_closedGenerated_: _dafny.Seq
                            d_6_closedInside_: bool
                            d_7_closedCurrent_: _dafny.Seq
                            out1_: _dafny.Seq
                            out2_: bool
                            out3_: _dafny.Seq
                            out1_, out2_, out3_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_5_closedGenerated_ = out1_
                            d_6_closedInside_ = out2_
                            d_7_closedCurrent_ = out3_
                            generated = d_5_closedGenerated_
                            insideConstrainedOut = d_6_closedInside_
                            currentConstrainedOut = d_7_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_8_stablePrefix_: _dafny.Seq
                            d_8_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_9_validCount_: int
                            out4_: int
                            out4_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                            d_9_validCount_ = out4_
                            if (d_9_validCount_) <= (d_2_narrowThreshold_):
                                d_10_next_: _dafny.Seq
                                out5_: _dafny.Seq
                                out5_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, (prompt) + (d_8_stablePrefix_), currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_2_narrowThreshold_, eosToken)
                                d_10_next_ = out5_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_10_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_11_appendedGenerated_: _dafny.Seq
                                    d_12_appendedInside_: bool
                                    d_13_appendedCurrent_: _dafny.Seq
                                    out6_: _dafny.Seq
                                    out7_: bool
                                    out8_: _dafny.Seq
                                    out6_, out7_, out8_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_10_next_)
                                    d_11_appendedGenerated_ = out6_
                                    d_12_appendedInside_ = out7_
                                    d_13_appendedCurrent_ = out8_
                                    generated = d_11_appendedGenerated_
                                    insideConstrainedOut = d_12_appendedInside_
                                    currentConstrainedOut = d_13_appendedCurrent_
                            elif True:
                                d_14_rhsTokens_: _dafny.Seq
                                out9_: _dafny.Seq
                                out9_ = VerifiedDecoderAgent.CSDHelpers.ExtractAfterKeyword(currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")))
                                d_14_rhsTokens_ = out9_
                                d_15_beforeEqTok_: _dafny.Seq
                                d_16_foundEqLeft_: bool
                                out10_: _dafny.Seq
                                out11_: bool
                                out10_, out11_ = (d_0_helpers_).LastTokenBefore(currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")))
                                d_15_beforeEqTok_ = out10_
                                d_16_foundEqLeft_ = out11_
                                d_17_flatPreferred_: _dafny.Seq
                                out12_: _dafny.Seq
                                out12_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
                                d_17_flatPreferred_ = out12_
                                (lm).GenerateLogits(((prompt) + (d_8_stablePrefix_)) + (currentConstrainedOut))
                                if (len(d_14_rhsTokens_)) > (0):
                                    (d_0_helpers_).BoostTokenLogits(lm, d_14_rhsTokens_, _dafny.BigRational('6e0'))
                                if (d_16_foundEqLeft_) and ((d_15_beforeEqTok_) in ((lm).Tokens)):
                                    (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([d_15_beforeEqTok_]), _dafny.BigRational('3e0'))
                                if (len(d_17_flatPreferred_)) > (0):
                                    d_18_preferredValid_: bool
                                    out13_: bool
                                    out13_ = (d_0_helpers_).GroupHasValidMember(parser, currentConstrainedOut, d_17_flatPreferred_)
                                    d_18_preferredValid_ = out13_
                                    if d_18_preferredValid_:
                                        (d_0_helpers_).BoostTokenLogits(lm, d_17_flatPreferred_, _dafny.BigRational('2e0'))
                                (lm).MaskValidNextAndEos(parser, currentConstrainedOut, eosToken)
                                d_19_next_: _dafny.Seq
                                out14_: _dafny.Seq
                                out14_ = (lm).ChooseNextToken()
                                d_19_next_ = out14_
                                (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_19_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_20_appendedGenerated_: _dafny.Seq
                                    d_21_appendedInside_: bool
                                    d_22_appendedCurrent_: _dafny.Seq
                                    out15_: _dafny.Seq
                                    out16_: bool
                                    out17_: _dafny.Seq
                                    out15_, out16_, out17_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_19_next_)
                                    d_20_appendedGenerated_ = out15_
                                    d_21_appendedInside_ = out16_
                                    d_22_appendedCurrent_ = out17_
                                    generated = d_20_appendedGenerated_
                                    insideConstrainedOut = d_21_appendedInside_
                                    currentConstrainedOut = d_22_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

