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
        d_2_recentNumbers_: _dafny.Seq
        d_2_recentNumbers_ = _dafny.SeqWithoutIsStrInference([])
        d_3_activePreferred_: _dafny.Seq
        d_3_activePreferred_ = _dafny.SeqWithoutIsStrInference([])
        d_4_openWarmup_: int
        d_4_openWarmup_ = 8
        d_5_earlySpanThreshold_: int
        d_5_earlySpanThreshold_ = 2
        d_6_longSpanThreshold_: int
        d_6_longSpanThreshold_ = 6
        d_7_narrowThreshold_: int
        d_7_narrowThreshold_ = 8
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        (lm).GenerateLogits((prompt) + (generated))
                        if ((d_4_openWarmup_) <= (d_1_steps_)) and ((_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))) in ((lm).Tokens)):
                            (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]), _dafny.BigRational('8e0'))
                        d_8_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (lm).ChooseNextTokenUnconstrained()
                        d_8_next_ = out0_
                        (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_8_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_8_next_]))
                            if (d_8_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                d_2_recentNumbers_ = _dafny.SeqWithoutIsStrInference([])
                                d_3_activePreferred_ = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        d_9_isComplete_: bool
                        d_9_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_9_isComplete_:
                            d_10_closedGenerated_: _dafny.Seq
                            d_11_closedInside_: bool
                            d_12_closedCurrent_: _dafny.Seq
                            out1_: _dafny.Seq
                            out2_: bool
                            out3_: _dafny.Seq
                            out1_, out2_, out3_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_10_closedGenerated_ = out1_
                            d_11_closedInside_ = out2_
                            d_12_closedCurrent_ = out3_
                            generated = d_10_closedGenerated_
                            insideConstrainedOut = d_11_closedInside_
                            currentConstrainedOut = d_12_closedCurrent_
                            d_2_recentNumbers_ = _dafny.SeqWithoutIsStrInference([])
                            d_3_activePreferred_ = _dafny.SeqWithoutIsStrInference([])
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_13_stablePrefix_: _dafny.Seq
                            d_13_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_14_validCount_: int
                            out4_: int
                            out4_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                            d_14_validCount_ = out4_
                            if ((len(currentConstrainedOut)) >= (d_6_longSpanThreshold_)) or ((d_14_validCount_) <= (d_7_narrowThreshold_)):
                                d_15_constrainedGenerated_: _dafny.Seq
                                d_16_constrainedInside_: bool
                                d_17_constrainedCurrent_: _dafny.Seq
                                d_18_hitEos_: bool
                                out5_: _dafny.Seq
                                out6_: bool
                                out7_: _dafny.Seq
                                out8_: bool
                                out5_, out6_, out7_, out8_ = (d_0_helpers_).ConstrainedStep(lm, parser, (prompt) + (d_13_stablePrefix_), generated, currentConstrainedOut, eosToken)
                                d_15_constrainedGenerated_ = out5_
                                d_16_constrainedInside_ = out6_
                                d_17_constrainedCurrent_ = out7_
                                d_18_hitEos_ = out8_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if d_18_hitEos_:
                                    raise _dafny.Break("0")
                                elif True:
                                    generated = d_15_constrainedGenerated_
                                    insideConstrainedOut = d_16_constrainedInside_
                                    currentConstrainedOut = d_17_constrainedCurrent_
                                    out9_: _dafny.Seq
                                    out9_ = VerifiedDecoderAgent.CSDHelpers.ExtractAfterKeyword(currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")))
                                    d_2_recentNumbers_ = out9_
                                    d_19_flatPreferred1_: _dafny.Seq
                                    out10_: _dafny.Seq
                                    out10_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
                                    d_19_flatPreferred1_ = out10_
                                    out11_: _dafny.Seq
                                    out11_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_19_flatPreferred1_, d_2_recentNumbers_)
                                    d_3_activePreferred_ = out11_
                            elif True:
                                out12_: _dafny.Seq
                                out12_ = VerifiedDecoderAgent.CSDHelpers.ExtractAfterKeyword(currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")))
                                d_2_recentNumbers_ = out12_
                                d_20_flatPreferred_: _dafny.Seq
                                out13_: _dafny.Seq
                                out13_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
                                d_20_flatPreferred_ = out13_
                                out14_: _dafny.Seq
                                out14_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_20_flatPreferred_, d_2_recentNumbers_)
                                d_3_activePreferred_ = out14_
                                if ((len(currentConstrainedOut)) <= (d_5_earlySpanThreshold_)) and ((len(d_3_activePreferred_)) > (0)):
                                    (lm).GenerateLogits(((prompt) + (d_13_stablePrefix_)) + (currentConstrainedOut))
                                    (d_0_helpers_).BoostTokenLogits(lm, d_3_activePreferred_, _dafny.BigRational('3e0'))
                                    if (len(validTokenGroups)) > (0):
                                        (d_0_helpers_).BoostValidGroups(lm, parser, currentConstrainedOut, validTokenGroups, _dafny.BigRational('2e0'))
                                    (lm).MaskValidNextAndEos(parser, currentConstrainedOut, eosToken)
                                    d_21_next2_: _dafny.Seq
                                    out15_: _dafny.Seq
                                    out15_ = (lm).ChooseNextToken()
                                    d_21_next2_ = out15_
                                    (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    if (d_21_next2_) == (eosToken):
                                        raise _dafny.Break("0")
                                    elif True:
                                        d_22_appendedGenerated2_: _dafny.Seq
                                        d_23_appendedInside2_: bool
                                        d_24_appendedCurrent2_: _dafny.Seq
                                        out16_: _dafny.Seq
                                        out17_: bool
                                        out18_: _dafny.Seq
                                        out16_, out17_, out18_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_21_next2_)
                                        d_22_appendedGenerated2_ = out16_
                                        d_23_appendedInside2_ = out17_
                                        d_24_appendedCurrent2_ = out18_
                                        generated = d_22_appendedGenerated2_
                                        insideConstrainedOut = d_23_appendedInside2_
                                        currentConstrainedOut = d_24_appendedCurrent2_
                                        d_25_idx2_: int
                                        out19_: int
                                        out19_ = VerifiedDecoderAgent.CSDHelpers.GroupContaining(validTokenGroups, d_21_next2_)
                                        d_25_idx2_ = out19_
                                        if (d_25_idx2_) >= (0):
                                            d_3_activePreferred_ = (validTokenGroups)[d_25_idx2_]
                                elif True:
                                    d_26_next3_: _dafny.Seq
                                    out20_: _dafny.Seq
                                    out20_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, (prompt) + (d_13_stablePrefix_), currentConstrainedOut, validTokenGroups, _dafny.BigRational('2e0'), 10, eosToken)
                                    d_26_next3_ = out20_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    if (d_26_next3_) == (eosToken):
                                        raise _dafny.Break("0")
                                    elif True:
                                        d_27_appendedGenerated3_: _dafny.Seq
                                        d_28_appendedInside3_: bool
                                        d_29_appendedCurrent3_: _dafny.Seq
                                        out21_: _dafny.Seq
                                        out22_: bool
                                        out23_: _dafny.Seq
                                        out21_, out22_, out23_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_26_next3_)
                                        d_27_appendedGenerated3_ = out21_
                                        d_28_appendedInside3_ = out22_
                                        d_29_appendedCurrent3_ = out23_
                                        generated = d_27_appendedGenerated3_
                                        insideConstrainedOut = d_28_appendedInside3_
                                        currentConstrainedOut = d_29_appendedCurrent3_
                                        d_30_idx3_: int
                                        out24_: int
                                        out24_ = VerifiedDecoderAgent.CSDHelpers.GroupContaining(validTokenGroups, d_26_next3_)
                                        d_30_idx3_ = out24_
                                        if (d_30_idx3_) >= (0):
                                            d_3_activePreferred_ = (validTokenGroups)[d_30_idx3_]
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

