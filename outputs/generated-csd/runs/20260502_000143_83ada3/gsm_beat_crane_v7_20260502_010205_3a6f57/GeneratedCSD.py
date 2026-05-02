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
        d_2_spansClosed_: int
        d_2_spansClosed_ = 0
        d_3_recentNumbers_: _dafny.Seq
        d_3_recentNumbers_ = _dafny.SeqWithoutIsStrInference([])
        d_4_activePreferred_: _dafny.Seq
        d_4_activePreferred_ = _dafny.SeqWithoutIsStrInference([])
        d_5_openWarmup_: int
        d_5_openWarmup_ = 8
        d_6_earlySpanThreshold_: int
        d_6_earlySpanThreshold_ = 2
        d_7_longSpanThreshold_: int
        d_7_longSpanThreshold_ = 6
        d_8_narrowThreshold_: int
        d_8_narrowThreshold_ = 8
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if (d_2_spansClosed_) > (0):
                            raise _dafny.Break("0")
                        elif True:
                            (lm).GenerateLogits((prompt) + (generated))
                            if ((d_5_openWarmup_) <= (d_1_steps_)) and ((_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))) in ((lm).Tokens)):
                                (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]), _dafny.BigRational('8e0'))
                            d_9_next_: _dafny.Seq
                            out0_: _dafny.Seq
                            out0_ = (lm).ChooseNextTokenUnconstrained()
                            d_9_next_ = out0_
                            (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_9_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_9_next_]))
                                if (d_9_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    insideConstrainedOut = True
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                    d_3_recentNumbers_ = _dafny.SeqWithoutIsStrInference([])
                                    d_4_activePreferred_ = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        d_10_isComplete_: bool
                        d_10_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_10_isComplete_:
                            d_11_closedGenerated_: _dafny.Seq
                            d_12_closedInside_: bool
                            d_13_closedCurrent_: _dafny.Seq
                            out1_: _dafny.Seq
                            out2_: bool
                            out3_: _dafny.Seq
                            out1_, out2_, out3_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_11_closedGenerated_ = out1_
                            d_12_closedInside_ = out2_
                            d_13_closedCurrent_ = out3_
                            generated = d_11_closedGenerated_
                            insideConstrainedOut = d_12_closedInside_
                            currentConstrainedOut = d_13_closedCurrent_
                            d_3_recentNumbers_ = _dafny.SeqWithoutIsStrInference([])
                            d_4_activePreferred_ = _dafny.SeqWithoutIsStrInference([])
                            d_2_spansClosed_ = (d_2_spansClosed_) + (1)
                            d_1_steps_ = (d_1_steps_) + (1)
                            raise _dafny.Break("0")
                        elif True:
                            d_14_stablePrefix_: _dafny.Seq
                            d_14_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_15_validCount_: int
                            out4_: int
                            out4_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                            d_15_validCount_ = out4_
                            if ((len(currentConstrainedOut)) >= (d_7_longSpanThreshold_)) or ((d_15_validCount_) <= (d_8_narrowThreshold_)):
                                d_16_constrainedGenerated_: _dafny.Seq
                                d_17_constrainedInside_: bool
                                d_18_constrainedCurrent_: _dafny.Seq
                                d_19_hitEos_: bool
                                out5_: _dafny.Seq
                                out6_: bool
                                out7_: _dafny.Seq
                                out8_: bool
                                out5_, out6_, out7_, out8_ = (d_0_helpers_).ConstrainedStep(lm, parser, (prompt) + (d_14_stablePrefix_), generated, currentConstrainedOut, eosToken)
                                d_16_constrainedGenerated_ = out5_
                                d_17_constrainedInside_ = out6_
                                d_18_constrainedCurrent_ = out7_
                                d_19_hitEos_ = out8_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if d_19_hitEos_:
                                    raise _dafny.Break("0")
                                elif True:
                                    generated = d_16_constrainedGenerated_
                                    insideConstrainedOut = d_17_constrainedInside_
                                    currentConstrainedOut = d_18_constrainedCurrent_
                                    out9_: _dafny.Seq
                                    out9_ = VerifiedDecoderAgent.CSDHelpers.ExtractAfterKeyword(currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")))
                                    d_3_recentNumbers_ = out9_
                                    d_20_flatPreferred1_: _dafny.Seq
                                    out10_: _dafny.Seq
                                    out10_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
                                    d_20_flatPreferred1_ = out10_
                                    out11_: _dafny.Seq
                                    out11_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_20_flatPreferred1_, d_3_recentNumbers_)
                                    d_4_activePreferred_ = out11_
                            elif True:
                                out12_: _dafny.Seq
                                out12_ = VerifiedDecoderAgent.CSDHelpers.ExtractAfterKeyword(currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")))
                                d_3_recentNumbers_ = out12_
                                d_21_flatPreferred_: _dafny.Seq
                                out13_: _dafny.Seq
                                out13_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
                                d_21_flatPreferred_ = out13_
                                out14_: _dafny.Seq
                                out14_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_21_flatPreferred_, d_3_recentNumbers_)
                                d_4_activePreferred_ = out14_
                                if ((len(currentConstrainedOut)) <= (d_6_earlySpanThreshold_)) and ((len(d_4_activePreferred_)) > (0)):
                                    (lm).GenerateLogits(((prompt) + (d_14_stablePrefix_)) + (currentConstrainedOut))
                                    (d_0_helpers_).BoostTokenLogits(lm, d_4_activePreferred_, _dafny.BigRational('3e0'))
                                    if (len(validTokenGroups)) > (0):
                                        (d_0_helpers_).BoostValidGroups(lm, parser, currentConstrainedOut, validTokenGroups, _dafny.BigRational('2e0'))
                                    (lm).MaskValidNextAndEos(parser, currentConstrainedOut, eosToken)
                                    d_22_next2_: _dafny.Seq
                                    out15_: _dafny.Seq
                                    out15_ = (lm).ChooseNextToken()
                                    d_22_next2_ = out15_
                                    (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    if (d_22_next2_) == (eosToken):
                                        raise _dafny.Break("0")
                                    elif True:
                                        d_23_appendedGenerated2_: _dafny.Seq
                                        d_24_appendedInside2_: bool
                                        d_25_appendedCurrent2_: _dafny.Seq
                                        out16_: _dafny.Seq
                                        out17_: bool
                                        out18_: _dafny.Seq
                                        out16_, out17_, out18_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_22_next2_)
                                        d_23_appendedGenerated2_ = out16_
                                        d_24_appendedInside2_ = out17_
                                        d_25_appendedCurrent2_ = out18_
                                        generated = d_23_appendedGenerated2_
                                        insideConstrainedOut = d_24_appendedInside2_
                                        currentConstrainedOut = d_25_appendedCurrent2_
                                        d_26_idx2_: int
                                        out19_: int
                                        out19_ = VerifiedDecoderAgent.CSDHelpers.GroupContaining(validTokenGroups, d_22_next2_)
                                        d_26_idx2_ = out19_
                                        if (d_26_idx2_) >= (0):
                                            d_4_activePreferred_ = (validTokenGroups)[d_26_idx2_]
                                elif True:
                                    d_27_next3_: _dafny.Seq
                                    out20_: _dafny.Seq
                                    out20_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, (prompt) + (d_14_stablePrefix_), currentConstrainedOut, validTokenGroups, _dafny.BigRational('2e0'), 10, eosToken)
                                    d_27_next3_ = out20_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    if (d_27_next3_) == (eosToken):
                                        raise _dafny.Break("0")
                                    elif True:
                                        d_28_appendedGenerated3_: _dafny.Seq
                                        d_29_appendedInside3_: bool
                                        d_30_appendedCurrent3_: _dafny.Seq
                                        out21_: _dafny.Seq
                                        out22_: bool
                                        out23_: _dafny.Seq
                                        out21_, out22_, out23_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_27_next3_)
                                        d_28_appendedGenerated3_ = out21_
                                        d_29_appendedInside3_ = out22_
                                        d_30_appendedCurrent3_ = out23_
                                        generated = d_28_appendedGenerated3_
                                        insideConstrainedOut = d_29_appendedInside3_
                                        currentConstrainedOut = d_30_appendedCurrent3_
                                        d_31_idx3_: int
                                        out24_: int
                                        out24_ = VerifiedDecoderAgent.CSDHelpers.GroupContaining(validTokenGroups, d_27_next3_)
                                        d_31_idx3_ = out24_
                                        if (d_31_idx3_) >= (0):
                                            d_4_activePreferred_ = (validTokenGroups)[d_31_idx3_]
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

