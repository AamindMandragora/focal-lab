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
        d_4_longSpanThreshold_: int
        d_4_longSpanThreshold_ = 10
        d_5_narrowThreshold_: int
        d_5_narrowThreshold_ = 6
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_6_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_6_next_ = out0_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_6_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_6_next_]))
                            if (d_6_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                d_2_recentNumbers_ = _dafny.SeqWithoutIsStrInference([])
                                d_3_activePreferred_ = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        d_7_isComplete_: bool
                        d_7_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_7_isComplete_:
                            d_8_closedGenerated_: _dafny.Seq
                            d_9_closedInside_: bool
                            d_10_closedCurrent_: _dafny.Seq
                            out1_: _dafny.Seq
                            out2_: bool
                            out3_: _dafny.Seq
                            out1_, out2_, out3_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_8_closedGenerated_ = out1_
                            d_9_closedInside_ = out2_
                            d_10_closedCurrent_ = out3_
                            generated = d_8_closedGenerated_
                            insideConstrainedOut = d_9_closedInside_
                            currentConstrainedOut = d_10_closedCurrent_
                            d_2_recentNumbers_ = _dafny.SeqWithoutIsStrInference([])
                            d_3_activePreferred_ = _dafny.SeqWithoutIsStrInference([])
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_11_stablePrefix_: _dafny.Seq
                            d_11_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_12_validCount_: int
                            out4_: int
                            out4_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                            d_12_validCount_ = out4_
                            if ((len(currentConstrainedOut)) >= (d_4_longSpanThreshold_)) or ((d_12_validCount_) <= (d_5_narrowThreshold_)):
                                d_13_constrainedGenerated_: _dafny.Seq
                                d_14_constrainedInside_: bool
                                d_15_constrainedCurrent_: _dafny.Seq
                                d_16_hitEos_: bool
                                out5_: _dafny.Seq
                                out6_: bool
                                out7_: _dafny.Seq
                                out8_: bool
                                out5_, out6_, out7_, out8_ = (d_0_helpers_).ConstrainedStep(lm, parser, (prompt) + (d_11_stablePrefix_), generated, currentConstrainedOut, eosToken)
                                d_13_constrainedGenerated_ = out5_
                                d_14_constrainedInside_ = out6_
                                d_15_constrainedCurrent_ = out7_
                                d_16_hitEos_ = out8_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if d_16_hitEos_:
                                    raise _dafny.Break("0")
                                elif True:
                                    generated = d_13_constrainedGenerated_
                                    insideConstrainedOut = d_14_constrainedInside_
                                    currentConstrainedOut = d_15_constrainedCurrent_
                                    out9_: _dafny.Seq
                                    out9_ = VerifiedDecoderAgent.CSDHelpers.ExtractAfterKeyword(currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")))
                                    d_2_recentNumbers_ = out9_
                                    d_17_flatPreferred1_: _dafny.Seq
                                    out10_: _dafny.Seq
                                    out10_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
                                    d_17_flatPreferred1_ = out10_
                                    out11_: _dafny.Seq
                                    out11_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_17_flatPreferred1_, d_2_recentNumbers_)
                                    d_3_activePreferred_ = out11_
                            elif True:
                                out12_: _dafny.Seq
                                out12_ = VerifiedDecoderAgent.CSDHelpers.ExtractAfterKeyword(currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")))
                                d_2_recentNumbers_ = out12_
                                d_18_flatPreferred_: _dafny.Seq
                                out13_: _dafny.Seq
                                out13_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
                                d_18_flatPreferred_ = out13_
                                out14_: _dafny.Seq
                                out14_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_18_flatPreferred_, d_2_recentNumbers_)
                                d_3_activePreferred_ = out14_
                                if (len(d_3_activePreferred_)) == (0):
                                    d_19_next2_: _dafny.Seq
                                    out15_: _dafny.Seq
                                    out15_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, (prompt) + (d_11_stablePrefix_), currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                                    d_19_next2_ = out15_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    if (d_19_next2_) == (eosToken):
                                        raise _dafny.Break("0")
                                    elif True:
                                        d_20_appendedGenerated2_: _dafny.Seq
                                        d_21_appendedInside2_: bool
                                        d_22_appendedCurrent2_: _dafny.Seq
                                        out16_: _dafny.Seq
                                        out17_: bool
                                        out18_: _dafny.Seq
                                        out16_, out17_, out18_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_19_next2_)
                                        d_20_appendedGenerated2_ = out16_
                                        d_21_appendedInside2_ = out17_
                                        d_22_appendedCurrent2_ = out18_
                                        generated = d_20_appendedGenerated2_
                                        insideConstrainedOut = d_21_appendedInside2_
                                        currentConstrainedOut = d_22_appendedCurrent2_
                                        d_23_idx2_: int
                                        out19_: int
                                        out19_ = VerifiedDecoderAgent.CSDHelpers.GroupContaining(validTokenGroups, d_19_next2_)
                                        d_23_idx2_ = out19_
                                        if (d_23_idx2_) >= (0):
                                            d_3_activePreferred_ = (validTokenGroups)[d_23_idx2_]
                                elif True:
                                    (lm).GenerateLogits(((prompt) + (d_11_stablePrefix_)) + (currentConstrainedOut))
                                    (d_0_helpers_).BoostTokenLogits(lm, d_3_activePreferred_, _dafny.BigRational('8e0'))
                                    if (len(validTokenGroups)) > (0):
                                        (d_0_helpers_).BoostValidGroups(lm, parser, currentConstrainedOut, validTokenGroups, _dafny.BigRational('3e0'))
                                    (lm).MaskValidNextAndEos(parser, currentConstrainedOut, eosToken)
                                    d_24_next3_: _dafny.Seq
                                    out20_: _dafny.Seq
                                    out20_ = (lm).ChooseNextToken()
                                    d_24_next3_ = out20_
                                    (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    if (d_24_next3_) == (eosToken):
                                        raise _dafny.Break("0")
                                    elif True:
                                        d_25_appendedGenerated3_: _dafny.Seq
                                        d_26_appendedInside3_: bool
                                        d_27_appendedCurrent3_: _dafny.Seq
                                        out21_: _dafny.Seq
                                        out22_: bool
                                        out23_: _dafny.Seq
                                        out21_, out22_, out23_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_24_next3_)
                                        d_25_appendedGenerated3_ = out21_
                                        d_26_appendedInside3_ = out22_
                                        d_27_appendedCurrent3_ = out23_
                                        generated = d_25_appendedGenerated3_
                                        insideConstrainedOut = d_26_appendedInside3_
                                        currentConstrainedOut = d_27_appendedCurrent3_
                                        d_28_idx3_: int
                                        out24_: int
                                        out24_ = VerifiedDecoderAgent.CSDHelpers.GroupContaining(validTokenGroups, d_24_next3_)
                                        d_28_idx3_ = out24_
                                        if (d_28_idx3_) >= (0):
                                            d_3_activePreferred_ = (validTokenGroups)[d_28_idx3_]
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

