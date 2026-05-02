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
                                d_2_recentNumbers_ = _dafny.SeqWithoutIsStrInference([])
                                d_3_activePreferred_ = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        if (parser).IsCompletePrefix(currentConstrainedOut):
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
                            d_2_recentNumbers_ = _dafny.SeqWithoutIsStrInference([])
                            d_3_activePreferred_ = _dafny.SeqWithoutIsStrInference([])
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            out4_: _dafny.Seq
                            out4_ = VerifiedDecoderAgent.CSDHelpers.ExtractAfterKeyword(currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")))
                            d_2_recentNumbers_ = out4_
                            d_8_flatPreferred_: _dafny.Seq
                            out5_: _dafny.Seq
                            out5_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
                            d_8_flatPreferred_ = out5_
                            out6_: _dafny.Seq
                            out6_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_8_flatPreferred_, d_2_recentNumbers_)
                            d_3_activePreferred_ = out6_
                            d_9_stablePrefix_: _dafny.Seq
                            d_9_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            if (len(d_3_activePreferred_)) == (0):
                                d_10_next_: _dafny.Seq
                                out7_: _dafny.Seq
                                out7_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, (prompt) + (d_9_stablePrefix_), currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                                d_10_next_ = out7_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_10_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_11_appendedGenerated_: _dafny.Seq
                                    d_12_appendedInside_: bool
                                    d_13_appendedCurrent_: _dafny.Seq
                                    out8_: _dafny.Seq
                                    out9_: bool
                                    out10_: _dafny.Seq
                                    out8_, out9_, out10_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_10_next_)
                                    d_11_appendedGenerated_ = out8_
                                    d_12_appendedInside_ = out9_
                                    d_13_appendedCurrent_ = out10_
                                    generated = d_11_appendedGenerated_
                                    insideConstrainedOut = d_12_appendedInside_
                                    currentConstrainedOut = d_13_appendedCurrent_
                                    d_14_idx_: int
                                    out11_: int
                                    out11_ = VerifiedDecoderAgent.CSDHelpers.GroupContaining(validTokenGroups, d_10_next_)
                                    d_14_idx_ = out11_
                                    if (d_14_idx_) >= (0):
                                        d_3_activePreferred_ = (validTokenGroups)[d_14_idx_]
                            elif True:
                                (lm).GenerateLogits(((prompt) + (d_9_stablePrefix_)) + (currentConstrainedOut))
                                (d_0_helpers_).BoostTokenLogits(lm, d_3_activePreferred_, _dafny.BigRational('8e0'))
                                if (len(validTokenGroups)) > (0):
                                    (d_0_helpers_).BoostValidGroups(lm, parser, currentConstrainedOut, validTokenGroups, _dafny.BigRational('3e0'))
                                (lm).MaskValidNextAndEos(parser, currentConstrainedOut, eosToken)
                                d_15_next_: _dafny.Seq
                                out12_: _dafny.Seq
                                out12_ = (lm).ChooseNextToken()
                                d_15_next_ = out12_
                                (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_15_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_16_appendedGenerated_: _dafny.Seq
                                    d_17_appendedInside_: bool
                                    d_18_appendedCurrent_: _dafny.Seq
                                    out13_: _dafny.Seq
                                    out14_: bool
                                    out15_: _dafny.Seq
                                    out13_, out14_, out15_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_15_next_)
                                    d_16_appendedGenerated_ = out13_
                                    d_17_appendedInside_ = out14_
                                    d_18_appendedCurrent_ = out15_
                                    generated = d_16_appendedGenerated_
                                    insideConstrainedOut = d_17_appendedInside_
                                    currentConstrainedOut = d_18_appendedCurrent_
                                    d_19_idx_: int
                                    out16_: int
                                    out16_ = VerifiedDecoderAgent.CSDHelpers.GroupContaining(validTokenGroups, d_15_next_)
                                    d_19_idx_ = out16_
                                    if (d_19_idx_) >= (0):
                                        d_3_activePreferred_ = (validTokenGroups)[d_19_idx_]
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

