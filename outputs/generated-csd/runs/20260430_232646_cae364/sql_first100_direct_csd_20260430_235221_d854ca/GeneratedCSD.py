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
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_2_openedGenerated_: _dafny.Seq
                        d_3_openedInside_: bool
                        d_4_openedCurrent_: _dafny.Seq
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: _dafny.Seq
                        out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                        d_2_openedGenerated_ = out0_
                        d_3_openedInside_ = out1_
                        d_4_openedCurrent_ = out2_
                        generated = d_2_openedGenerated_
                        insideConstrainedOut = d_3_openedInside_
                        currentConstrainedOut = d_4_openedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            d_5_closedGenerated_: _dafny.Seq
                            d_6_closedInside_: bool
                            d_7_closedCurrent_: _dafny.Seq
                            out3_: _dafny.Seq
                            out4_: bool
                            out5_: _dafny.Seq
                            out3_, out4_, out5_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_5_closedGenerated_ = out3_
                            d_6_closedInside_ = out4_
                            d_7_closedCurrent_ = out5_
                            generated = d_5_closedGenerated_
                            insideConstrainedOut = d_6_closedInside_
                            currentConstrainedOut = d_7_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                            raise _dafny.Break("0")
                        elif True:
                            d_8_stablePrefix_: _dafny.Seq
                            d_8_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_9_usePenalty_: bool
                            out6_: bool
                            out6_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 2)
                            d_9_usePenalty_ = out6_
                            if not(d_9_usePenalty_):
                                d_10_prevTok_: _dafny.Seq
                                d_11_foundPrev_: bool
                                out7_: _dafny.Seq
                                out8_: bool
                                out7_, out8_ = (d_0_helpers_).LastTokenBefore(currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ".")))
                                d_10_prevTok_ = out7_
                                d_11_foundPrev_ = out8_
                                if d_11_foundPrev_:
                                    d_12_flatGroups_: _dafny.Seq
                                    out9_: _dafny.Seq
                                    out9_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
                                    d_12_flatGroups_ = out9_
                                    if (d_10_prevTok_) in (d_12_flatGroups_):
                                        d_9_usePenalty_ = True
                            if d_9_usePenalty_:
                                d_13_flatGroups_: _dafny.Seq
                                out10_: _dafny.Seq
                                out10_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
                                d_13_flatGroups_ = out10_
                                d_14_penalizeTokens_: _dafny.Seq
                                out11_: _dafny.Seq
                                out11_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_13_flatGroups_, (lm).Tokens)
                                d_14_penalizeTokens_ = out11_
                                d_15_next_: _dafny.Seq
                                out12_: _dafny.Seq
                                out12_ = (d_0_helpers_).PenalizedConstrainedStep(lm, parser, (prompt) + (d_8_stablePrefix_), currentConstrainedOut, d_14_penalizeTokens_, _dafny.BigRational('3e0'), eosToken)
                                d_15_next_ = out12_
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
                            elif True:
                                d_19_steppedGenerated_: _dafny.Seq
                                d_20_steppedInside_: bool
                                d_21_steppedCurrent_: _dafny.Seq
                                d_22_hitEos_: bool
                                out16_: _dafny.Seq
                                out17_: bool
                                out18_: _dafny.Seq
                                out19_: bool
                                out16_, out17_, out18_, out19_ = (d_0_helpers_).ConstrainedStep(lm, parser, (prompt) + (d_8_stablePrefix_), generated, currentConstrainedOut, eosToken)
                                d_19_steppedGenerated_ = out16_
                                d_20_steppedInside_ = out17_
                                d_21_steppedCurrent_ = out18_
                                d_22_hitEos_ = out19_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if d_22_hitEos_:
                                    raise _dafny.Break("0")
                                elif True:
                                    generated = d_19_steppedGenerated_
                                    insideConstrainedOut = d_20_steppedInside_
                                    currentConstrainedOut = d_21_steppedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

