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
        d_2_narrowThreshold_ = 10
        d_3_longPrefixThreshold_: int
        d_3_longPrefixThreshold_ = 40
        d_4_flatGroups_: _dafny.Seq
        out0_: _dafny.Seq
        out0_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
        d_4_flatGroups_ = out0_
        d_5_penalizableTokens_: _dafny.Seq
        out1_: _dafny.Seq
        out1_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_4_flatGroups_, (lm).Tokens)
        d_5_penalizableTokens_ = out1_
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))) in ((lm).Tokens):
                            d_6_openedGenerated_: _dafny.Seq
                            d_7_openedInside_: bool
                            d_8_openedCurrent_: _dafny.Seq
                            out2_: _dafny.Seq
                            out3_: bool
                            out4_: _dafny.Seq
                            out2_, out3_, out4_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_6_openedGenerated_ = out2_
                            d_7_openedInside_ = out3_
                            d_8_openedCurrent_ = out4_
                            generated = d_6_openedGenerated_
                            insideConstrainedOut = d_7_openedInside_
                            currentConstrainedOut = d_8_openedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            raise _dafny.Break("0")
                    elif True:
                        d_9_completeNow_: bool
                        d_9_completeNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_9_completeNow_:
                            if (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))) in ((lm).Tokens):
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
                                raise _dafny.Break("0")
                        elif True:
                            d_13_stablePrefix_: _dafny.Seq
                            d_13_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_14_constrainedPrompt_: _dafny.Seq
                            d_14_constrainedPrompt_ = (prompt) + (d_13_stablePrefix_)
                            d_15_validCount_: int
                            out8_: int
                            out8_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                            d_15_validCount_ = out8_
                            d_16_nearDeadEnd_: bool
                            out9_: bool
                            out9_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 2)
                            d_16_nearDeadEnd_ = out9_
                            if (d_16_nearDeadEnd_) or ((len(currentConstrainedOut)) >= (d_3_longPrefixThreshold_)):
                                d_17_nextPen_: _dafny.Seq
                                out10_: _dafny.Seq
                                out10_ = (d_0_helpers_).PenalizedConstrainedStep(lm, parser, d_14_constrainedPrompt_, currentConstrainedOut, d_5_penalizableTokens_, _dafny.BigRational('5e0'), eosToken)
                                d_17_nextPen_ = out10_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_17_nextPen_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_18_appendedGenerated1_: _dafny.Seq
                                    d_19_appendedInside1_: bool
                                    d_20_appendedCurrent1_: _dafny.Seq
                                    out11_: _dafny.Seq
                                    out12_: bool
                                    out13_: _dafny.Seq
                                    out11_, out12_, out13_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_17_nextPen_)
                                    d_18_appendedGenerated1_ = out11_
                                    d_19_appendedInside1_ = out12_
                                    d_20_appendedCurrent1_ = out13_
                                    generated = d_18_appendedGenerated1_
                                    insideConstrainedOut = d_19_appendedInside1_
                                    currentConstrainedOut = d_20_appendedCurrent1_
                            elif True:
                                d_21_next_: _dafny.Seq
                                out14_: _dafny.Seq
                                out14_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_14_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_2_narrowThreshold_, eosToken)
                                d_21_next_ = out14_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_21_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_22_appendedGenerated2_: _dafny.Seq
                                    d_23_appendedInside2_: bool
                                    d_24_appendedCurrent2_: _dafny.Seq
                                    out15_: _dafny.Seq
                                    out16_: bool
                                    out17_: _dafny.Seq
                                    out15_, out16_, out17_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_21_next_)
                                    d_22_appendedGenerated2_ = out15_
                                    d_23_appendedInside2_ = out16_
                                    d_24_appendedCurrent2_ = out17_
                                    generated = d_22_appendedGenerated2_
                                    insideConstrainedOut = d_23_appendedInside2_
                                    currentConstrainedOut = d_24_appendedCurrent2_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

