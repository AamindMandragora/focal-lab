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
    def MyCSDStrategy(lm, parser, prompt, generatedPrefix, insideConstrained, currentConstrained, maxSteps, eosToken):
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
                    if insideConstrainedOut:
                        d_2_completeNow_: bool
                        d_2_completeNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_2_completeNow_:
                            d_3_closedGenerated_: _dafny.Seq
                            d_4_closedInside_: bool
                            d_5_closedCurrent_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_3_closedGenerated_ = out0_
                            d_4_closedInside_ = out1_
                            d_5_closedCurrent_ = out2_
                            generated = d_3_closedGenerated_
                            insideConstrainedOut = d_4_closedInside_
                            currentConstrainedOut = d_5_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                            cost = d_1_steps_
                        elif True:
                            d_6_remainingInside_: int
                            d_6_remainingInside_ = (maxSteps) - (d_1_steps_)
                            if (d_6_remainingInside_) <= (1):
                                d_7_rolled_: _dafny.Seq
                                out3_: _dafny.Seq
                                out3_ = VerifiedDecoderAgent.CSDHelpers.RollbackToValidPrefix(parser, currentConstrainedOut)
                                d_7_rolled_ = out3_
                                d_8_rbGenerated_: _dafny.Seq
                                d_9_rbCurrent_: _dafny.Seq
                                out4_: _dafny.Seq
                                out5_: _dafny.Seq
                                out4_, out5_ = (d_0_helpers_).RollbackConstrainedSpan(parser, _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]), generated, currentConstrainedOut)
                                d_8_rbGenerated_ = out4_
                                d_9_rbCurrent_ = out5_
                                generated = d_8_rbGenerated_
                                currentConstrainedOut = d_9_rbCurrent_
                                d_10_rolledComplete_: bool
                                d_10_rolledComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                                if (d_10_rolledComplete_) and ((d_1_steps_) < (maxSteps)):
                                    d_11_closedGenerated2_: _dafny.Seq
                                    d_12_closedInside2_: bool
                                    d_13_closedCurrent2_: _dafny.Seq
                                    out6_: _dafny.Seq
                                    out7_: bool
                                    out8_: _dafny.Seq
                                    out6_, out7_, out8_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_11_closedGenerated2_ = out6_
                                    d_12_closedInside2_ = out7_
                                    d_13_closedCurrent2_ = out8_
                                    generated = d_11_closedGenerated2_
                                    insideConstrainedOut = d_12_closedInside2_
                                    currentConstrainedOut = d_13_closedCurrent2_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    cost = d_1_steps_
                                elif True:
                                    insideConstrainedOut = False
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                    raise _dafny.Break("0")
                            elif True:
                                d_14_narrow_: bool
                                out9_: bool
                                out9_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 2)
                                d_14_narrow_ = out9_
                                if d_14_narrow_:
                                    (lm).GenerateLogits((prompt) + (generated))
                                    d_15_candidates_: _dafny.Seq
                                    out10_: _dafny.Seq
                                    out10_ = (d_0_helpers_).TopValidCandidates(lm, parser, prompt, currentConstrainedOut, 3, eosToken)
                                    d_15_candidates_ = out10_
                                    (d_0_helpers_).BoostTokenLogits(lm, d_15_candidates_, _dafny.BigRational('1e2'))
                                d_16_nextInside_: _dafny.Seq
                                out11_: _dafny.Seq
                                out11_ = (d_0_helpers_).ConstrainedStep(lm, parser, prompt, currentConstrainedOut, eosToken)
                                d_16_nextInside_ = out11_
                                d_1_steps_ = (d_1_steps_) + (1)
                                cost = d_1_steps_
                                if (d_16_nextInside_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_17_appendedGenerated_: _dafny.Seq
                                    d_18_appendedInside_: bool
                                    d_19_appendedCurrent_: _dafny.Seq
                                    out12_: _dafny.Seq
                                    out13_: bool
                                    out14_: _dafny.Seq
                                    out12_, out13_, out14_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_16_nextInside_)
                                    d_17_appendedGenerated_ = out12_
                                    d_18_appendedInside_ = out13_
                                    d_19_appendedCurrent_ = out14_
                                    generated = d_17_appendedGenerated_
                                    insideConstrainedOut = d_18_appendedInside_
                                    currentConstrainedOut = d_19_appendedCurrent_
                    elif True:
                        d_20_remaining_: int
                        d_20_remaining_ = (maxSteps) - (d_1_steps_)
                        (lm).GenerateLogits((prompt) + (generated))
                        if (d_20_remaining_) >= (3):
                            (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]), _dafny.BigRational('3e0'))
                        elif True:
                            (d_0_helpers_).PenalizeTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]), _dafny.BigRational('8e0'))
                        d_21_next_: _dafny.Seq
                        out15_: _dafny.Seq
                        out15_ = (lm).ChooseNextToken()
                        d_21_next_ = out15_
                        (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                        d_1_steps_ = (d_1_steps_) + (1)
                        cost = d_1_steps_
                        if (d_21_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif (d_21_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                            if (d_20_remaining_) >= (3):
                                d_22_openedGenerated_: _dafny.Seq
                                d_23_openedInside_: bool
                                d_24_openedCurrent_: _dafny.Seq
                                out16_: _dafny.Seq
                                out17_: bool
                                out18_: _dafny.Seq
                                out16_, out17_, out18_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                d_22_openedGenerated_ = out16_
                                d_23_openedInside_ = out17_
                                d_24_openedCurrent_ = out18_
                                generated = d_22_openedGenerated_
                                insideConstrainedOut = d_23_openedInside_
                                currentConstrainedOut = d_24_openedCurrent_
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_21_next_]))
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_21_next_]))
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

