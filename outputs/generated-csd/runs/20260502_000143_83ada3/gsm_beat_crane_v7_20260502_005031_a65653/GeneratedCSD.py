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
        d_3_openWarmup_: int
        d_3_openWarmup_ = 10
        d_4_earlySpanThreshold_: int
        d_4_earlySpanThreshold_ = 1
        d_5_longSpanThreshold_: int
        d_5_longSpanThreshold_ = 6
        d_6_narrowThreshold_: int
        d_6_narrowThreshold_ = 8
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if (d_2_spansClosed_) > (0):
                            raise _dafny.Break("0")
                        elif True:
                            (lm).GenerateLogits((prompt) + (generated))
                            if ((d_3_openWarmup_) <= (d_1_steps_)) and ((_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))) in ((lm).Tokens)):
                                (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]), _dafny.BigRational('5e0'))
                            d_7_next_: _dafny.Seq
                            out0_: _dafny.Seq
                            out0_ = (lm).ChooseNextTokenUnconstrained()
                            d_7_next_ = out0_
                            (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_7_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_7_next_]))
                                if (d_7_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    insideConstrainedOut = True
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        d_8_isComplete_: bool
                        d_8_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_8_isComplete_:
                            d_9_closedGenerated_: _dafny.Seq
                            d_10_closedInside_: bool
                            d_11_closedCurrent_: _dafny.Seq
                            out1_: _dafny.Seq
                            out2_: bool
                            out3_: _dafny.Seq
                            out1_, out2_, out3_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_9_closedGenerated_ = out1_
                            d_10_closedInside_ = out2_
                            d_11_closedCurrent_ = out3_
                            generated = d_9_closedGenerated_
                            insideConstrainedOut = d_10_closedInside_
                            currentConstrainedOut = d_11_closedCurrent_
                            d_2_spansClosed_ = (d_2_spansClosed_) + (1)
                            d_1_steps_ = (d_1_steps_) + (1)
                            raise _dafny.Break("0")
                        elif True:
                            d_12_stablePrefix_: _dafny.Seq
                            d_12_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_13_validCount_: int
                            out4_: int
                            out4_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                            d_13_validCount_ = out4_
                            if ((len(currentConstrainedOut)) >= (d_5_longSpanThreshold_)) or ((d_13_validCount_) <= (d_6_narrowThreshold_)):
                                d_14_constrainedGenerated_: _dafny.Seq
                                d_15_constrainedInside_: bool
                                d_16_constrainedCurrent_: _dafny.Seq
                                d_17_hitEos_: bool
                                out5_: _dafny.Seq
                                out6_: bool
                                out7_: _dafny.Seq
                                out8_: bool
                                out5_, out6_, out7_, out8_ = (d_0_helpers_).ConstrainedStep(lm, parser, (prompt) + (d_12_stablePrefix_), generated, currentConstrainedOut, eosToken)
                                d_14_constrainedGenerated_ = out5_
                                d_15_constrainedInside_ = out6_
                                d_16_constrainedCurrent_ = out7_
                                d_17_hitEos_ = out8_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if d_17_hitEos_:
                                    raise _dafny.Break("0")
                                elif True:
                                    generated = d_14_constrainedGenerated_
                                    insideConstrainedOut = d_15_constrainedInside_
                                    currentConstrainedOut = d_16_constrainedCurrent_
                            elif True:
                                d_18_useGroups_: bool
                                d_18_useGroups_ = False
                                if ((len(currentConstrainedOut)) <= (d_4_earlySpanThreshold_)) and ((len(validTokenGroups)) > (0)):
                                    out9_: bool
                                    out9_ = (d_0_helpers_).GroupHasValidMember(parser, currentConstrainedOut, (validTokenGroups)[0])
                                    d_18_useGroups_ = out9_
                                if d_18_useGroups_:
                                    d_19_next2_: _dafny.Seq
                                    out10_: _dafny.Seq
                                    out10_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, (prompt) + (d_12_stablePrefix_), currentConstrainedOut, validTokenGroups, _dafny.BigRational('15e-1'), eosToken)
                                    d_19_next2_ = out10_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    if (d_19_next2_) == (eosToken):
                                        raise _dafny.Break("0")
                                    elif True:
                                        d_20_appendedGenerated2_: _dafny.Seq
                                        d_21_appendedInside2_: bool
                                        d_22_appendedCurrent2_: _dafny.Seq
                                        out11_: _dafny.Seq
                                        out12_: bool
                                        out13_: _dafny.Seq
                                        out11_, out12_, out13_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_19_next2_)
                                        d_20_appendedGenerated2_ = out11_
                                        d_21_appendedInside2_ = out12_
                                        d_22_appendedCurrent2_ = out13_
                                        generated = d_20_appendedGenerated2_
                                        insideConstrainedOut = d_21_appendedInside2_
                                        currentConstrainedOut = d_22_appendedCurrent2_
                                elif True:
                                    d_23_constrainedGenerated2_: _dafny.Seq
                                    d_24_constrainedInside2_: bool
                                    d_25_constrainedCurrent2_: _dafny.Seq
                                    d_26_hitEos2_: bool
                                    out14_: _dafny.Seq
                                    out15_: bool
                                    out16_: _dafny.Seq
                                    out17_: bool
                                    out14_, out15_, out16_, out17_ = (d_0_helpers_).ConstrainedStep(lm, parser, (prompt) + (d_12_stablePrefix_), generated, currentConstrainedOut, eosToken)
                                    d_23_constrainedGenerated2_ = out14_
                                    d_24_constrainedInside2_ = out15_
                                    d_25_constrainedCurrent2_ = out16_
                                    d_26_hitEos2_ = out17_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    if d_26_hitEos2_:
                                        raise _dafny.Break("0")
                                    elif True:
                                        generated = d_23_constrainedGenerated2_
                                        insideConstrainedOut = d_24_constrainedInside2_
                                        currentConstrainedOut = d_25_constrainedCurrent2_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

