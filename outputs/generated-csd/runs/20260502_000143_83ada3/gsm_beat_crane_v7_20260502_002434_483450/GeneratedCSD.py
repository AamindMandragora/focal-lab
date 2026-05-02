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
        d_2_longSpanThreshold_: int
        d_2_longSpanThreshold_ = 10
        d_3_narrowThreshold_: int
        d_3_narrowThreshold_ = 6
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        (lm).GenerateLogits((prompt) + (generated))
                        if (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))) in ((lm).Tokens):
                            (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]), _dafny.BigRational('1e2'))
                        d_4_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (lm).ChooseNextTokenUnconstrained()
                        d_4_next_ = out0_
                        (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_4_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_4_next_]))
                            if (d_4_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        d_5_isComplete_: bool
                        d_5_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_5_isComplete_:
                            d_6_closedGenerated_: _dafny.Seq
                            d_7_closedInside_: bool
                            d_8_closedCurrent_: _dafny.Seq
                            out1_: _dafny.Seq
                            out2_: bool
                            out3_: _dafny.Seq
                            out1_, out2_, out3_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_6_closedGenerated_ = out1_
                            d_7_closedInside_ = out2_
                            d_8_closedCurrent_ = out3_
                            generated = d_6_closedGenerated_
                            insideConstrainedOut = d_7_closedInside_
                            currentConstrainedOut = d_8_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_9_stablePrefix_: _dafny.Seq
                            d_9_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_10_validCount_: int
                            out4_: int
                            out4_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                            d_10_validCount_ = out4_
                            if ((len(currentConstrainedOut)) >= (d_2_longSpanThreshold_)) or ((d_10_validCount_) <= (d_3_narrowThreshold_)):
                                d_11_constrainedGenerated_: _dafny.Seq
                                d_12_constrainedInside_: bool
                                d_13_constrainedCurrent_: _dafny.Seq
                                d_14_hitEos_: bool
                                out5_: _dafny.Seq
                                out6_: bool
                                out7_: _dafny.Seq
                                out8_: bool
                                out5_, out6_, out7_, out8_ = (d_0_helpers_).ConstrainedStep(lm, parser, (prompt) + (d_9_stablePrefix_), generated, currentConstrainedOut, eosToken)
                                d_11_constrainedGenerated_ = out5_
                                d_12_constrainedInside_ = out6_
                                d_13_constrainedCurrent_ = out7_
                                d_14_hitEos_ = out8_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if d_14_hitEos_:
                                    raise _dafny.Break("0")
                                elif True:
                                    generated = d_11_constrainedGenerated_
                                    insideConstrainedOut = d_12_constrainedInside_
                                    currentConstrainedOut = d_13_constrainedCurrent_
                            elif (len(validTokenGroups)) > (0):
                                d_15_next2_: _dafny.Seq
                                out9_: _dafny.Seq
                                out9_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, (prompt) + (d_9_stablePrefix_), currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), eosToken)
                                d_15_next2_ = out9_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_15_next2_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_16_appendedGenerated2_: _dafny.Seq
                                    d_17_appendedInside2_: bool
                                    d_18_appendedCurrent2_: _dafny.Seq
                                    out10_: _dafny.Seq
                                    out11_: bool
                                    out12_: _dafny.Seq
                                    out10_, out11_, out12_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_15_next2_)
                                    d_16_appendedGenerated2_ = out10_
                                    d_17_appendedInside2_ = out11_
                                    d_18_appendedCurrent2_ = out12_
                                    generated = d_16_appendedGenerated2_
                                    insideConstrainedOut = d_17_appendedInside2_
                                    currentConstrainedOut = d_18_appendedCurrent2_
                            elif True:
                                d_19_next3_: _dafny.Seq
                                out13_: _dafny.Seq
                                out13_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, (prompt) + (d_9_stablePrefix_), currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                                d_19_next3_ = out13_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_19_next3_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_20_appendedGenerated3_: _dafny.Seq
                                    d_21_appendedInside3_: bool
                                    d_22_appendedCurrent3_: _dafny.Seq
                                    out14_: _dafny.Seq
                                    out15_: bool
                                    out16_: _dafny.Seq
                                    out14_, out15_, out16_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_19_next3_)
                                    d_20_appendedGenerated3_ = out14_
                                    d_21_appendedInside3_ = out15_
                                    d_22_appendedCurrent3_ = out16_
                                    generated = d_20_appendedGenerated3_
                                    insideConstrainedOut = d_21_appendedInside3_
                                    currentConstrainedOut = d_22_appendedCurrent3_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

