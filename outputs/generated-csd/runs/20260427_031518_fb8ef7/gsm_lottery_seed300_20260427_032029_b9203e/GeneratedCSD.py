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
    def MyCSDStrategy(lm, parser, prompt, generatedPrefix, insideConstrained, currentConstrained, maxSteps, stepTokenBudget, eosToken):
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
                        d_2_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_2_next_ = out0_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_2_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_2_next_]))
                            if (d_2_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        d_3_complete_: bool
                        d_3_complete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_3_complete_:
                            d_4_closedGenerated_: _dafny.Seq
                            d_5_closedInside_: bool
                            d_6_closedCurrent_: _dafny.Seq
                            out1_: _dafny.Seq
                            out2_: bool
                            out3_: _dafny.Seq
                            out1_, out2_, out3_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_4_closedGenerated_ = out1_
                            d_5_closedInside_ = out2_
                            d_6_closedCurrent_ = out3_
                            generated = d_4_closedGenerated_
                            insideConstrainedOut = d_5_closedInside_
                            currentConstrainedOut = d_6_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_7_dead_: bool
                            out4_: bool
                            out4_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                            d_7_dead_ = out4_
                            if d_7_dead_:
                                d_8_gLen_: int
                                d_8_gLen_ = len(generated)
                                d_9_cLen_: int
                                d_9_cLen_ = len(currentConstrainedOut)
                                d_10_stablePrefix_: _dafny.Seq
                                d_10_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(d_8_gLen_) - (d_9_cLen_):])
                                d_11_rolled_: _dafny.Seq
                                out5_: _dafny.Seq
                                out5_ = VerifiedDecoderAgent.CSDHelpers.RollbackToValidPrefix(parser, currentConstrainedOut)
                                d_11_rolled_ = out5_
                                d_12_rolledGenerated_: _dafny.Seq
                                d_13_rolledCurrent_: _dafny.Seq
                                out6_: _dafny.Seq
                                out7_: _dafny.Seq
                                out6_, out7_ = (d_0_helpers_).RollbackConstrainedSpan(parser, d_10_stablePrefix_, generated, currentConstrainedOut)
                                d_12_rolledGenerated_ = out6_
                                d_13_rolledCurrent_ = out7_
                                generated = d_12_rolledGenerated_
                                currentConstrainedOut = d_13_rolledCurrent_
                                d_14_rolledComplete_: bool
                                d_14_rolledComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                                if d_14_rolledComplete_:
                                    d_15_closedGenerated2_: _dafny.Seq
                                    d_16_closedInside2_: bool
                                    d_17_closedCurrent2_: _dafny.Seq
                                    out8_: _dafny.Seq
                                    out9_: bool
                                    out10_: _dafny.Seq
                                    out8_, out9_, out10_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_15_closedGenerated2_ = out8_
                                    d_16_closedInside2_ = out9_
                                    d_17_closedCurrent2_ = out10_
                                    generated = d_15_closedGenerated2_
                                    insideConstrainedOut = d_16_closedInside2_
                                    currentConstrainedOut = d_17_closedCurrent2_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                d_18_candidates_: _dafny.Seq
                                out11_: _dafny.Seq
                                out11_ = (d_0_helpers_).TopValidCandidates(lm, parser, prompt, currentConstrainedOut, 3, eosToken)
                                d_18_candidates_ = out11_
                                if (len(d_18_candidates_)) == (0):
                                    d_19_gLen2_: int
                                    d_19_gLen2_ = len(generated)
                                    d_20_cLen2_: int
                                    d_20_cLen2_ = len(currentConstrainedOut)
                                    d_21_stablePrefix2_: _dafny.Seq
                                    d_21_stablePrefix2_ = _dafny.SeqWithoutIsStrInference((generated)[:(d_19_gLen2_) - (d_20_cLen2_):])
                                    d_22_rolledGenerated2_: _dafny.Seq
                                    d_23_rolledCurrent2_: _dafny.Seq
                                    out12_: _dafny.Seq
                                    out13_: _dafny.Seq
                                    out12_, out13_ = (d_0_helpers_).RollbackConstrainedSpan(parser, d_21_stablePrefix2_, generated, currentConstrainedOut)
                                    d_22_rolledGenerated2_ = out12_
                                    d_23_rolledCurrent2_ = out13_
                                    generated = d_22_rolledGenerated2_
                                    currentConstrainedOut = d_23_rolledCurrent2_
                                    d_24_rolledComplete2_: bool
                                    d_24_rolledComplete2_ = (parser).IsCompletePrefix(currentConstrainedOut)
                                    if d_24_rolledComplete2_:
                                        d_25_closedGenerated3_: _dafny.Seq
                                        d_26_closedInside3_: bool
                                        d_27_closedCurrent3_: _dafny.Seq
                                        out14_: _dafny.Seq
                                        out15_: bool
                                        out16_: _dafny.Seq
                                        out14_, out15_, out16_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                        d_25_closedGenerated3_ = out14_
                                        d_26_closedInside3_ = out15_
                                        d_27_closedCurrent3_ = out16_
                                        generated = d_25_closedGenerated3_
                                        insideConstrainedOut = d_26_closedInside3_
                                        currentConstrainedOut = d_27_closedCurrent3_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                elif True:
                                    d_28_next_: _dafny.Seq
                                    d_28_next_ = (d_18_candidates_)[0]
                                    d_29_nextValid_: bool
                                    out17_: bool
                                    out17_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_28_next_)
                                    d_29_nextValid_ = out17_
                                    if d_29_nextValid_:
                                        d_30_appendedGenerated_: _dafny.Seq
                                        d_31_appendedInside_: bool
                                        d_32_appendedCurrent_: _dafny.Seq
                                        out18_: _dafny.Seq
                                        out19_: bool
                                        out20_: _dafny.Seq
                                        out18_, out19_, out20_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_28_next_)
                                        d_30_appendedGenerated_ = out18_
                                        d_31_appendedInside_ = out19_
                                        d_32_appendedCurrent_ = out20_
                                        generated = d_30_appendedGenerated_
                                        insideConstrainedOut = d_31_appendedInside_
                                        currentConstrainedOut = d_32_appendedCurrent_
                                        d_1_steps_ = (d_1_steps_) + (1)
                                        if (d_28_next_) == (eosToken):
                                            raise _dafny.Break("0")
                                    elif True:
                                        d_33_gLen3_: int
                                        d_33_gLen3_ = len(generated)
                                        d_34_cLen3_: int
                                        d_34_cLen3_ = len(currentConstrainedOut)
                                        d_35_stablePrefix3_: _dafny.Seq
                                        d_35_stablePrefix3_ = _dafny.SeqWithoutIsStrInference((generated)[:(d_33_gLen3_) - (d_34_cLen3_):])
                                        d_36_rolledGenerated3_: _dafny.Seq
                                        d_37_rolledCurrent3_: _dafny.Seq
                                        out21_: _dafny.Seq
                                        out22_: _dafny.Seq
                                        out21_, out22_ = (d_0_helpers_).RollbackConstrainedSpan(parser, d_35_stablePrefix3_, generated, currentConstrainedOut)
                                        d_36_rolledGenerated3_ = out21_
                                        d_37_rolledCurrent3_ = out22_
                                        generated = d_36_rolledGenerated3_
                                        currentConstrainedOut = d_37_rolledCurrent3_
                                        d_38_rolledComplete3_: bool
                                        d_38_rolledComplete3_ = (parser).IsCompletePrefix(currentConstrainedOut)
                                        if d_38_rolledComplete3_:
                                            d_39_closedGenerated4_: _dafny.Seq
                                            d_40_closedInside4_: bool
                                            d_41_closedCurrent4_: _dafny.Seq
                                            out23_: _dafny.Seq
                                            out24_: bool
                                            out25_: _dafny.Seq
                                            out23_, out24_, out25_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                            d_39_closedGenerated4_ = out23_
                                            d_40_closedInside4_ = out24_
                                            d_41_closedCurrent4_ = out25_
                                            generated = d_39_closedGenerated4_
                                            insideConstrainedOut = d_40_closedInside4_
                                            currentConstrainedOut = d_41_closedCurrent4_
                                        d_1_steps_ = (d_1_steps_) + (1)
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

