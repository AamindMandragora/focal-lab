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
                    if not(insideConstrainedOut):
                        (lm).GenerateLogits((prompt) + (generated))
                        (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]), _dafny.BigRational('1e2'))
                        d_2_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (lm).ChooseNextToken()
                        d_2_next_ = out0_
                        (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_2_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif (d_2_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                            d_3_openedGenerated_: _dafny.Seq
                            d_4_openedInside_: bool
                            d_5_openedCurrent_: _dafny.Seq
                            out1_: _dafny.Seq
                            out2_: bool
                            out3_: _dafny.Seq
                            out1_, out2_, out3_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_3_openedGenerated_ = out1_
                            d_4_openedInside_ = out2_
                            d_5_openedCurrent_ = out3_
                            generated = d_3_openedGenerated_
                            insideConstrainedOut = d_4_openedInside_
                            currentConstrainedOut = d_5_openedCurrent_
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_2_next_]))
                    elif True:
                        d_6_complete_: bool
                        d_6_complete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_6_complete_:
                            d_7_closedGenerated_: _dafny.Seq
                            d_8_closedInside_: bool
                            d_9_closedCurrent_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_7_closedGenerated_ = out4_
                            d_8_closedInside_ = out5_
                            d_9_closedCurrent_ = out6_
                            generated = d_7_closedGenerated_
                            insideConstrainedOut = d_8_closedInside_
                            currentConstrainedOut = d_9_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_10_narrow_: bool
                            out7_: bool
                            out7_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 3)
                            d_10_narrow_ = out7_
                            if d_10_narrow_:
                                (lm).GenerateLogits((prompt) + (generated))
                                d_11_candidates_: _dafny.Seq
                                out8_: _dafny.Seq
                                out8_ = (d_0_helpers_).TopValidCandidates(lm, parser, prompt, currentConstrainedOut, 5, eosToken)
                                d_11_candidates_ = out8_
                                (d_0_helpers_).BoostTokenLogits(lm, d_11_candidates_, _dafny.BigRational('8e0'))
                                d_12_sampled_: _dafny.Seq
                                out9_: _dafny.Seq
                                out9_ = (lm).ChooseNextToken()
                                d_12_sampled_ = out9_
                                (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_12_sampled_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_13_sampledValid_: bool
                                    out10_: bool
                                    out10_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_12_sampled_)
                                    d_13_sampledValid_ = out10_
                                    if d_13_sampledValid_:
                                        d_14_appendedGenerated1_: _dafny.Seq
                                        d_15_appendedInside1_: bool
                                        d_16_appendedCurrent1_: _dafny.Seq
                                        out11_: _dafny.Seq
                                        out12_: bool
                                        out13_: _dafny.Seq
                                        out11_, out12_, out13_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_12_sampled_)
                                        d_14_appendedGenerated1_ = out11_
                                        d_15_appendedInside1_ = out12_
                                        d_16_appendedCurrent1_ = out13_
                                        generated = d_14_appendedGenerated1_
                                        insideConstrainedOut = d_15_appendedInside1_
                                        currentConstrainedOut = d_16_appendedCurrent1_
                                    elif True:
                                        d_17_fallback_: _dafny.Seq
                                        out14_: _dafny.Seq
                                        out14_ = (d_0_helpers_).ConstrainedStep(lm, parser, prompt, currentConstrainedOut, eosToken)
                                        d_17_fallback_ = out14_
                                        if (d_17_fallback_) == (eosToken):
                                            raise _dafny.Break("0")
                                        elif True:
                                            d_18_appendedGenerated2_: _dafny.Seq
                                            d_19_appendedInside2_: bool
                                            d_20_appendedCurrent2_: _dafny.Seq
                                            out15_: _dafny.Seq
                                            out16_: bool
                                            out17_: _dafny.Seq
                                            out15_, out16_, out17_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_17_fallback_)
                                            d_18_appendedGenerated2_ = out15_
                                            d_19_appendedInside2_ = out16_
                                            d_20_appendedCurrent2_ = out17_
                                            generated = d_18_appendedGenerated2_
                                            insideConstrainedOut = d_19_appendedInside2_
                                            currentConstrainedOut = d_20_appendedCurrent2_
                            elif True:
                                d_21_nextInside_: _dafny.Seq
                                out18_: _dafny.Seq
                                out18_ = (d_0_helpers_).ConstrainedStep(lm, parser, prompt, currentConstrainedOut, eosToken)
                                d_21_nextInside_ = out18_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_21_nextInside_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_22_appendedGenerated3_: _dafny.Seq
                                    d_23_appendedInside3_: bool
                                    d_24_appendedCurrent3_: _dafny.Seq
                                    out19_: _dafny.Seq
                                    out20_: bool
                                    out21_: _dafny.Seq
                                    out19_, out20_, out21_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_21_nextInside_)
                                    d_22_appendedGenerated3_ = out19_
                                    d_23_appendedInside3_ = out20_
                                    d_24_appendedCurrent3_ = out21_
                                    generated = d_22_appendedGenerated3_
                                    insideConstrainedOut = d_23_appendedInside3_
                                    currentConstrainedOut = d_24_appendedCurrent3_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

