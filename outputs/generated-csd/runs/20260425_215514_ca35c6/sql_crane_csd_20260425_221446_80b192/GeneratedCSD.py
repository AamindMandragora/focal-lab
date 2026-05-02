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
        d_2_openedCount_: int
        d_2_openedCount_ = 0
        d_3_closedCount_: int
        d_3_closedCount_ = 0
        if insideConstrainedOut:
            d_2_openedCount_ = 1
            d_3_closedCount_ = 0
        elif True:
            d_2_openedCount_ = 0
            d_3_closedCount_ = 0
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if (d_3_closedCount_) == (1):
                        raise _dafny.Break("0")
                    elif True:
                        if not(insideConstrainedOut):
                            d_4_openedGenerated_: _dafny.Seq
                            d_5_openedInside_: bool
                            d_6_openedCurrent_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_4_openedGenerated_ = out0_
                            d_5_openedInside_ = out1_
                            d_6_openedCurrent_ = out2_
                            generated = d_4_openedGenerated_
                            insideConstrainedOut = d_5_openedInside_
                            currentConstrainedOut = d_6_openedCurrent_
                            d_2_openedCount_ = 1
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_7_complete_: bool
                            d_7_complete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                            if d_7_complete_:
                                d_8_closedGenerated2_: _dafny.Seq
                                d_9_closedInside2_: bool
                                d_10_closedCurrent2_: _dafny.Seq
                                out3_: _dafny.Seq
                                out4_: bool
                                out5_: _dafny.Seq
                                out3_, out4_, out5_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_8_closedGenerated2_ = out3_
                                d_9_closedInside2_ = out4_
                                d_10_closedCurrent2_ = out5_
                                generated = d_8_closedGenerated2_
                                insideConstrainedOut = d_9_closedInside2_
                                currentConstrainedOut = d_10_closedCurrent2_
                                d_3_closedCount_ = 1
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                d_11_constrainedPrompt_: _dafny.Seq
                                d_11_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                (lm).GenerateLogits((d_11_constrainedPrompt_) + (currentConstrainedOut))
                                (d_0_helpers_).PenalizeTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]), _dafny.BigRational('1e2'))
                                (d_0_helpers_).PenalizeTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]), _dafny.BigRational('1e2'))
                                d_12_argmaxIn_: _dafny.Seq
                                out6_: _dafny.Seq
                                out6_ = (d_0_helpers_).GetHighestLogitToken(lm)
                                d_12_argmaxIn_ = out6_
                                d_13_argmaxValid_: bool
                                out7_: bool
                                out7_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_12_argmaxIn_)
                                d_13_argmaxValid_ = out7_
                                if (d_13_argmaxValid_) and ((d_12_argmaxIn_) != (eosToken)):
                                    d_14_sampled_: _dafny.Seq
                                    out8_: _dafny.Seq
                                    out8_ = (lm).ChooseNextToken()
                                    d_14_sampled_ = out8_
                                    (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    if (d_14_sampled_) == (eosToken):
                                        raise _dafny.Break("0")
                                    elif True:
                                        d_15_sampledValid_: bool
                                        out9_: bool
                                        out9_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_14_sampled_)
                                        d_15_sampledValid_ = out9_
                                        if d_15_sampledValid_:
                                            d_16_appendedGenerated_: _dafny.Seq
                                            d_17_appendedInside_: bool
                                            d_18_appendedCurrent_: _dafny.Seq
                                            out10_: _dafny.Seq
                                            out11_: bool
                                            out12_: _dafny.Seq
                                            out10_, out11_, out12_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_14_sampled_)
                                            d_16_appendedGenerated_ = out10_
                                            d_17_appendedInside_ = out11_
                                            d_18_appendedCurrent_ = out12_
                                            generated = d_16_appendedGenerated_
                                            insideConstrainedOut = d_17_appendedInside_
                                            currentConstrainedOut = d_18_appendedCurrent_
                                        elif True:
                                            if (d_1_steps_) < (maxSteps):
                                                d_19_nextFallback_: _dafny.Seq
                                                out13_: _dafny.Seq
                                                out13_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_11_constrainedPrompt_, currentConstrainedOut, eosToken)
                                                d_19_nextFallback_ = out13_
                                                d_1_steps_ = (d_1_steps_) + (1)
                                                if (d_19_nextFallback_) == (eosToken):
                                                    raise _dafny.Break("0")
                                                elif True:
                                                    d_20_appendedGeneratedFb_: _dafny.Seq
                                                    d_21_appendedInsideFb_: bool
                                                    d_22_appendedCurrentFb_: _dafny.Seq
                                                    out14_: _dafny.Seq
                                                    out15_: bool
                                                    out16_: _dafny.Seq
                                                    out14_, out15_, out16_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_19_nextFallback_)
                                                    d_20_appendedGeneratedFb_ = out14_
                                                    d_21_appendedInsideFb_ = out15_
                                                    d_22_appendedCurrentFb_ = out16_
                                                    generated = d_20_appendedGeneratedFb_
                                                    insideConstrainedOut = d_21_appendedInsideFb_
                                                    currentConstrainedOut = d_22_appendedCurrentFb_
                                            elif True:
                                                raise _dafny.Break("0")
                                elif True:
                                    d_23_nextIn_: _dafny.Seq
                                    out17_: _dafny.Seq
                                    out17_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_11_constrainedPrompt_, currentConstrainedOut, eosToken)
                                    d_23_nextIn_ = out17_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    if (d_23_nextIn_) == (eosToken):
                                        raise _dafny.Break("0")
                                    elif True:
                                        d_24_appendedGenerated2_: _dafny.Seq
                                        d_25_appendedInside2_: bool
                                        d_26_appendedCurrent2_: _dafny.Seq
                                        out18_: _dafny.Seq
                                        out19_: bool
                                        out20_: _dafny.Seq
                                        out18_, out19_, out20_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_23_nextIn_)
                                        d_24_appendedGenerated2_ = out18_
                                        d_25_appendedInside2_ = out19_
                                        d_26_appendedCurrent2_ = out20_
                                        generated = d_24_appendedGenerated2_
                                        insideConstrainedOut = d_25_appendedInside2_
                                        currentConstrainedOut = d_26_appendedCurrent2_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

