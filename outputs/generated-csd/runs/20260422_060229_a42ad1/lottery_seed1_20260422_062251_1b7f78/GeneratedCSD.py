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
        (d_0_helpers_).cost = 0
        d_1_steps_: int
        d_1_steps_ = 0
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if insideConstrainedOut:
                        d_2_complete_: bool
                        d_2_complete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_2_complete_:
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
                        elif True:
                            d_6_constrainedPrompt_: _dafny.Seq
                            d_6_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            (lm).GenerateLogits((d_6_constrainedPrompt_) + (currentConstrainedOut))
                            d_7_argmax_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).GetHighestLogitToken(lm)
                            d_7_argmax_ = out3_
                            d_8_argmaxValid_: bool
                            out4_: bool
                            out4_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_7_argmax_)
                            d_8_argmaxValid_ = out4_
                            if (d_8_argmaxValid_) and ((d_7_argmax_) != (eosToken)):
                                d_9_appendedGenerated_: _dafny.Seq
                                d_10_appendedInside_: bool
                                d_11_appendedCurrent_: _dafny.Seq
                                out5_: _dafny.Seq
                                out6_: bool
                                out7_: _dafny.Seq
                                out5_, out6_, out7_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_7_argmax_)
                                d_9_appendedGenerated_ = out5_
                                d_10_appendedInside_ = out6_
                                d_11_appendedCurrent_ = out7_
                                generated = d_9_appendedGenerated_
                                insideConstrainedOut = d_10_appendedInside_
                                currentConstrainedOut = d_11_appendedCurrent_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                d_12_next_: _dafny.Seq
                                out8_: _dafny.Seq
                                out8_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_6_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_12_next_ = out8_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_12_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_13_appendedGenerated2_: _dafny.Seq
                                    d_14_appendedInside2_: bool
                                    d_15_appendedCurrent2_: _dafny.Seq
                                    out9_: _dafny.Seq
                                    out10_: bool
                                    out11_: _dafny.Seq
                                    out9_, out10_, out11_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_12_next_)
                                    d_13_appendedGenerated2_ = out9_
                                    d_14_appendedInside2_ = out10_
                                    d_15_appendedCurrent2_ = out11_
                                    generated = d_13_appendedGenerated2_
                                    insideConstrainedOut = d_14_appendedInside2_
                                    currentConstrainedOut = d_15_appendedCurrent2_
                    elif True:
                        (lm).GenerateLogits((prompt) + (generated))
                        d_16_top_: _dafny.Seq
                        out12_: _dafny.Seq
                        out12_ = (d_0_helpers_).GetHighestLogitToken(lm)
                        d_16_top_ = out12_
                        d_17_openLogit_: _dafny.BigRational
                        out13_: _dafny.BigRational
                        out13_ = (d_0_helpers_).GetTokenLogit(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))
                        d_17_openLogit_ = out13_
                        d_18_topLogit_: _dafny.BigRational
                        out14_: _dafny.BigRational
                        out14_ = (d_0_helpers_).GetTokenLogit(lm, d_16_top_)
                        d_18_topLogit_ = out14_
                        if (((d_16_top_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))) or (VerifiedDecoderAgent.default__.Contains(d_16_top_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))))) or ((d_17_openLogit_) >= ((d_18_topLogit_) - (_dafny.BigRational('5e-1')))):
                            d_19_openedGenerated_: _dafny.Seq
                            d_20_openedInside_: bool
                            d_21_openedCurrent_: _dafny.Seq
                            out15_: _dafny.Seq
                            out16_: bool
                            out17_: _dafny.Seq
                            out15_, out16_, out17_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_19_openedGenerated_ = out15_
                            d_20_openedInside_ = out16_
                            d_21_openedCurrent_ = out17_
                            generated = d_19_openedGenerated_
                            insideConstrainedOut = d_20_openedInside_
                            currentConstrainedOut = d_21_openedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_22_next_: _dafny.Seq
                            out18_: _dafny.Seq
                            out18_ = (lm).ChooseNextTokenUnconstrained()
                            d_22_next_ = out18_
                            (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_22_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif ((d_22_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))) or (VerifiedDecoderAgent.default__.Contains(d_22_next_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))):
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]))
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_22_next_]))
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

