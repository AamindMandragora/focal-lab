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
        if insideConstrainedOut:
            d_2_openedCount_ = 1
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        (lm).GenerateLogits((prompt) + (generated))
                        if (d_2_openedCount_) == (0):
                            d_3_argmax_: _dafny.Seq
                            out0_: _dafny.Seq
                            out0_ = (d_0_helpers_).GetHighestLogitToken(lm)
                            d_3_argmax_ = out0_
                            d_4_argmaxLogit_: _dafny.BigRational
                            out1_: _dafny.BigRational
                            out1_ = (d_0_helpers_).GetTokenLogit(lm, d_3_argmax_)
                            d_4_argmaxLogit_ = out1_
                            d_5_openLogit_: _dafny.BigRational
                            out2_: _dafny.BigRational
                            out2_ = (d_0_helpers_).GetTokenLogit(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))
                            d_5_openLogit_ = out2_
                            if (((d_5_openLogit_) >= ((d_4_argmaxLogit_) - (_dafny.BigRational('2e0')))) or (VerifiedDecoderAgent.default__.Contains(d_3_argmax_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SELECT"))))) or (VerifiedDecoderAgent.default__.Contains(d_3_argmax_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "FROM")))):
                                d_6_openedGenerated_: _dafny.Seq
                                d_7_openedInside_: bool
                                d_8_openedCurrent_: _dafny.Seq
                                out3_: _dafny.Seq
                                out4_: bool
                                out5_: _dafny.Seq
                                out3_, out4_, out5_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                d_6_openedGenerated_ = out3_
                                d_7_openedInside_ = out4_
                                d_8_openedCurrent_ = out5_
                                generated = d_6_openedGenerated_
                                insideConstrainedOut = d_7_openedInside_
                                currentConstrainedOut = d_8_openedCurrent_
                                d_2_openedCount_ = 1
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                d_9_next_: _dafny.Seq
                                out6_: _dafny.Seq
                                out6_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                                d_9_next_ = out6_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_9_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_9_next_]))
                                    if (d_9_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                        insideConstrainedOut = True
                                        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                        d_2_openedCount_ = 1
                        elif True:
                            (d_0_helpers_).PenalizeTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]), _dafny.BigRational('1e2'))
                            d_10_next_: _dafny.Seq
                            out7_: _dafny.Seq
                            out7_ = (lm).ChooseNextTokenUnconstrained()
                            d_10_next_ = out7_
                            (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_10_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_10_next_]))
                    elif True:
                        d_11_complete_: bool
                        d_11_complete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_11_complete_:
                            d_12_closedGenerated_: _dafny.Seq
                            d_13_closedInside_: bool
                            d_14_closedCurrent_: _dafny.Seq
                            out8_: _dafny.Seq
                            out9_: bool
                            out10_: _dafny.Seq
                            out8_, out9_, out10_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_12_closedGenerated_ = out8_
                            d_13_closedInside_ = out9_
                            d_14_closedCurrent_ = out10_
                            generated = d_12_closedGenerated_
                            insideConstrainedOut = d_13_closedInside_
                            currentConstrainedOut = d_14_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_15_constrainedPrompt_: _dafny.Seq
                            d_15_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            (lm).GenerateLogits((d_15_constrainedPrompt_) + (currentConstrainedOut))
                            (d_0_helpers_).PenalizeTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]), _dafny.BigRational('1e2'))
                            d_16_argmaxIn_: _dafny.Seq
                            out11_: _dafny.Seq
                            out11_ = (d_0_helpers_).GetHighestLogitToken(lm)
                            d_16_argmaxIn_ = out11_
                            d_17_argmaxValid_: bool
                            out12_: bool
                            out12_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_16_argmaxIn_)
                            d_17_argmaxValid_ = out12_
                            if (d_17_argmaxValid_) and ((d_16_argmaxIn_) != (eosToken)):
                                d_18_sampled_: _dafny.Seq
                                out13_: _dafny.Seq
                                out13_ = (lm).ChooseNextToken()
                                d_18_sampled_ = out13_
                                (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_18_sampled_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_19_sampledValid_: bool
                                    out14_: bool
                                    out14_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_18_sampled_)
                                    d_19_sampledValid_ = out14_
                                    if d_19_sampledValid_:
                                        d_20_appendedGenerated_: _dafny.Seq
                                        d_21_appendedInside_: bool
                                        d_22_appendedCurrent_: _dafny.Seq
                                        out15_: _dafny.Seq
                                        out16_: bool
                                        out17_: _dafny.Seq
                                        out15_, out16_, out17_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_18_sampled_)
                                        d_20_appendedGenerated_ = out15_
                                        d_21_appendedInside_ = out16_
                                        d_22_appendedCurrent_ = out17_
                                        generated = d_20_appendedGenerated_
                                        insideConstrainedOut = d_21_appendedInside_
                                        currentConstrainedOut = d_22_appendedCurrent_
                            elif True:
                                d_23_nextIn_: _dafny.Seq
                                out18_: _dafny.Seq
                                out18_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_15_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_23_nextIn_ = out18_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_23_nextIn_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_24_appendedGenerated2_: _dafny.Seq
                                    d_25_appendedInside2_: bool
                                    d_26_appendedCurrent2_: _dafny.Seq
                                    out19_: _dafny.Seq
                                    out20_: bool
                                    out21_: _dafny.Seq
                                    out19_, out20_, out21_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_23_nextIn_)
                                    d_24_appendedGenerated2_ = out19_
                                    d_25_appendedInside2_ = out20_
                                    d_26_appendedCurrent2_ = out21_
                                    generated = d_24_appendedGenerated2_
                                    insideConstrainedOut = d_25_appendedInside2_
                                    currentConstrainedOut = d_26_appendedCurrent2_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

