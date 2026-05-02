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
                        d_2_argmax_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).GetHighestLogitToken(lm)
                        d_2_argmax_ = out0_
                        d_3_argmaxLogit_: _dafny.BigRational
                        out1_: _dafny.BigRational
                        out1_ = (d_0_helpers_).GetTokenLogit(lm, d_2_argmax_)
                        d_3_argmaxLogit_ = out1_
                        d_4_openLogit_: _dafny.BigRational
                        out2_: _dafny.BigRational
                        out2_ = (d_0_helpers_).GetTokenLogit(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))
                        d_4_openLogit_ = out2_
                        d_5_next_: _dafny.Seq
                        out3_: _dafny.Seq
                        out3_ = (lm).ChooseNextToken()
                        d_5_next_ = out3_
                        (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_5_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif (((d_4_openLogit_) >= ((d_3_argmaxLogit_) - (_dafny.BigRational('2e0')))) and ((d_5_next_) != (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))))) and (not(VerifiedDecoderAgent.default__.Contains(d_5_next_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))))):
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]))
                            insideConstrainedOut = True
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_5_next_]))
                            if ((d_5_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))) or (VerifiedDecoderAgent.default__.Contains(d_5_next_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))):
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            d_6_closedGenerated_: _dafny.Seq
                            d_7_closedInside_: bool
                            d_8_closedCurrent_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_6_closedGenerated_ = out4_
                            d_7_closedInside_ = out5_
                            d_8_closedCurrent_ = out6_
                            generated = d_6_closedGenerated_
                            insideConstrainedOut = d_7_closedInside_
                            currentConstrainedOut = d_8_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_9_constrainedPrompt_: _dafny.Seq
                            d_9_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_10_narrow_: bool
                            out7_: bool
                            out7_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 2)
                            d_10_narrow_ = out7_
                            if d_10_narrow_:
                                d_11_next2_: _dafny.Seq
                                out8_: _dafny.Seq
                                out8_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_9_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_11_next2_ = out8_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_11_next2_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_12_appendedGenerated2_: _dafny.Seq
                                    d_13_appendedInside2_: bool
                                    d_14_appendedCurrent2_: _dafny.Seq
                                    out9_: _dafny.Seq
                                    out10_: bool
                                    out11_: _dafny.Seq
                                    out9_, out10_, out11_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_11_next2_)
                                    d_12_appendedGenerated2_ = out9_
                                    d_13_appendedInside2_ = out10_
                                    d_14_appendedCurrent2_ = out11_
                                    generated = d_12_appendedGenerated2_
                                    insideConstrainedOut = d_13_appendedInside2_
                                    currentConstrainedOut = d_14_appendedCurrent2_
                            elif True:
                                (lm).GenerateLogits((d_9_constrainedPrompt_) + (currentConstrainedOut))
                                d_15_argmax2_: _dafny.Seq
                                out12_: _dafny.Seq
                                out12_ = (d_0_helpers_).GetHighestLogitToken(lm)
                                d_15_argmax2_ = out12_
                                d_16_valid2_: bool
                                out13_: bool
                                out13_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_15_argmax2_)
                                d_16_valid2_ = out13_
                                if (d_16_valid2_) and ((d_15_argmax2_) != (eosToken)):
                                    d_17_sampled_: _dafny.Seq
                                    out14_: _dafny.Seq
                                    out14_ = (lm).ChooseNextToken()
                                    d_17_sampled_ = out14_
                                    (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    d_18_sampledValid_: bool
                                    out15_: bool
                                    out15_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_17_sampled_)
                                    d_18_sampledValid_ = out15_
                                    if (d_17_sampled_) == (eosToken):
                                        raise _dafny.Break("0")
                                    elif d_18_sampledValid_:
                                        d_19_appendedGenerated_: _dafny.Seq
                                        d_20_appendedInside_: bool
                                        d_21_appendedCurrent_: _dafny.Seq
                                        out16_: _dafny.Seq
                                        out17_: bool
                                        out18_: _dafny.Seq
                                        out16_, out17_, out18_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_17_sampled_)
                                        d_19_appendedGenerated_ = out16_
                                        d_20_appendedInside_ = out17_
                                        d_21_appendedCurrent_ = out18_
                                        generated = d_19_appendedGenerated_
                                        insideConstrainedOut = d_20_appendedInside_
                                        currentConstrainedOut = d_21_appendedCurrent_
                                    elif True:
                                        d_22_appendedGenerated3_: _dafny.Seq
                                        d_23_appendedInside3_: bool
                                        d_24_appendedCurrent3_: _dafny.Seq
                                        out19_: _dafny.Seq
                                        out20_: bool
                                        out21_: _dafny.Seq
                                        out19_, out20_, out21_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_15_argmax2_)
                                        d_22_appendedGenerated3_ = out19_
                                        d_23_appendedInside3_ = out20_
                                        d_24_appendedCurrent3_ = out21_
                                        generated = d_22_appendedGenerated3_
                                        insideConstrainedOut = d_23_appendedInside3_
                                        currentConstrainedOut = d_24_appendedCurrent3_
                                elif True:
                                    d_25_next3_: _dafny.Seq
                                    out22_: _dafny.Seq
                                    out22_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_9_constrainedPrompt_, currentConstrainedOut, eosToken)
                                    d_25_next3_ = out22_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    if (d_25_next3_) == (eosToken):
                                        raise _dafny.Break("0")
                                    elif True:
                                        d_26_appendedGenerated4_: _dafny.Seq
                                        d_27_appendedInside4_: bool
                                        d_28_appendedCurrent4_: _dafny.Seq
                                        out23_: _dafny.Seq
                                        out24_: bool
                                        out25_: _dafny.Seq
                                        out23_, out24_, out25_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_25_next3_)
                                        d_26_appendedGenerated4_ = out23_
                                        d_27_appendedInside4_ = out24_
                                        d_28_appendedCurrent4_ = out25_
                                        generated = d_26_appendedGenerated4_
                                        insideConstrainedOut = d_27_appendedInside4_
                                        currentConstrainedOut = d_28_appendedCurrent4_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

