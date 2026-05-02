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
        d_2_openedAny_: bool
        d_2_openedAny_ = (insideConstrained) or ((_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))) in (generatedPrefix))
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        (lm).GenerateLogits((prompt) + (generated))
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
                        d_6_chooseOpen_: bool
                        d_6_chooseOpen_ = False
                        if not(d_2_openedAny_):
                            if (d_5_openLogit_) >= ((d_4_argmaxLogit_) - (_dafny.BigRational('2e0'))):
                                d_6_chooseOpen_ = True
                            elif VerifiedDecoderAgent.default__.Contains(d_3_argmax_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                d_6_chooseOpen_ = True
                        d_7_next_: _dafny.Seq
                        d_7_next_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                        if d_6_chooseOpen_:
                            d_7_next_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))
                        elif True:
                            out3_: _dafny.Seq
                            out3_ = (lm).ChooseNextToken()
                            d_7_next_ = out3_
                            (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_7_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_7_next_]))
                            if (d_7_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                d_2_openedAny_ = True
                    elif True:
                        d_8_complete_: bool
                        d_8_complete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_8_complete_:
                            d_9_closedGenerated_: _dafny.Seq
                            d_10_closedInside_: bool
                            d_11_closedCurrent_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_9_closedGenerated_ = out4_
                            d_10_closedInside_ = out5_
                            d_11_closedCurrent_ = out6_
                            generated = d_9_closedGenerated_
                            insideConstrainedOut = d_10_closedInside_
                            currentConstrainedOut = d_11_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_12_constrainedPrompt_: _dafny.Seq
                            d_12_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_13_narrow_: bool
                            out7_: bool
                            out7_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 2)
                            d_13_narrow_ = out7_
                            if not(d_13_narrow_):
                                (lm).GenerateLogits((d_12_constrainedPrompt_) + (currentConstrainedOut))
                                d_14_argmaxInside_: _dafny.Seq
                                out8_: _dafny.Seq
                                out8_ = (d_0_helpers_).GetHighestLogitToken(lm)
                                d_14_argmaxInside_ = out8_
                                d_15_argmaxValid_: bool
                                out9_: bool
                                out9_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_14_argmaxInside_)
                                d_15_argmaxValid_ = out9_
                                d_16_nextInside_: _dafny.Seq
                                d_16_nextInside_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                                if d_15_argmaxValid_:
                                    d_16_nextInside_ = d_14_argmaxInside_
                                    (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    if (d_16_nextInside_) == (eosToken):
                                        raise _dafny.Break("0")
                                    elif True:
                                        d_17_appendedGenerated1_: _dafny.Seq
                                        d_18_appendedInside1_: bool
                                        d_19_appendedCurrent1_: _dafny.Seq
                                        out10_: _dafny.Seq
                                        out11_: bool
                                        out12_: _dafny.Seq
                                        out10_, out11_, out12_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_16_nextInside_)
                                        d_17_appendedGenerated1_ = out10_
                                        d_18_appendedInside1_ = out11_
                                        d_19_appendedCurrent1_ = out12_
                                        generated = d_17_appendedGenerated1_
                                        insideConstrainedOut = d_18_appendedInside1_
                                        currentConstrainedOut = d_19_appendedCurrent1_
                                elif True:
                                    d_20_sampled_: _dafny.Seq
                                    out13_: _dafny.Seq
                                    out13_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_12_constrainedPrompt_, currentConstrainedOut, eosToken)
                                    d_20_sampled_ = out13_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    if (d_20_sampled_) == (eosToken):
                                        raise _dafny.Break("0")
                                    elif True:
                                        d_21_appendedGenerated2_: _dafny.Seq
                                        d_22_appendedInside2_: bool
                                        d_23_appendedCurrent2_: _dafny.Seq
                                        out14_: _dafny.Seq
                                        out15_: bool
                                        out16_: _dafny.Seq
                                        out14_, out15_, out16_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_20_sampled_)
                                        d_21_appendedGenerated2_ = out14_
                                        d_22_appendedInside2_ = out15_
                                        d_23_appendedCurrent2_ = out16_
                                        generated = d_21_appendedGenerated2_
                                        insideConstrainedOut = d_22_appendedInside2_
                                        currentConstrainedOut = d_23_appendedCurrent2_
                            elif True:
                                d_24_repaired_: _dafny.Seq
                                out17_: _dafny.Seq
                                out17_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_12_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_24_repaired_ = out17_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_24_repaired_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_25_appendedGenerated3_: _dafny.Seq
                                    d_26_appendedInside3_: bool
                                    d_27_appendedCurrent3_: _dafny.Seq
                                    out18_: _dafny.Seq
                                    out19_: bool
                                    out20_: _dafny.Seq
                                    out18_, out19_, out20_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_24_repaired_)
                                    d_25_appendedGenerated3_ = out18_
                                    d_26_appendedInside3_ = out19_
                                    d_27_appendedCurrent3_ = out20_
                                    generated = d_25_appendedGenerated3_
                                    insideConstrainedOut = d_26_appendedInside3_
                                    currentConstrainedOut = d_27_appendedCurrent3_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

