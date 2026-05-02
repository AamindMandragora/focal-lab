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
                        d_5_shouldBiasOpen_: bool
                        d_5_shouldBiasOpen_ = False
                        if (len(generated)) > (0):
                            d_6_lastTok_: _dafny.Seq
                            d_6_lastTok_ = (generated)[(len(generated)) - (1)]
                            if ((((((((((VerifiedDecoderAgent.default__.Contains(d_6_lastTok_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "0")))) or (VerifiedDecoderAgent.default__.Contains(d_6_lastTok_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "1"))))) or (VerifiedDecoderAgent.default__.Contains(d_6_lastTok_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "2"))))) or (VerifiedDecoderAgent.default__.Contains(d_6_lastTok_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "3"))))) or (VerifiedDecoderAgent.default__.Contains(d_6_lastTok_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "4"))))) or (VerifiedDecoderAgent.default__.Contains(d_6_lastTok_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "5"))))) or (VerifiedDecoderAgent.default__.Contains(d_6_lastTok_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "6"))))) or (VerifiedDecoderAgent.default__.Contains(d_6_lastTok_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "7"))))) or (VerifiedDecoderAgent.default__.Contains(d_6_lastTok_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "8"))))) or (VerifiedDecoderAgent.default__.Contains(d_6_lastTok_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "9"))))) or (VerifiedDecoderAgent.default__.Contains(d_6_lastTok_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")))):
                                d_5_shouldBiasOpen_ = True
                        if ((d_5_shouldBiasOpen_) and ((d_4_openLogit_) >= ((d_3_argmaxLogit_) - (_dafny.BigRational('2e0'))))) and ((d_2_argmax_) != (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))):
                            (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]), _dafny.BigRational('3e0'))
                        d_7_next_: _dafny.Seq
                        out3_: _dafny.Seq
                        out3_ = (lm).ChooseNextTokenUnconstrained()
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
                    elif True:
                        d_8_isComplete_: bool
                        d_8_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_8_isComplete_:
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
                            (lm).GenerateLogits((d_12_constrainedPrompt_) + (currentConstrainedOut))
                            d_13_argmax_: _dafny.Seq
                            out7_: _dafny.Seq
                            out7_ = (d_0_helpers_).GetHighestLogitToken(lm)
                            d_13_argmax_ = out7_
                            d_14_argmaxValid_: bool
                            d_14_argmaxValid_ = False
                            if (d_13_argmax_) != (eosToken):
                                out8_: bool
                                out8_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_13_argmax_)
                                d_14_argmaxValid_ = out8_
                            if d_14_argmaxValid_:
                                d_15_appendedGenerated_: _dafny.Seq
                                d_16_appendedInside_: bool
                                d_17_appendedCurrent_: _dafny.Seq
                                out9_: _dafny.Seq
                                out10_: bool
                                out11_: _dafny.Seq
                                out9_, out10_, out11_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_13_argmax_)
                                d_15_appendedGenerated_ = out9_
                                d_16_appendedInside_ = out10_
                                d_17_appendedCurrent_ = out11_
                                generated = d_15_appendedGenerated_
                                insideConstrainedOut = d_16_appendedInside_
                                currentConstrainedOut = d_17_appendedCurrent_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                d_18_next_: _dafny.Seq
                                out12_: _dafny.Seq
                                out12_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_12_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_18_next_ = out12_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_18_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_19_appendedGenerated2_: _dafny.Seq
                                    d_20_appendedInside2_: bool
                                    d_21_appendedCurrent2_: _dafny.Seq
                                    out13_: _dafny.Seq
                                    out14_: bool
                                    out15_: _dafny.Seq
                                    out13_, out14_, out15_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_18_next_)
                                    d_19_appendedGenerated2_ = out13_
                                    d_20_appendedInside2_ = out14_
                                    d_21_appendedCurrent2_ = out15_
                                    generated = d_19_appendedGenerated2_
                                    insideConstrainedOut = d_20_appendedInside2_
                                    currentConstrainedOut = d_21_appendedCurrent2_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

