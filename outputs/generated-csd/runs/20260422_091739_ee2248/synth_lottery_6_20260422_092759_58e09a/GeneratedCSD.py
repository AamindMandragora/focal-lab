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
                        if ((d_4_openLogit_) >= ((d_3_argmaxLogit_) - (_dafny.BigRational('2e0')))) and ((d_2_argmax_) != (eosToken)):
                            (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]), _dafny.BigRational('8e0'))
                            d_5_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (lm).ChooseNextTokenUnconstrained()
                            d_5_next_ = out3_
                            (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_5_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif (d_5_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_5_next_]))
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_5_next_]))
                        elif True:
                            d_6_next_: _dafny.Seq
                            out4_: _dafny.Seq
                            out4_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_6_next_ = out4_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_6_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_6_next_]))
                                if (d_6_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    insideConstrainedOut = True
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_7_closedGenerated_: _dafny.Seq
                        d_8_closedInside_: bool
                        d_9_closedCurrent_: _dafny.Seq
                        out5_: _dafny.Seq
                        out6_: bool
                        out7_: _dafny.Seq
                        out5_, out6_, out7_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_7_closedGenerated_ = out5_
                        d_8_closedInside_ = out6_
                        d_9_closedCurrent_ = out7_
                        generated = d_7_closedGenerated_
                        insideConstrainedOut = d_8_closedInside_
                        currentConstrainedOut = d_9_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_10_constrainedPrompt_: _dafny.Seq
                        d_10_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        (lm).GenerateLogits((d_10_constrainedPrompt_) + (currentConstrainedOut))
                        d_11_argmax_: _dafny.Seq
                        out8_: _dafny.Seq
                        out8_ = (d_0_helpers_).GetHighestLogitToken(lm)
                        d_11_argmax_ = out8_
                        d_12_argmaxValid_: bool
                        out9_: bool
                        out9_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_11_argmax_)
                        d_12_argmaxValid_ = out9_
                        if ((d_12_argmaxValid_) and ((d_11_argmax_) != (eosToken))) and (not(VerifiedDecoderAgent.default__.Contains(d_11_argmax_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))))):
                            (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([d_11_argmax_]), _dafny.BigRational('3e0'))
                            d_13_next_: _dafny.Seq
                            out10_: _dafny.Seq
                            out10_ = (lm).ChooseNextTokenUnconstrained()
                            d_13_next_ = out10_
                            (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_13_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_14_sampledValid_: bool
                                out11_: bool
                                out11_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_13_next_)
                                d_14_sampledValid_ = out11_
                                if d_14_sampledValid_:
                                    d_15_appendedGenerated_: _dafny.Seq
                                    d_16_appendedInside_: bool
                                    d_17_appendedCurrent_: _dafny.Seq
                                    out12_: _dafny.Seq
                                    out13_: bool
                                    out14_: _dafny.Seq
                                    out12_, out13_, out14_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_13_next_)
                                    d_15_appendedGenerated_ = out12_
                                    d_16_appendedInside_ = out13_
                                    d_17_appendedCurrent_ = out14_
                                    generated = d_15_appendedGenerated_
                                    insideConstrainedOut = d_16_appendedInside_
                                    currentConstrainedOut = d_17_appendedCurrent_
                        elif True:
                            d_18_next_: _dafny.Seq
                            out15_: _dafny.Seq
                            out15_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_10_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_18_next_ = out15_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_18_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_19_appendedGenerated_: _dafny.Seq
                                d_20_appendedInside_: bool
                                d_21_appendedCurrent_: _dafny.Seq
                                out16_: _dafny.Seq
                                out17_: bool
                                out18_: _dafny.Seq
                                out16_, out17_, out18_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_18_next_)
                                d_19_appendedGenerated_ = out16_
                                d_20_appendedInside_ = out17_
                                d_21_appendedCurrent_ = out18_
                                generated = d_19_appendedGenerated_
                                insideConstrainedOut = d_20_appendedInside_
                                currentConstrainedOut = d_21_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

