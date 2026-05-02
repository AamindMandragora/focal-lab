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
                        if (((d_4_openLogit_) >= ((d_3_argmaxLogit_) - (_dafny.BigRational('2e0')))) and ((d_2_argmax_) != (eosToken))) and ((d_2_argmax_) != (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")))):
                            d_5_openedGenerated_: _dafny.Seq
                            d_6_openedInside_: bool
                            d_7_openedCurrent_: _dafny.Seq
                            out3_: _dafny.Seq
                            out4_: bool
                            out5_: _dafny.Seq
                            out3_, out4_, out5_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_5_openedGenerated_ = out3_
                            d_6_openedInside_ = out4_
                            d_7_openedCurrent_ = out5_
                            generated = d_5_openedGenerated_
                            insideConstrainedOut = d_6_openedInside_
                            currentConstrainedOut = d_7_openedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_8_next_: _dafny.Seq
                            out6_: _dafny.Seq
                            out6_ = (lm).ChooseNextTokenUnconstrained()
                            d_8_next_ = out6_
                            (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_8_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_8_next_]))
                    elif True:
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            d_9_closedGenerated_: _dafny.Seq
                            d_10_closedInside_: bool
                            d_11_closedCurrent_: _dafny.Seq
                            out7_: _dafny.Seq
                            out8_: bool
                            out9_: _dafny.Seq
                            out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_9_closedGenerated_ = out7_
                            d_10_closedInside_ = out8_
                            d_11_closedCurrent_ = out9_
                            generated = d_9_closedGenerated_
                            insideConstrainedOut = d_10_closedInside_
                            currentConstrainedOut = d_11_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_12_constrainedPrompt_: _dafny.Seq
                            d_12_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            (lm).GenerateLogits((d_12_constrainedPrompt_) + (currentConstrainedOut))
                            d_13_argmax2_: _dafny.Seq
                            out10_: _dafny.Seq
                            out10_ = (d_0_helpers_).GetHighestLogitToken(lm)
                            d_13_argmax2_ = out10_
                            d_14_argmaxValid_: bool
                            out11_: bool
                            out11_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_13_argmax2_)
                            d_14_argmaxValid_ = out11_
                            if (d_14_argmaxValid_) and ((d_13_argmax2_) != (eosToken)):
                                d_15_sampled_: _dafny.Seq
                                out12_: _dafny.Seq
                                out12_ = (lm).ChooseNextTokenUnconstrained()
                                d_15_sampled_ = out12_
                                (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                                d_1_steps_ = (d_1_steps_) + (1)
                                d_16_sampledValid_: bool
                                out13_: bool
                                out13_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_15_sampled_)
                                d_16_sampledValid_ = out13_
                                if ((d_15_sampled_) != (eosToken)) and (d_16_sampledValid_):
                                    d_17_appendedGenerated_: _dafny.Seq
                                    d_18_appendedInside_: bool
                                    d_19_appendedCurrent_: _dafny.Seq
                                    out14_: _dafny.Seq
                                    out15_: bool
                                    out16_: _dafny.Seq
                                    out14_, out15_, out16_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_15_sampled_)
                                    d_17_appendedGenerated_ = out14_
                                    d_18_appendedInside_ = out15_
                                    d_19_appendedCurrent_ = out16_
                                    generated = d_17_appendedGenerated_
                                    insideConstrainedOut = d_18_appendedInside_
                                    currentConstrainedOut = d_19_appendedCurrent_
                                elif True:
                                    d_20_appendedGenerated2_: _dafny.Seq
                                    d_21_appendedInside2_: bool
                                    d_22_appendedCurrent2_: _dafny.Seq
                                    out17_: _dafny.Seq
                                    out18_: bool
                                    out19_: _dafny.Seq
                                    out17_, out18_, out19_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_13_argmax2_)
                                    d_20_appendedGenerated2_ = out17_
                                    d_21_appendedInside2_ = out18_
                                    d_22_appendedCurrent2_ = out19_
                                    generated = d_20_appendedGenerated2_
                                    insideConstrainedOut = d_21_appendedInside2_
                                    currentConstrainedOut = d_22_appendedCurrent2_
                            elif True:
                                d_23_next2_: _dafny.Seq
                                out20_: _dafny.Seq
                                out20_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_12_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_23_next2_ = out20_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_23_next2_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_24_appendedGenerated3_: _dafny.Seq
                                    d_25_appendedInside3_: bool
                                    d_26_appendedCurrent3_: _dafny.Seq
                                    out21_: _dafny.Seq
                                    out22_: bool
                                    out23_: _dafny.Seq
                                    out21_, out22_, out23_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_23_next2_)
                                    d_24_appendedGenerated3_ = out21_
                                    d_25_appendedInside3_ = out22_
                                    d_26_appendedCurrent3_ = out23_
                                    generated = d_24_appendedGenerated3_
                                    insideConstrainedOut = d_25_appendedInside3_
                                    currentConstrainedOut = d_26_appendedCurrent3_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

