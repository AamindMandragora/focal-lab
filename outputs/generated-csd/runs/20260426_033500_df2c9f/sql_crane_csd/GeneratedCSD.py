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
                        d_5_sqlish_: bool
                        d_5_sqlish_ = (((((d_2_argmax_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SELECT")))) or ((d_2_argmax_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "FROM"))))) or ((d_2_argmax_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "WHERE"))))) or ((d_2_argmax_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "JOIN"))))) or (VerifiedDecoderAgent.default__.Contains(d_2_argmax_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))))
                        d_6_next_: _dafny.Seq
                        out3_: _dafny.Seq
                        out3_ = (lm).ChooseNextTokenUnconstrained()
                        d_6_next_ = out3_
                        (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_6_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif ((d_6_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))) or (((d_4_openLogit_) >= ((d_3_argmaxLogit_) - (_dafny.BigRational('2e0')))) and (d_5_sqlish_)):
                            d_7_openedGenerated_: _dafny.Seq
                            d_8_openedInside_: bool
                            d_9_openedCurrent_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_7_openedGenerated_ = out4_
                            d_8_openedInside_ = out5_
                            d_9_openedCurrent_ = out6_
                            generated = d_7_openedGenerated_
                            insideConstrainedOut = d_8_openedInside_
                            currentConstrainedOut = d_9_openedCurrent_
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_6_next_]))
                    elif True:
                        d_10_complete_: bool
                        d_10_complete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_10_complete_:
                            d_11_closedGenerated_: _dafny.Seq
                            d_12_closedInside_: bool
                            d_13_closedCurrent_: _dafny.Seq
                            out7_: _dafny.Seq
                            out8_: bool
                            out9_: _dafny.Seq
                            out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_11_closedGenerated_ = out7_
                            d_12_closedInside_ = out8_
                            d_13_closedCurrent_ = out9_
                            generated = d_11_closedGenerated_
                            insideConstrainedOut = d_12_closedInside_
                            currentConstrainedOut = d_13_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_14_constrainedPrompt_: _dafny.Seq
                            d_14_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_15_validCount_: int
                            out10_: int
                            out10_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                            d_15_validCount_ = out10_
                            d_16_narrow_: bool
                            out11_: bool
                            out11_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 2)
                            d_16_narrow_ = out11_
                            if (d_16_narrow_) or ((d_15_validCount_) <= (2)):
                                d_17_nextStrict_: _dafny.Seq
                                out12_: _dafny.Seq
                                out12_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_14_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_17_nextStrict_ = out12_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_17_nextStrict_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_18_appendedGenerated1_: _dafny.Seq
                                    d_19_appendedInside1_: bool
                                    d_20_appendedCurrent1_: _dafny.Seq
                                    out13_: _dafny.Seq
                                    out14_: bool
                                    out15_: _dafny.Seq
                                    out13_, out14_, out15_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_17_nextStrict_)
                                    d_18_appendedGenerated1_ = out13_
                                    d_19_appendedInside1_ = out14_
                                    d_20_appendedCurrent1_ = out15_
                                    generated = d_18_appendedGenerated1_
                                    insideConstrainedOut = d_19_appendedInside1_
                                    currentConstrainedOut = d_20_appendedCurrent1_
                            elif True:
                                d_21_candidates_: _dafny.Seq
                                out16_: _dafny.Seq
                                out16_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_14_constrainedPrompt_, currentConstrainedOut, 4, eosToken)
                                d_21_candidates_ = out16_
                                (lm).GenerateLogits((d_14_constrainedPrompt_) + (currentConstrainedOut))
                                (d_0_helpers_).BoostTokenLogits(lm, d_21_candidates_, _dafny.BigRational('8e0'))
                                d_22_sampled_: _dafny.Seq
                                out17_: _dafny.Seq
                                out17_ = (lm).ChooseNextTokenUnconstrained()
                                d_22_sampled_ = out17_
                                (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_22_sampled_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_23_sampledValid_: bool
                                    out18_: bool
                                    out18_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_22_sampled_)
                                    d_23_sampledValid_ = out18_
                                    if d_23_sampledValid_:
                                        d_24_appendedGenerated2_: _dafny.Seq
                                        d_25_appendedInside2_: bool
                                        d_26_appendedCurrent2_: _dafny.Seq
                                        out19_: _dafny.Seq
                                        out20_: bool
                                        out21_: _dafny.Seq
                                        out19_, out20_, out21_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_22_sampled_)
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

