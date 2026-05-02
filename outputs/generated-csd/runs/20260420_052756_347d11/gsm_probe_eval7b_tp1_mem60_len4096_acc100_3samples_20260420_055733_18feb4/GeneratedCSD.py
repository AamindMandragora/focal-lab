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
        while (d_1_steps_) < (maxSteps):
            if not(insideConstrainedOut):
                (lm).GenerateLogits((prompt) + (generated))
                if (len(generated)) > (0):
                    d_2_lastTok_: _dafny.Seq
                    d_2_lastTok_ = (generated)[(len(generated)) - (1)]
                    if (((VerifiedDecoderAgent.default__.Contains(d_2_lastTok_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "answer")))) or (VerifiedDecoderAgent.default__.Contains(d_2_lastTok_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Answer"))))) or (VerifiedDecoderAgent.default__.Contains(d_2_lastTok_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "final"))))) or (VerifiedDecoderAgent.default__.Contains(d_2_lastTok_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Final")))):
                        (d_0_helpers_).PenalizeTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]), _dafny.BigRational('8e0'))
                        (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([eosToken]), _dafny.BigRational('3e0'))
                    elif True:
                        if (((((VerifiedDecoderAgent.default__.Contains(d_2_lastTok_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Let")))) or (VerifiedDecoderAgent.default__.Contains(d_2_lastTok_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "let"))))) or (VerifiedDecoderAgent.default__.Contains(d_2_lastTok_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "="))))) or (VerifiedDecoderAgent.default__.Contains(d_2_lastTok_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ","))))) or (VerifiedDecoderAgent.default__.Contains(d_2_lastTok_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ":"))))) or (VerifiedDecoderAgent.default__.Contains(d_2_lastTok_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\n")))):
                            (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]), _dafny.BigRational('4e0'))
                d_3_next_: _dafny.Seq
                out0_: _dafny.Seq
                out0_ = (lm).ChooseNextTokenUnconstrained()
                d_3_next_ = out0_
                (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                d_1_steps_ = (d_1_steps_) + (1)
                if (d_3_next_) == (eosToken):
                    d_1_steps_ = maxSteps
                elif True:
                    if (d_3_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                        d_4_openedGenerated_: _dafny.Seq
                        d_5_openedInside_: bool
                        d_6_openedCurrent_: _dafny.Seq
                        out1_: _dafny.Seq
                        out2_: bool
                        out3_: _dafny.Seq
                        out1_, out2_, out3_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                        d_4_openedGenerated_ = out1_
                        d_5_openedInside_ = out2_
                        d_6_openedCurrent_ = out3_
                        generated = d_4_openedGenerated_
                        insideConstrainedOut = d_5_openedInside_
                        currentConstrainedOut = d_6_openedCurrent_
                    elif True:
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_3_next_]))
            elif True:
                if (parser).IsCompletePrefix(currentConstrainedOut):
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
                    d_10_stablePrefix_: _dafny.Seq
                    d_10_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                    d_11_constrainedPrompt_: _dafny.Seq
                    d_11_constrainedPrompt_ = (prompt) + (d_10_stablePrefix_)
                    if (len(currentConstrainedOut)) >= (3):
                        d_12_rolledGenerated_: _dafny.Seq
                        d_13_rolledCurrent_: _dafny.Seq
                        out7_: _dafny.Seq
                        out8_: _dafny.Seq
                        out7_, out8_ = (d_0_helpers_).RollbackConstrainedSpan(parser, d_10_stablePrefix_, generated, currentConstrainedOut)
                        d_12_rolledGenerated_ = out7_
                        d_13_rolledCurrent_ = out8_
                        generated = d_12_rolledGenerated_
                        currentConstrainedOut = d_13_rolledCurrent_
                        insideConstrainedOut = True
                        if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                            d_14_closedGenerated2_: _dafny.Seq
                            d_15_closedInside2_: bool
                            d_16_closedCurrent2_: _dafny.Seq
                            out9_: _dafny.Seq
                            out10_: bool
                            out11_: _dafny.Seq
                            out9_, out10_, out11_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_14_closedGenerated2_ = out9_
                            d_15_closedInside2_ = out10_
                            d_16_closedCurrent2_ = out11_
                            generated = d_14_closedGenerated2_
                            insideConstrainedOut = d_15_closedInside2_
                            currentConstrainedOut = d_16_closedCurrent2_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_17_nextForced_: _dafny.Seq
                            out12_: _dafny.Seq
                            out12_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_11_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_17_nextForced_ = out12_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_17_nextForced_) == (eosToken):
                                d_1_steps_ = maxSteps
                            elif True:
                                d_18_appendedGeneratedForced_: _dafny.Seq
                                d_19_appendedInsideForced_: bool
                                d_20_appendedCurrentForced_: _dafny.Seq
                                out13_: _dafny.Seq
                                out14_: bool
                                out15_: _dafny.Seq
                                out13_, out14_, out15_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_17_nextForced_)
                                d_18_appendedGeneratedForced_ = out13_
                                d_19_appendedInsideForced_ = out14_
                                d_20_appendedCurrentForced_ = out15_
                                generated = d_18_appendedGeneratedForced_
                                insideConstrainedOut = d_19_appendedInsideForced_
                                currentConstrainedOut = d_20_appendedCurrentForced_
                    elif True:
                        d_21_narrow_: bool
                        out16_: bool
                        out16_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 2)
                        d_21_narrow_ = out16_
                        if d_21_narrow_:
                            d_22_next1_: _dafny.Seq
                            out17_: _dafny.Seq
                            out17_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_11_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_22_next1_ = out17_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_22_next1_) == (eosToken):
                                d_1_steps_ = maxSteps
                            elif True:
                                d_23_appendedGenerated1_: _dafny.Seq
                                d_24_appendedInside1_: bool
                                d_25_appendedCurrent1_: _dafny.Seq
                                out18_: _dafny.Seq
                                out19_: bool
                                out20_: _dafny.Seq
                                out18_, out19_, out20_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_22_next1_)
                                d_23_appendedGenerated1_ = out18_
                                d_24_appendedInside1_ = out19_
                                d_25_appendedCurrent1_ = out20_
                                generated = d_23_appendedGenerated1_
                                insideConstrainedOut = d_24_appendedInside1_
                                currentConstrainedOut = d_25_appendedCurrent1_
                        elif True:
                            d_26_next2_: _dafny.Seq
                            d_27_isValid2_: bool
                            out21_: _dafny.Seq
                            out22_: bool
                            out21_, out22_ = (d_0_helpers_).SoftConstrainedStep(lm, parser, d_11_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('8e0'), eosToken)
                            d_26_next2_ = out21_
                            d_27_isValid2_ = out22_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_26_next2_) == (eosToken):
                                d_1_steps_ = maxSteps
                            elif True:
                                if d_27_isValid2_:
                                    d_28_appendedGenerated2_: _dafny.Seq
                                    d_29_appendedInside2_: bool
                                    d_30_appendedCurrent2_: _dafny.Seq
                                    out23_: _dafny.Seq
                                    out24_: bool
                                    out25_: _dafny.Seq
                                    out23_, out24_, out25_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_26_next2_)
                                    d_28_appendedGenerated2_ = out23_
                                    d_29_appendedInside2_ = out24_
                                    d_30_appendedCurrent2_ = out25_
                                    generated = d_28_appendedGenerated2_
                                    insideConstrainedOut = d_29_appendedInside2_
                                    currentConstrainedOut = d_30_appendedCurrent2_
                                elif True:
                                    d_31_repairedGenerated_: _dafny.Seq
                                    d_32_repairedCurrent_: _dafny.Seq
                                    out26_: _dafny.Seq
                                    out27_: _dafny.Seq
                                    out26_, out27_ = (d_0_helpers_).RollbackConstrainedSpan(parser, d_10_stablePrefix_, generated, currentConstrainedOut)
                                    d_31_repairedGenerated_ = out26_
                                    d_32_repairedCurrent_ = out27_
                                    generated = d_31_repairedGenerated_
                                    currentConstrainedOut = d_32_repairedCurrent_
                                    insideConstrainedOut = True
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

