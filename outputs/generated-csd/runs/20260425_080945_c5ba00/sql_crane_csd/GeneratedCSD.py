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
        d_2_openedAny_ = insideConstrained
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
                        d_6_shouldOpen_: bool
                        d_6_shouldOpen_ = False
                        if not(d_2_openedAny_):
                            if (d_5_openLogit_) >= ((d_4_argmaxLogit_) - (_dafny.BigRational('3e0'))):
                                d_6_shouldOpen_ = True
                            elif True:
                                if ((VerifiedDecoderAgent.default__.Contains(d_3_argmax_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SELECT")))) or (VerifiedDecoderAgent.default__.Contains(d_3_argmax_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "FROM"))))) or (VerifiedDecoderAgent.default__.Contains(d_3_argmax_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "sql")))):
                                    d_6_shouldOpen_ = True
                        (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                        d_1_steps_ = (d_1_steps_) + (1)
                        if d_6_shouldOpen_:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]))
                            insideConstrainedOut = True
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                            d_2_openedAny_ = True
                        elif (d_3_argmax_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_3_argmax_]))
                            if ((d_3_argmax_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))) and (not(d_2_openedAny_)):
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                d_2_openedAny_ = True
                    elif True:
                        d_7_complete_: bool
                        d_7_complete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_7_complete_:
                            d_8_closedGenerated_: _dafny.Seq
                            d_9_closedInside_: bool
                            d_10_closedCurrent_: _dafny.Seq
                            out3_: _dafny.Seq
                            out4_: bool
                            out5_: _dafny.Seq
                            out3_, out4_, out5_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_8_closedGenerated_ = out3_
                            d_9_closedInside_ = out4_
                            d_10_closedCurrent_ = out5_
                            generated = d_8_closedGenerated_
                            insideConstrainedOut = d_9_closedInside_
                            currentConstrainedOut = d_10_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_11_constrainedPrompt_: _dafny.Seq
                            d_11_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_12_validCount_: int
                            out6_: int
                            out6_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                            d_12_validCount_ = out6_
                            d_13_narrow_: bool
                            out7_: bool
                            out7_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 2)
                            d_13_narrow_ = out7_
                            if (d_13_narrow_) or ((d_12_validCount_) <= (2)):
                                d_14_next_: _dafny.Seq
                                out8_: _dafny.Seq
                                out8_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_11_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_14_next_ = out8_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_14_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_15_appendedGenerated_: _dafny.Seq
                                    d_16_appendedInside_: bool
                                    d_17_appendedCurrent_: _dafny.Seq
                                    out9_: _dafny.Seq
                                    out10_: bool
                                    out11_: _dafny.Seq
                                    out9_, out10_, out11_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_14_next_)
                                    d_15_appendedGenerated_ = out9_
                                    d_16_appendedInside_ = out10_
                                    d_17_appendedCurrent_ = out11_
                                    generated = d_15_appendedGenerated_
                                    insideConstrainedOut = d_16_appendedInside_
                                    currentConstrainedOut = d_17_appendedCurrent_
                            elif True:
                                d_18_candidates_: _dafny.Seq
                                out12_: _dafny.Seq
                                out12_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_11_constrainedPrompt_, currentConstrainedOut, 4, eosToken)
                                d_18_candidates_ = out12_
                                (lm).GenerateLogits((d_11_constrainedPrompt_) + (currentConstrainedOut))
                                (d_0_helpers_).BoostTokenLogits(lm, d_18_candidates_, _dafny.BigRational('8e0'))
                                (d_0_helpers_).PenalizeTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]), _dafny.BigRational('1e2'))
                                d_19_next_: _dafny.Seq
                                out13_: _dafny.Seq
                                out13_ = (lm).ChooseNextTokenUnconstrained()
                                d_19_next_ = out13_
                                (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_19_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_20_valid_: bool
                                    out14_: bool
                                    out14_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_19_next_)
                                    d_20_valid_ = out14_
                                    if d_20_valid_:
                                        d_21_appendedGenerated_: _dafny.Seq
                                        d_22_appendedInside_: bool
                                        d_23_appendedCurrent_: _dafny.Seq
                                        out15_: _dafny.Seq
                                        out16_: bool
                                        out17_: _dafny.Seq
                                        out15_, out16_, out17_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_19_next_)
                                        d_21_appendedGenerated_ = out15_
                                        d_22_appendedInside_ = out16_
                                        d_23_appendedCurrent_ = out17_
                                        generated = d_21_appendedGenerated_
                                        insideConstrainedOut = d_22_appendedInside_
                                        currentConstrainedOut = d_23_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

