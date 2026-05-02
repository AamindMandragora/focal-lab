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
                        d_5_shouldEagerOpen_: bool
                        d_5_shouldEagerOpen_ = (((d_4_openLogit_) >= ((d_3_argmaxLogit_) - (_dafny.BigRational('3e0')))) and ((d_2_argmax_) != (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))))) and ((d_2_argmax_) != (eosToken))
                        if d_5_shouldEagerOpen_:
                            (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                            d_1_steps_ = (d_1_steps_) + (1)
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]))
                            insideConstrainedOut = True
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                        elif True:
                            d_6_chunkBudget_: int
                            d_6_chunkBudget_ = (maxSteps) - (d_1_steps_)
                            d_7_chunkedGenerated_: _dafny.Seq
                            d_8_stoppedOnOpen_: bool
                            d_9_stoppedOnEos_: bool
                            d_10_stepsUsed_: int
                            out3_: _dafny.Seq
                            out4_: bool
                            out5_: bool
                            out6_: int
                            out3_, out4_, out5_, out6_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_6_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_7_chunkedGenerated_ = out3_
                            d_8_stoppedOnOpen_ = out4_
                            d_9_stoppedOnEos_ = out5_
                            d_10_stepsUsed_ = out6_
                            generated = d_7_chunkedGenerated_
                            d_1_steps_ = (d_1_steps_) + (d_10_stepsUsed_)
                            if d_9_stoppedOnEos_:
                                raise _dafny.Break("0")
                            elif d_8_stoppedOnOpen_:
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        d_11_complete_: bool
                        d_11_complete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_11_complete_:
                            d_12_closedGenerated_: _dafny.Seq
                            d_13_closedInside_: bool
                            d_14_closedCurrent_: _dafny.Seq
                            out7_: _dafny.Seq
                            out8_: bool
                            out9_: _dafny.Seq
                            out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_12_closedGenerated_ = out7_
                            d_13_closedInside_ = out8_
                            d_14_closedCurrent_ = out9_
                            generated = d_12_closedGenerated_
                            insideConstrainedOut = d_13_closedInside_
                            currentConstrainedOut = d_14_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_15_constrainedPrompt_: _dafny.Seq
                            d_15_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            (lm).GenerateLogits((d_15_constrainedPrompt_) + (currentConstrainedOut))
                            d_16_argmaxInside_: _dafny.Seq
                            out10_: _dafny.Seq
                            out10_ = (d_0_helpers_).GetHighestLogitToken(lm)
                            d_16_argmaxInside_ = out10_
                            d_17_argmaxInsideValid_: bool
                            out11_: bool
                            out11_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_16_argmaxInside_)
                            d_17_argmaxInsideValid_ = out11_
                            if (d_17_argmaxInsideValid_) and ((d_16_argmaxInside_) != (eosToken)):
                                d_18_appendedGeneratedFast_: _dafny.Seq
                                d_19_appendedInsideFast_: bool
                                d_20_appendedCurrentFast_: _dafny.Seq
                                out12_: _dafny.Seq
                                out13_: bool
                                out14_: _dafny.Seq
                                out12_, out13_, out14_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_16_argmaxInside_)
                                d_18_appendedGeneratedFast_ = out12_
                                d_19_appendedInsideFast_ = out13_
                                d_20_appendedCurrentFast_ = out14_
                                generated = d_18_appendedGeneratedFast_
                                insideConstrainedOut = d_19_appendedInsideFast_
                                currentConstrainedOut = d_20_appendedCurrentFast_
                                (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                d_21_next_: _dafny.Seq
                                out15_: _dafny.Seq
                                out15_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_15_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_21_next_ = out15_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_21_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_22_appendedGenerated_: _dafny.Seq
                                    d_23_appendedInside_: bool
                                    d_24_appendedCurrent_: _dafny.Seq
                                    out16_: _dafny.Seq
                                    out17_: bool
                                    out18_: _dafny.Seq
                                    out16_, out17_, out18_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_21_next_)
                                    d_22_appendedGenerated_ = out16_
                                    d_23_appendedInside_ = out17_
                                    d_24_appendedCurrent_ = out18_
                                    generated = d_22_appendedGenerated_
                                    insideConstrainedOut = d_23_appendedInside_
                                    currentConstrainedOut = d_24_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

