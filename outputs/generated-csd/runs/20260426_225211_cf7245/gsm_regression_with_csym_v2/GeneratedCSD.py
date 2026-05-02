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
                        if (((d_4_openLogit_) >= ((d_3_argmaxLogit_) - (_dafny.BigRational('2e0')))) and ((d_2_argmax_) != (eosToken))) and ((d_2_argmax_) != (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))):
                            d_1_steps_ = (d_1_steps_) + (1)
                            (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]))
                            insideConstrainedOut = True
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                        elif True:
                            d_5_chunkBudget_: int
                            d_5_chunkBudget_ = (maxSteps) - (d_1_steps_)
                            d_6_chunkedGenerated_: _dafny.Seq
                            d_7_stoppedOpen_: bool
                            d_8_stoppedEos_: bool
                            d_9_stepsUsed_: int
                            out3_: _dafny.Seq
                            out4_: bool
                            out5_: bool
                            out6_: int
                            out3_, out4_, out5_, out6_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_5_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_6_chunkedGenerated_ = out3_
                            d_7_stoppedOpen_ = out4_
                            d_8_stoppedEos_ = out5_
                            d_9_stepsUsed_ = out6_
                            generated = d_6_chunkedGenerated_
                            d_1_steps_ = (d_1_steps_) + (d_9_stepsUsed_)
                            if d_8_stoppedEos_:
                                raise _dafny.Break("0")
                            elif d_7_stoppedOpen_:
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
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
                            (lm).GenerateLogits((d_14_constrainedPrompt_) + (currentConstrainedOut))
                            d_15_argmaxInside_: _dafny.Seq
                            out10_: _dafny.Seq
                            out10_ = (d_0_helpers_).GetHighestLogitToken(lm)
                            d_15_argmaxInside_ = out10_
                            d_16_argmaxInsideValid_: bool
                            out11_: bool
                            out11_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_15_argmaxInside_)
                            d_16_argmaxInsideValid_ = out11_
                            if (d_16_argmaxInsideValid_) and ((d_15_argmaxInside_) != (eosToken)):
                                d_17_appendedGenerated_: _dafny.Seq
                                d_18_appendedInside_: bool
                                d_19_appendedCurrent_: _dafny.Seq
                                out12_: _dafny.Seq
                                out13_: bool
                                out14_: _dafny.Seq
                                out12_, out13_, out14_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_15_argmaxInside_)
                                d_17_appendedGenerated_ = out12_
                                d_18_appendedInside_ = out13_
                                d_19_appendedCurrent_ = out14_
                                generated = d_17_appendedGenerated_
                                insideConstrainedOut = d_18_appendedInside_
                                currentConstrainedOut = d_19_appendedCurrent_
                                d_1_steps_ = (d_1_steps_) + (1)
                                (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                            elif True:
                                d_20_next_: _dafny.Seq
                                out15_: _dafny.Seq
                                out15_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_14_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_20_next_ = out15_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_20_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_21_appendedGenerated2_: _dafny.Seq
                                    d_22_appendedInside2_: bool
                                    d_23_appendedCurrent2_: _dafny.Seq
                                    out16_: _dafny.Seq
                                    out17_: bool
                                    out18_: _dafny.Seq
                                    out16_, out17_, out18_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_20_next_)
                                    d_21_appendedGenerated2_ = out16_
                                    d_22_appendedInside2_ = out17_
                                    d_23_appendedCurrent2_ = out18_
                                    generated = d_21_appendedGenerated2_
                                    insideConstrainedOut = d_22_appendedInside2_
                                    currentConstrainedOut = d_23_appendedCurrent2_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

