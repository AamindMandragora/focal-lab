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
    def MyCSDStrategy(lm, parser, prompt, generatedPrefix, insideConstrained, currentConstrained, maxSteps, stepTokenBudget, eosToken):
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
                        d_2_chunkBudget_: int
                        d_2_chunkBudget_ = (maxSteps) - (d_1_steps_)
                        d_3_chunkedGenerated_: _dafny.Seq
                        d_4_stoppedOnOpenSpan_: bool
                        d_5_stoppedOnEos_: bool
                        d_6_stepsUsed_: int
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: bool
                        out3_: int
                        out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_2_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_3_chunkedGenerated_ = out0_
                        d_4_stoppedOnOpenSpan_ = out1_
                        d_5_stoppedOnEos_ = out2_
                        d_6_stepsUsed_ = out3_
                        generated = d_3_chunkedGenerated_
                        d_1_steps_ = (d_1_steps_) + (d_6_stepsUsed_)
                        if d_5_stoppedOnEos_:
                            raise _dafny.Break("0")
                        elif d_4_stoppedOnOpenSpan_:
                            insideConstrainedOut = True
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        d_7_isComplete_: bool
                        d_7_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_7_isComplete_:
                            d_8_closedGenerated_: _dafny.Seq
                            d_9_closedInside_: bool
                            d_10_closedCurrent_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_8_closedGenerated_ = out4_
                            d_9_closedInside_ = out5_
                            d_10_closedCurrent_ = out6_
                            generated = d_8_closedGenerated_
                            insideConstrainedOut = d_9_closedInside_
                            currentConstrainedOut = d_10_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_11_stablePrefix_: _dafny.Seq
                            d_11_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_12_constrainedPrompt_: _dafny.Seq
                            d_12_constrainedPrompt_ = (prompt) + (d_11_stablePrefix_)
                            d_13_narrow_: bool
                            out7_: bool
                            out7_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 2)
                            d_13_narrow_ = out7_
                            if d_13_narrow_:
                                d_14_next_: _dafny.Seq
                                out8_: _dafny.Seq
                                out8_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_12_constrainedPrompt_, currentConstrainedOut, eosToken)
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
                                (lm).GenerateLogits((d_12_constrainedPrompt_) + (currentConstrainedOut))
                                d_18_argmax_: _dafny.Seq
                                out12_: _dafny.Seq
                                out12_ = (d_0_helpers_).GetHighestLogitToken(lm)
                                d_18_argmax_ = out12_
                                d_19_argmaxValid_: bool
                                out13_: bool
                                out13_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_18_argmax_)
                                d_19_argmaxValid_ = out13_
                                if (d_19_argmaxValid_) and ((d_18_argmax_) != (eosToken)):
                                    d_20_appendedGenerated2_: _dafny.Seq
                                    d_21_appendedInside2_: bool
                                    d_22_appendedCurrent2_: _dafny.Seq
                                    out14_: _dafny.Seq
                                    out15_: bool
                                    out16_: _dafny.Seq
                                    out14_, out15_, out16_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_18_argmax_)
                                    d_20_appendedGenerated2_ = out14_
                                    d_21_appendedInside2_ = out15_
                                    d_22_appendedCurrent2_ = out16_
                                    generated = d_20_appendedGenerated2_
                                    insideConstrainedOut = d_21_appendedInside2_
                                    currentConstrainedOut = d_22_appendedCurrent2_
                                    (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                                    d_1_steps_ = (d_1_steps_) + (1)
                                elif True:
                                    d_23_next2_: _dafny.Seq
                                    out17_: _dafny.Seq
                                    out17_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_12_constrainedPrompt_, currentConstrainedOut, eosToken)
                                    d_23_next2_ = out17_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    if (d_23_next2_) == (eosToken):
                                        raise _dafny.Break("0")
                                    elif True:
                                        d_24_appendedGenerated3_: _dafny.Seq
                                        d_25_appendedInside3_: bool
                                        d_26_appendedCurrent3_: _dafny.Seq
                                        out18_: _dafny.Seq
                                        out19_: bool
                                        out20_: _dafny.Seq
                                        out18_, out19_, out20_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_23_next2_)
                                        d_24_appendedGenerated3_ = out18_
                                        d_25_appendedInside3_ = out19_
                                        d_26_appendedCurrent3_ = out20_
                                        generated = d_24_appendedGenerated3_
                                        insideConstrainedOut = d_25_appendedInside3_
                                        currentConstrainedOut = d_26_appendedCurrent3_
                    pass
            pass
        cost = d_1_steps_
        generated = generatedPrefix
        insideConstrainedOut = insideConstrained
        currentConstrainedOut = currentConstrained
        cost = 0
        if (maxSteps) > (0):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

