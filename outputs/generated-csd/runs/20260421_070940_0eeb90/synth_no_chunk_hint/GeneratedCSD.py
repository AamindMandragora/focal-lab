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
                        d_2_chunkLimit_: int
                        d_2_chunkLimit_ = (maxSteps) - (d_1_steps_)
                        if (d_2_chunkLimit_) > (8):
                            d_2_chunkLimit_ = 8
                        d_3_chunkGenerated_: _dafny.Seq
                        d_4_stoppedOnOpenSpan_: bool
                        d_5_stoppedOnEos_: bool
                        d_6_stepsUsed_: int
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: bool
                        out3_: int
                        out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_2_chunkLimit_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_3_chunkGenerated_ = out0_
                        d_4_stoppedOnOpenSpan_ = out1_
                        d_5_stoppedOnEos_ = out2_
                        d_6_stepsUsed_ = out3_
                        generated = d_3_chunkGenerated_
                        d_1_steps_ = (d_1_steps_) + (d_6_stepsUsed_)
                        if d_5_stoppedOnEos_:
                            raise _dafny.Break("0")
                        elif d_4_stoppedOnOpenSpan_:
                            insideConstrainedOut = True
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
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
                            d_12_narrow_: bool
                            out7_: bool
                            out7_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 2)
                            d_12_narrow_ = out7_
                            if d_12_narrow_:
                                d_13_next_: _dafny.Seq
                                out8_: _dafny.Seq
                                out8_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_11_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_13_next_ = out8_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_13_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_14_appendedGenerated_: _dafny.Seq
                                    d_15_appendedInside_: bool
                                    d_16_appendedCurrent_: _dafny.Seq
                                    out9_: _dafny.Seq
                                    out10_: bool
                                    out11_: _dafny.Seq
                                    out9_, out10_, out11_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_13_next_)
                                    d_14_appendedGenerated_ = out9_
                                    d_15_appendedInside_ = out10_
                                    d_16_appendedCurrent_ = out11_
                                    generated = d_14_appendedGenerated_
                                    insideConstrainedOut = d_15_appendedInside_
                                    currentConstrainedOut = d_16_appendedCurrent_
                            elif True:
                                d_17_next_: _dafny.Seq
                                d_18_isValid_: bool
                                out12_: _dafny.Seq
                                out13_: bool
                                out12_, out13_ = (d_0_helpers_).SoftConstrainedStep(lm, parser, d_11_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('3e0'), eosToken)
                                d_17_next_ = out12_
                                d_18_isValid_ = out13_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_17_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif d_18_isValid_:
                                    d_19_appendedGenerated_: _dafny.Seq
                                    d_20_appendedInside_: bool
                                    d_21_appendedCurrent_: _dafny.Seq
                                    out14_: _dafny.Seq
                                    out15_: bool
                                    out16_: _dafny.Seq
                                    out14_, out15_, out16_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_17_next_)
                                    d_19_appendedGenerated_ = out14_
                                    d_20_appendedInside_ = out15_
                                    d_21_appendedCurrent_ = out16_
                                    generated = d_19_appendedGenerated_
                                    insideConstrainedOut = d_20_appendedInside_
                                    currentConstrainedOut = d_21_appendedCurrent_
                                elif True:
                                    d_22_repairedGenerated_: _dafny.Seq
                                    d_23_repairedCurrent_: _dafny.Seq
                                    out17_: _dafny.Seq
                                    out18_: _dafny.Seq
                                    out17_, out18_ = (d_0_helpers_).RollbackConstrainedSpan(parser, d_10_stablePrefix_, generated, currentConstrainedOut)
                                    d_22_repairedGenerated_ = out17_
                                    d_23_repairedCurrent_ = out18_
                                    generated = d_22_repairedGenerated_
                                    currentConstrainedOut = d_23_repairedCurrent_
                                    if (parser).IsValidPrefix(currentConstrainedOut):
                                        insideConstrainedOut = True
                                    elif True:
                                        insideConstrainedOut = False
                                        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

