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
    def MyCSDStrategy(lm, parser, prompt, generatedPrefix, insideConstrained, currentConstrained, maxSteps, stepTokenBudget, validTokenGroups, eosToken):
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
        d_2_chunkLimit_: int
        d_2_chunkLimit_ = (stepTokenBudget) + (1)
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_3_remaining_: int
                        d_3_remaining_ = (maxSteps) - (d_1_steps_)
                        d_4_maxChunk_: int
                        if (d_2_chunkLimit_) < (d_3_remaining_):
                            d_4_maxChunk_ = d_2_chunkLimit_
                        elif True:
                            d_4_maxChunk_ = d_3_remaining_
                        d_5_chunkGenerated_: _dafny.Seq
                        d_6_stoppedOnOpenSpan_: bool
                        d_7_stoppedOnEos_: bool
                        d_8_stepsUsed_: int
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: bool
                        out3_: int
                        out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_4_maxChunk_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_5_chunkGenerated_ = out0_
                        d_6_stoppedOnOpenSpan_ = out1_
                        d_7_stoppedOnEos_ = out2_
                        d_8_stepsUsed_ = out3_
                        generated = d_5_chunkGenerated_
                        d_1_steps_ = (d_1_steps_) + (d_8_stepsUsed_)
                        if d_7_stoppedOnEos_:
                            raise _dafny.Break("0")
                        elif True:
                            if d_6_stoppedOnOpenSpan_:
                                if (d_1_steps_) < (maxSteps):
                                    d_9_openedGenerated_: _dafny.Seq
                                    d_10_openedInside_: bool
                                    d_11_openedCurrent_: _dafny.Seq
                                    out4_: _dafny.Seq
                                    out5_: bool
                                    out6_: _dafny.Seq
                                    out4_, out5_, out6_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                    d_9_openedGenerated_ = out4_
                                    d_10_openedInside_ = out5_
                                    d_11_openedCurrent_ = out6_
                                    generated = d_9_openedGenerated_
                                    insideConstrainedOut = d_10_openedInside_
                                    currentConstrainedOut = d_11_openedCurrent_
                                    d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_12_isComplete_: bool
                        d_12_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_12_isComplete_:
                            d_13_closedGenerated_: _dafny.Seq
                            d_14_closedInside_: bool
                            d_15_closedCurrent_: _dafny.Seq
                            out7_: _dafny.Seq
                            out8_: bool
                            out9_: _dafny.Seq
                            out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_13_closedGenerated_ = out7_
                            d_14_closedInside_ = out8_
                            d_15_closedCurrent_ = out9_
                            generated = d_13_closedGenerated_
                            insideConstrainedOut = d_14_closedInside_
                            currentConstrainedOut = d_15_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_16_stablePrefix_: _dafny.Seq
                            d_16_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_17_validCount_: int
                            out10_: int
                            out10_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                            d_17_validCount_ = out10_
                            d_18_next_: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                            if (d_17_validCount_) <= (2):
                                d_19_stepGenerated_: _dafny.Seq
                                d_20_stepInside_: bool
                                d_21_stepCurrent_: _dafny.Seq
                                d_22_hitEos_: bool
                                out11_: _dafny.Seq
                                out12_: bool
                                out13_: _dafny.Seq
                                out14_: bool
                                out11_, out12_, out13_, out14_ = (d_0_helpers_).ConstrainedStep(lm, parser, (prompt) + (d_16_stablePrefix_), generated, currentConstrainedOut, eosToken)
                                d_19_stepGenerated_ = out11_
                                d_20_stepInside_ = out12_
                                d_21_stepCurrent_ = out13_
                                d_22_hitEos_ = out14_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if d_22_hitEos_:
                                    raise _dafny.Break("0")
                                elif True:
                                    generated = d_19_stepGenerated_
                                    insideConstrainedOut = d_20_stepInside_
                                    currentConstrainedOut = d_21_stepCurrent_
                            elif True:
                                out15_: _dafny.Seq
                                out15_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, (prompt) + (d_16_stablePrefix_), currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                                d_18_next_ = out15_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_18_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_23_appendedGenerated_: _dafny.Seq
                                    d_24_appendedInside_: bool
                                    d_25_appendedCurrent_: _dafny.Seq
                                    out16_: _dafny.Seq
                                    out17_: bool
                                    out18_: _dafny.Seq
                                    out16_, out17_, out18_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_18_next_)
                                    d_23_appendedGenerated_ = out16_
                                    d_24_appendedInside_ = out17_
                                    d_25_appendedCurrent_ = out18_
                                    generated = d_23_appendedGenerated_
                                    insideConstrainedOut = d_24_appendedInside_
                                    currentConstrainedOut = d_25_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

