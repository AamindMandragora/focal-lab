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
        d_2_narrowThreshold_: int
        d_2_narrowThreshold_ = 10
        d_3_deadEndMinValid_: int
        d_3_deadEndMinValid_ = 2
        d_4_rollbackLenThreshold_: int
        d_4_rollbackLenThreshold_ = 80
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_5_chunkBudget_: int
                        d_5_chunkBudget_ = (maxSteps) - (d_1_steps_)
                        d_6_chunkedGenerated_: _dafny.Seq
                        d_7_stoppedOnOpenSpan_: bool
                        d_8_stoppedOnEos_: bool
                        d_9_stepsUsed_: int
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: bool
                        out3_: int
                        out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_5_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_6_chunkedGenerated_ = out0_
                        d_7_stoppedOnOpenSpan_ = out1_
                        d_8_stoppedOnEos_ = out2_
                        d_9_stepsUsed_ = out3_
                        generated = d_6_chunkedGenerated_
                        d_1_steps_ = (d_1_steps_) + (d_9_stepsUsed_)
                        if d_8_stoppedOnEos_:
                            raise _dafny.Break("0")
                        elif d_7_stoppedOnOpenSpan_:
                            insideConstrainedOut = True
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        d_10_complete_: bool
                        d_10_complete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_10_complete_:
                            d_11_closedGenerated_: _dafny.Seq
                            d_12_closedInside_: bool
                            d_13_closedCurrent_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_11_closedGenerated_ = out4_
                            d_12_closedInside_ = out5_
                            d_13_closedCurrent_ = out6_
                            generated = d_11_closedGenerated_
                            insideConstrainedOut = d_12_closedInside_
                            currentConstrainedOut = d_13_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_14_deadEnd_: bool
                            out7_: bool
                            out7_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, d_3_deadEndMinValid_)
                            d_14_deadEnd_ = out7_
                            if (d_14_deadEnd_) and ((d_4_rollbackLenThreshold_) <= (len(currentConstrainedOut))):
                                d_15_repaired_: _dafny.Seq
                                out8_: _dafny.Seq
                                out8_ = VerifiedDecoderAgent.CSDHelpers.RollbackToBoundary(parser, currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ",")))
                                d_15_repaired_ = out8_
                                d_16_trim_: int
                                d_16_trim_ = (len(currentConstrainedOut)) - (len(d_15_repaired_))
                                generated = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (d_16_trim_):])
                                currentConstrainedOut = d_15_repaired_
                                insideConstrainedOut = True
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                d_17_validCount_: int
                                out9_: int
                                out9_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                                d_17_validCount_ = out9_
                                d_18_stablePrefix_: _dafny.Seq
                                d_18_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                if (d_17_validCount_) <= (d_2_narrowThreshold_):
                                    d_19_next_: _dafny.Seq
                                    out10_: _dafny.Seq
                                    out10_ = (d_0_helpers_).PenalizedConstrainedStep(lm, parser, (prompt) + (d_18_stablePrefix_), currentConstrainedOut, _dafny.SeqWithoutIsStrInference([eosToken]), _dafny.BigRational('1e1'), eosToken)
                                    d_19_next_ = out10_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    if (d_19_next_) == (eosToken):
                                        raise _dafny.Break("0")
                                    elif True:
                                        d_20_appendedGenerated_: _dafny.Seq
                                        d_21_appendedInside_: bool
                                        d_22_appendedCurrent_: _dafny.Seq
                                        out11_: _dafny.Seq
                                        out12_: bool
                                        out13_: _dafny.Seq
                                        out11_, out12_, out13_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_19_next_)
                                        d_20_appendedGenerated_ = out11_
                                        d_21_appendedInside_ = out12_
                                        d_22_appendedCurrent_ = out13_
                                        generated = d_20_appendedGenerated_
                                        insideConstrainedOut = d_21_appendedInside_
                                        currentConstrainedOut = d_22_appendedCurrent_
                                elif True:
                                    d_23_next2_: _dafny.Seq
                                    out14_: _dafny.Seq
                                    out14_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, (prompt) + (d_18_stablePrefix_), currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 8, eosToken)
                                    d_23_next2_ = out14_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    if (d_23_next2_) == (eosToken):
                                        raise _dafny.Break("0")
                                    elif True:
                                        d_24_appendedGenerated2_: _dafny.Seq
                                        d_25_appendedInside2_: bool
                                        d_26_appendedCurrent2_: _dafny.Seq
                                        out15_: _dafny.Seq
                                        out16_: bool
                                        out17_: _dafny.Seq
                                        out15_, out16_, out17_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_23_next2_)
                                        d_24_appendedGenerated2_ = out15_
                                        d_25_appendedInside2_ = out16_
                                        d_26_appendedCurrent2_ = out17_
                                        generated = d_24_appendedGenerated2_
                                        insideConstrainedOut = d_25_appendedInside2_
                                        currentConstrainedOut = d_26_appendedCurrent2_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

