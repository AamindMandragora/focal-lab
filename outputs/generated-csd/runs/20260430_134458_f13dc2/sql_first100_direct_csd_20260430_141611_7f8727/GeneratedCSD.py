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
        generated = generatedPrefix
        insideConstrainedOut = insideConstrained
        currentConstrainedOut = currentConstrained
        cost = 0
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_canOpen_: bool
        d_2_canOpen_ = (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))) in ((lm).Tokens)
        d_3_chunkSize_: int
        d_3_chunkSize_ = 12
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_4_remaining_: int
                        d_4_remaining_ = (maxSteps) - (d_1_steps_)
                        d_5_maxChunk_: int
                        d_5_maxChunk_ = d_3_chunkSize_
                        if (d_4_remaining_) < (d_5_maxChunk_):
                            d_5_maxChunk_ = d_4_remaining_
                        if (d_5_maxChunk_) == (0):
                            raise _dafny.Break("0")
                        elif True:
                            d_6_chunkGenerated_: _dafny.Seq
                            d_7_stoppedOnOpenSpan_: bool
                            d_8_stoppedOnEos_: bool
                            d_9_stepsUsed_: int
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: bool
                            out3_: int
                            out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_5_maxChunk_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_6_chunkGenerated_ = out0_
                            d_7_stoppedOnOpenSpan_ = out1_
                            d_8_stoppedOnEos_ = out2_
                            d_9_stepsUsed_ = out3_
                            generated = d_6_chunkGenerated_
                            d_1_steps_ = (d_1_steps_) + (d_9_stepsUsed_)
                            if d_8_stoppedOnEos_:
                                raise _dafny.Break("0")
                            elif ((d_7_stoppedOnOpenSpan_) and ((d_1_steps_) < (maxSteps))) and (d_2_canOpen_):
                                d_10_openedGenerated_: _dafny.Seq
                                d_11_openedInside_: bool
                                d_12_openedCurrent_: _dafny.Seq
                                out4_: _dafny.Seq
                                out5_: bool
                                out6_: _dafny.Seq
                                out4_, out5_, out6_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                d_10_openedGenerated_ = out4_
                                d_11_openedInside_ = out5_
                                d_12_openedCurrent_ = out6_
                                generated = d_10_openedGenerated_
                                insideConstrainedOut = d_11_openedInside_
                                currentConstrainedOut = d_12_openedCurrent_
                                d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_13_completeNow_: bool
                        d_13_completeNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_13_completeNow_:
                            if (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))) in ((lm).Tokens):
                                d_14_closedGenerated_: _dafny.Seq
                                d_15_closedInside_: bool
                                d_16_closedCurrent_: _dafny.Seq
                                out7_: _dafny.Seq
                                out8_: bool
                                out9_: _dafny.Seq
                                out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_14_closedGenerated_ = out7_
                                d_15_closedInside_ = out8_
                                d_16_closedCurrent_ = out9_
                                generated = d_14_closedGenerated_
                                insideConstrainedOut = d_15_closedInside_
                                currentConstrainedOut = d_16_closedCurrent_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                raise _dafny.Break("0")
                        elif True:
                            d_17_dead_: bool
                            out10_: bool
                            out10_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 0)
                            d_17_dead_ = out10_
                            if d_17_dead_:
                                d_18_repairedFrom_: _dafny.Seq
                                out11_: _dafny.Seq
                                out11_ = VerifiedDecoderAgent.CSDHelpers.RollbackToBoundary(parser, currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "FROM")))
                                d_18_repairedFrom_ = out11_
                                d_19_repairedWhere_: _dafny.Seq
                                out12_: _dafny.Seq
                                out12_ = VerifiedDecoderAgent.CSDHelpers.RollbackToBoundary(parser, currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "WHERE")))
                                d_19_repairedWhere_ = out12_
                                d_20_repairedComma_: _dafny.Seq
                                out13_: _dafny.Seq
                                out13_ = VerifiedDecoderAgent.CSDHelpers.RollbackToBoundary(parser, currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ",")))
                                d_20_repairedComma_ = out13_
                                d_21_repaired_: _dafny.Seq
                                d_21_repaired_ = d_20_repairedComma_
                                if (len(d_19_repairedWhere_)) < (len(currentConstrainedOut)):
                                    d_21_repaired_ = d_19_repairedWhere_
                                if (len(d_18_repairedFrom_)) < (len(currentConstrainedOut)):
                                    d_21_repaired_ = d_18_repairedFrom_
                                d_22_stablePrefix_: _dafny.Seq
                                d_22_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                generated = (d_22_stablePrefix_) + (d_21_repaired_)
                                insideConstrainedOut = True
                                currentConstrainedOut = d_21_repaired_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                d_23_stablePrefix2_: _dafny.Seq
                                d_23_stablePrefix2_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                d_24_next_: _dafny.Seq
                                out14_: _dafny.Seq
                                out14_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, (prompt) + (d_23_stablePrefix2_), currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                                d_24_next_ = out14_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_24_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_25_appendedGenerated_: _dafny.Seq
                                    d_26_appendedInside_: bool
                                    d_27_appendedCurrent_: _dafny.Seq
                                    out15_: _dafny.Seq
                                    out16_: bool
                                    out17_: _dafny.Seq
                                    out15_, out16_, out17_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_24_next_)
                                    d_25_appendedGenerated_ = out15_
                                    d_26_appendedInside_ = out16_
                                    d_27_appendedCurrent_ = out17_
                                    generated = d_25_appendedGenerated_
                                    insideConstrainedOut = d_26_appendedInside_
                                    currentConstrainedOut = d_27_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

