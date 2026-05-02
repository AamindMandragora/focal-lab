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
        d_2_preambleBudget_: int
        d_2_preambleBudget_ = 6
        d_3_maxChunk_: int
        d_3_maxChunk_ = 4
        d_4_maxConstrainedLen_: int
        d_4_maxConstrainedLen_ = 10
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if insideConstrainedOut:
                        d_5_completeNow_: bool
                        d_5_completeNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_5_completeNow_:
                            d_6_closedGenerated_: _dafny.Seq
                            d_7_closedInside_: bool
                            d_8_closedCurrent_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_6_closedGenerated_ = out0_
                            d_7_closedInside_ = out1_
                            d_8_closedCurrent_ = out2_
                            generated = d_6_closedGenerated_
                            insideConstrainedOut = d_7_closedInside_
                            currentConstrainedOut = d_8_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif (len(currentConstrainedOut)) >= (d_4_maxConstrainedLen_):
                            d_9_repaired_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = VerifiedDecoderAgent.CSDHelpers.RollbackToBoundary(parser, currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")))
                            d_9_repaired_ = out3_
                            generated = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - ((len(currentConstrainedOut)) - (len(d_9_repaired_))):])
                            currentConstrainedOut = d_9_repaired_
                            if (len(currentConstrainedOut)) == (0):
                                insideConstrainedOut = False
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_10_stablePrefix_: _dafny.Seq
                            d_10_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_11_next_: _dafny.Seq
                            out4_: _dafny.Seq
                            out4_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, (prompt) + (d_10_stablePrefix_), currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                            d_11_next_ = out4_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_11_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_12_appendedGenerated_: _dafny.Seq
                                d_13_appendedInside_: bool
                                d_14_appendedCurrent_: _dafny.Seq
                                out5_: _dafny.Seq
                                out6_: bool
                                out7_: _dafny.Seq
                                out5_, out6_, out7_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_11_next_)
                                d_12_appendedGenerated_ = out5_
                                d_13_appendedInside_ = out6_
                                d_14_appendedCurrent_ = out7_
                                generated = d_12_appendedGenerated_
                                insideConstrainedOut = d_13_appendedInside_
                                currentConstrainedOut = d_14_appendedCurrent_
                    elif True:
                        if (d_1_steps_) < (d_2_preambleBudget_):
                            d_15_remaining_: int
                            d_15_remaining_ = (maxSteps) - (d_1_steps_)
                            d_16_chunkSize_: int
                            d_16_chunkSize_ = d_3_maxChunk_
                            if (d_15_remaining_) < (d_16_chunkSize_):
                                d_16_chunkSize_ = d_15_remaining_
                            if (d_16_chunkSize_) == (0):
                                raise _dafny.Break("0")
                            elif True:
                                d_17_chunkGenerated_: _dafny.Seq
                                d_18_stoppedOnOpenSpan_: bool
                                d_19_stoppedOnEos_: bool
                                d_20_stepsUsed_: int
                                out8_: _dafny.Seq
                                out9_: bool
                                out10_: bool
                                out11_: int
                                out8_, out9_, out10_, out11_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_16_chunkSize_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                                d_17_chunkGenerated_ = out8_
                                d_18_stoppedOnOpenSpan_ = out9_
                                d_19_stoppedOnEos_ = out10_
                                d_20_stepsUsed_ = out11_
                                generated = d_17_chunkGenerated_
                                d_1_steps_ = (d_1_steps_) + (d_20_stepsUsed_)
                                if d_19_stoppedOnEos_:
                                    raise _dafny.Break("0")
                                elif d_18_stoppedOnOpenSpan_:
                                    insideConstrainedOut = True
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                        elif True:
                            d_21_openedGenerated_: _dafny.Seq
                            d_22_openedInside_: bool
                            d_23_openedCurrent_: _dafny.Seq
                            out12_: _dafny.Seq
                            out13_: bool
                            out14_: _dafny.Seq
                            out12_, out13_, out14_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_21_openedGenerated_ = out12_
                            d_22_openedInside_ = out13_
                            d_23_openedCurrent_ = out14_
                            generated = d_21_openedGenerated_
                            insideConstrainedOut = d_22_openedInside_
                            currentConstrainedOut = d_23_openedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

