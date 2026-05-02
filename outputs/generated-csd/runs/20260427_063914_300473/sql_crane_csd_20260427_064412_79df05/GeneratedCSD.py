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
        generated = generatedPrefix
        insideConstrainedOut = insideConstrained
        currentConstrainedOut = currentConstrained
        cost = 0
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_openSpanToken_: _dafny.Seq
        d_2_openSpanToken_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_3_remaining_: int
                        d_3_remaining_ = (maxSteps) - (d_1_steps_)
                        d_4_chunkGenerated_: _dafny.Seq
                        d_5_stoppedOnOpenSpan_: bool
                        d_6_stoppedOnEos_: bool
                        d_7_stepsUsed_: int
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: bool
                        out3_: int
                        out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_3_remaining_, d_2_openSpanToken_, eosToken)
                        d_4_chunkGenerated_ = out0_
                        d_5_stoppedOnOpenSpan_ = out1_
                        d_6_stoppedOnEos_ = out2_
                        d_7_stepsUsed_ = out3_
                        generated = d_4_chunkGenerated_
                        d_1_steps_ = (d_1_steps_) + (d_7_stepsUsed_)
                        if d_6_stoppedOnEos_:
                            raise _dafny.Break("0")
                        elif True:
                            if (d_5_stoppedOnOpenSpan_) and ((d_1_steps_) < (maxSteps)):
                                d_8_openedGenerated_: _dafny.Seq
                                d_9_openedInside_: bool
                                d_10_openedCurrent_: _dafny.Seq
                                out4_: _dafny.Seq
                                out5_: bool
                                out6_: _dafny.Seq
                                out4_, out5_, out6_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                d_8_openedGenerated_ = out4_
                                d_9_openedInside_ = out5_
                                d_10_openedCurrent_ = out6_
                                generated = d_8_openedGenerated_
                                insideConstrainedOut = d_9_openedInside_
                                currentConstrainedOut = d_10_openedCurrent_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                raise _dafny.Break("0")
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
                            d_15_validCount_: int
                            out10_: int
                            out10_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                            d_15_validCount_ = out10_
                            d_16_remainingInside_: int
                            d_16_remainingInside_ = (maxSteps) - (d_1_steps_)
                            if ((d_15_validCount_) <= (2)) and ((0) < (d_16_remainingInside_)):
                                d_17_stablePrefix_: _dafny.Seq
                                d_17_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                d_18_constrainedPrompt_: _dafny.Seq
                                d_18_constrainedPrompt_ = (prompt) + (d_17_stablePrefix_)
                                d_19_symbolCurrent_: _dafny.Seq
                                d_20_hitEos_: bool
                                d_21_symbolSteps_: int
                                out11_: _dafny.Seq
                                out12_: bool
                                out13_: int
                                out11_, out12_, out13_ = (d_0_helpers_).ConstrainedSymbol(lm, parser, d_18_constrainedPrompt_, currentConstrainedOut, d_16_remainingInside_, eosToken)
                                d_19_symbolCurrent_ = out11_
                                d_20_hitEos_ = out12_
                                d_21_symbolSteps_ = out13_
                                d_22_oldLen_: int
                                d_22_oldLen_ = len(currentConstrainedOut)
                                currentConstrainedOut = d_19_symbolCurrent_
                                generated = (d_17_stablePrefix_) + (currentConstrainedOut)
                                d_1_steps_ = (d_1_steps_) + (d_21_symbolSteps_)
                                if d_20_hitEos_:
                                    raise _dafny.Break("0")
                            elif True:
                                d_23_stablePrefix2_: _dafny.Seq
                                d_23_stablePrefix2_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                d_24_constrainedPrompt2_: _dafny.Seq
                                d_24_constrainedPrompt2_ = (prompt) + (d_23_stablePrefix2_)
                                d_25_next_: _dafny.Seq
                                out14_: _dafny.Seq
                                out14_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_24_constrainedPrompt2_, currentConstrainedOut, eosToken)
                                d_25_next_ = out14_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_25_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_26_appendedGenerated_: _dafny.Seq
                                    d_27_appendedInside_: bool
                                    d_28_appendedCurrent_: _dafny.Seq
                                    out15_: _dafny.Seq
                                    out16_: bool
                                    out17_: _dafny.Seq
                                    out15_, out16_, out17_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_25_next_)
                                    d_26_appendedGenerated_ = out15_
                                    d_27_appendedInside_ = out16_
                                    d_28_appendedCurrent_ = out17_
                                    generated = d_26_appendedGenerated_
                                    insideConstrainedOut = d_27_appendedInside_
                                    currentConstrainedOut = d_28_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

