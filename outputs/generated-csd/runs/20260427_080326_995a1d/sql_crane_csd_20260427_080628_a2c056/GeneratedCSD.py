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
                        elif True:
                            if d_4_stoppedOnOpenSpan_:
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                            elif True:
                                raise _dafny.Break("0")
                    elif True:
                        d_7_remaining_: int
                        d_7_remaining_ = (maxSteps) - (d_1_steps_)
                        d_8_completeNow_: bool
                        d_8_completeNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_8_completeNow_:
                            if (d_7_remaining_) >= (1):
                                d_9_closedGenerated_: _dafny.Seq
                                d_10_closedInside_: bool
                                d_11_closedCurrent_: _dafny.Seq
                                out4_: _dafny.Seq
                                out5_: bool
                                out6_: _dafny.Seq
                                out4_, out5_, out6_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_9_closedGenerated_ = out4_
                                d_10_closedInside_ = out5_
                                d_11_closedCurrent_ = out6_
                                generated = d_9_closedGenerated_
                                insideConstrainedOut = d_10_closedInside_
                                currentConstrainedOut = d_11_closedCurrent_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                raise _dafny.Break("0")
                        elif True:
                            if (d_7_remaining_) <= (1):
                                raise _dafny.Break("0")
                            elif True:
                                d_12_deadEnd_: bool
                                out7_: bool
                                out7_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                                d_12_deadEnd_ = out7_
                                if d_12_deadEnd_:
                                    d_13_stablePrefixDead_: _dafny.Seq
                                    d_13_stablePrefixDead_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                    d_14_rolledGenerated_: _dafny.Seq
                                    d_15_rolledCurrent_: _dafny.Seq
                                    out8_: _dafny.Seq
                                    out9_: _dafny.Seq
                                    out8_, out9_ = (d_0_helpers_).RollbackConstrainedSpan(parser, d_13_stablePrefixDead_, generated, currentConstrainedOut)
                                    d_14_rolledGenerated_ = out8_
                                    d_15_rolledCurrent_ = out9_
                                    generated = d_14_rolledGenerated_
                                    currentConstrainedOut = d_15_rolledCurrent_
                                    d_16_completeAfterRollback_: bool
                                    d_16_completeAfterRollback_ = (parser).IsCompletePrefix(currentConstrainedOut)
                                    if d_16_completeAfterRollback_:
                                        if (d_1_steps_) < (maxSteps):
                                            d_17_closedGenerated2_: _dafny.Seq
                                            d_18_closedInside2_: bool
                                            d_19_closedCurrent2_: _dafny.Seq
                                            out10_: _dafny.Seq
                                            out11_: bool
                                            out12_: _dafny.Seq
                                            out10_, out11_, out12_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                            d_17_closedGenerated2_ = out10_
                                            d_18_closedInside2_ = out11_
                                            d_19_closedCurrent2_ = out12_
                                            generated = d_17_closedGenerated2_
                                            insideConstrainedOut = d_18_closedInside2_
                                            currentConstrainedOut = d_19_closedCurrent2_
                                            d_1_steps_ = (d_1_steps_) + (1)
                                        elif True:
                                            raise _dafny.Break("0")
                                    elif True:
                                        raise _dafny.Break("0")
                                elif True:
                                    d_20_stablePrefix_: _dafny.Seq
                                    d_20_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                    d_21_constrainedPrompt_: _dafny.Seq
                                    d_21_constrainedPrompt_ = (prompt) + (d_20_stablePrefix_)
                                    d_22_next_: _dafny.Seq
                                    out13_: _dafny.Seq
                                    out13_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_21_constrainedPrompt_, currentConstrainedOut, eosToken)
                                    d_22_next_ = out13_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    if (d_22_next_) == (eosToken):
                                        raise _dafny.Break("0")
                                    elif True:
                                        d_23_appendedGenerated_: _dafny.Seq
                                        d_24_appendedInside_: bool
                                        d_25_appendedCurrent_: _dafny.Seq
                                        out14_: _dafny.Seq
                                        out15_: bool
                                        out16_: _dafny.Seq
                                        out14_, out15_, out16_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_22_next_)
                                        d_23_appendedGenerated_ = out14_
                                        d_24_appendedInside_ = out15_
                                        d_25_appendedCurrent_ = out16_
                                        generated = d_23_appendedGenerated_
                                        insideConstrainedOut = d_24_appendedInside_
                                        currentConstrainedOut = d_25_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

