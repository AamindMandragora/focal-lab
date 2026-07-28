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
        if (maxSteps) == (0):
            pass
        elif True:
            (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math problem step by step. Keep normal explanation text outside delimiters, and write each arithmetic computation inside visible << >> delimiters.")))
            d_1_steps_: int
            d_1_steps_ = 0
            d_2_openedAnySpan_: bool
            d_2_openedAnySpan_ = insideConstrained
            d_3_delayedOpenThreshold_: int
            d_3_delayedOpenThreshold_ = 24
            d_4_rollbackLimit_: int
            d_4_rollbackLimit_ = 32
            d_5_chunkCap_: int
            d_5_chunkCap_ = 12
            with _dafny.label("1_0"):
                while (d_1_steps_) < (maxSteps):
                    with _dafny.c_label("1_0"):
                        if not(insideConstrainedOut):
                            d_6_remaining_: int
                            d_6_remaining_ = (maxSteps) - (d_1_steps_)
                            if ((not(d_2_openedAnySpan_)) and ((d_1_steps_) >= (d_3_delayedOpenThreshold_))) and ((d_6_remaining_) > (0)):
                                d_7_openedGenerated_: _dafny.Seq
                                d_8_openedInside_: bool
                                d_9_openedCurrent_: _dafny.Seq
                                out0_: _dafny.Seq
                                out1_: bool
                                out2_: _dafny.Seq
                                out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                d_7_openedGenerated_ = out0_
                                d_8_openedInside_ = out1_
                                d_9_openedCurrent_ = out2_
                                generated = d_7_openedGenerated_
                                insideConstrainedOut = d_8_openedInside_
                                currentConstrainedOut = d_9_openedCurrent_
                                d_2_openedAnySpan_ = True
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                d_10_chunkBudget_: int
                                if (d_6_remaining_) < (d_5_chunkCap_):
                                    d_10_chunkBudget_ = d_6_remaining_
                                elif True:
                                    d_10_chunkBudget_ = d_5_chunkCap_
                                d_11_chunkedGenerated_: _dafny.Seq
                                d_12_stoppedOnOpenSpan_: bool
                                d_13_stoppedOnEos_: bool
                                d_14_stepsUsed_: int
                                out3_: _dafny.Seq
                                out4_: bool
                                out5_: bool
                                out6_: int
                                out3_, out4_, out5_, out6_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_10_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                                d_11_chunkedGenerated_ = out3_
                                d_12_stoppedOnOpenSpan_ = out4_
                                d_13_stoppedOnEos_ = out5_
                                d_14_stepsUsed_ = out6_
                                generated = d_11_chunkedGenerated_
                                d_1_steps_ = (d_1_steps_) + (d_14_stepsUsed_)
                                if d_13_stoppedOnEos_:
                                    raise _dafny.Break("1_0")
                                elif d_12_stoppedOnOpenSpan_:
                                    d_15_observedGenerated_: _dafny.Seq
                                    d_16_observedInside_: bool
                                    d_17_observedCurrent_: _dafny.Seq
                                    out7_: _dafny.Seq
                                    out8_: bool
                                    out9_: _dafny.Seq
                                    out7_, out8_, out9_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                    d_15_observedGenerated_ = out7_
                                    d_16_observedInside_ = out8_
                                    d_17_observedCurrent_ = out9_
                                    generated = d_15_observedGenerated_
                                    insideConstrainedOut = d_16_observedInside_
                                    currentConstrainedOut = d_17_observedCurrent_
                                    d_2_openedAnySpan_ = True
                        elif (parser).IsCompletePrefix(currentConstrainedOut):
                            d_18_closedGenerated_: _dafny.Seq
                            d_19_closedInside_: bool
                            d_20_closedCurrent_: _dafny.Seq
                            out10_: _dafny.Seq
                            out11_: bool
                            out12_: _dafny.Seq
                            out10_, out11_, out12_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_18_closedGenerated_ = out10_
                            d_19_closedInside_ = out11_
                            d_20_closedCurrent_ = out12_
                            generated = d_18_closedGenerated_
                            insideConstrainedOut = d_19_closedInside_
                            currentConstrainedOut = d_20_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif (len(currentConstrainedOut)) >= (d_4_rollbackLimit_):
                            d_21_rolledGenerated_: _dafny.Seq
                            d_22_rolledCurrent_: _dafny.Seq
                            out13_: _dafny.Seq
                            out14_: _dafny.Seq
                            out13_, out14_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                            d_21_rolledGenerated_ = out13_
                            d_22_rolledCurrent_ = out14_
                            generated = d_21_rolledGenerated_
                            insideConstrainedOut = True
                            currentConstrainedOut = d_22_rolledCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_23_constrainedPrompt_: _dafny.Seq
                            d_23_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_24_next_: _dafny.Seq
                            out15_: _dafny.Seq
                            out15_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_23_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_24_next_ = out15_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_24_next_) == (eosToken):
                                raise _dafny.Break("1_0")
                            elif True:
                                d_25_appendedGenerated_: _dafny.Seq
                                d_26_appendedInside_: bool
                                d_27_appendedCurrent_: _dafny.Seq
                                out16_: _dafny.Seq
                                out17_: bool
                                out18_: _dafny.Seq
                                out16_, out17_, out18_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_24_next_)
                                d_25_appendedGenerated_ = out16_
                                d_26_appendedInside_ = out17_
                                d_27_appendedCurrent_ = out18_
                                generated = d_25_appendedGenerated_
                                insideConstrainedOut = d_26_appendedInside_
                                currentConstrainedOut = d_27_appendedCurrent_
                        pass
                pass
            cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

