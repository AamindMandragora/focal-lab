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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve this math problem step by step. Write your reasoning in plain text. At the very end, write ONLY the final arithmetic answer expression inside << >>. Do NOT use << >> anywhere else in your solution. The expression inside << >> must be a valid Python arithmetic expression using only variable names and operators like +, -, *, **, //, %, and parentheses.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_doneConstrained_: bool
        d_2_doneConstrained_ = False
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if d_2_doneConstrained_:
                            d_3_next_: _dafny.Seq
                            out0_: _dafny.Seq
                            out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_3_next_ = out0_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_3_next_) == (eosToken):
                                raise _dafny.Break("0")
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_3_next_]))
                        elif True:
                            d_4_remainingSteps_: int
                            d_4_remainingSteps_ = (maxSteps) - (d_1_steps_)
                            if (d_4_remainingSteps_) <= (30):
                                d_5_openGenerated_: _dafny.Seq
                                d_6_openInside_: bool
                                d_7_openCurrent_: _dafny.Seq
                                out1_: _dafny.Seq
                                out2_: bool
                                out3_: _dafny.Seq
                                out1_, out2_, out3_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                d_5_openGenerated_ = out1_
                                d_6_openInside_ = out2_
                                d_7_openCurrent_ = out3_
                                generated = d_5_openGenerated_
                                insideConstrainedOut = d_6_openInside_
                                currentConstrainedOut = d_7_openCurrent_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                d_8_chunkSize_: int
                                d_8_chunkSize_ = 30
                                if (d_8_chunkSize_) > ((d_4_remainingSteps_) - (5)):
                                    if (d_4_remainingSteps_) >= (6):
                                        d_8_chunkSize_ = (d_4_remainingSteps_) - (5)
                                    elif True:
                                        d_8_chunkSize_ = 1
                                if (d_8_chunkSize_) == (0):
                                    d_8_chunkSize_ = 1
                                d_9_chunkGenerated_: _dafny.Seq
                                d_10_stoppedOnOpenSpan_: bool
                                d_11_stoppedOnEos_: bool
                                d_12_stepsUsed_: int
                                out4_: _dafny.Seq
                                out5_: bool
                                out6_: bool
                                out7_: int
                                out4_, out5_, out6_, out7_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_8_chunkSize_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                                d_9_chunkGenerated_ = out4_
                                d_10_stoppedOnOpenSpan_ = out5_
                                d_11_stoppedOnEos_ = out6_
                                d_12_stepsUsed_ = out7_
                                generated = d_9_chunkGenerated_
                                d_1_steps_ = (d_1_steps_) + (d_12_stepsUsed_)
                                if d_11_stoppedOnEos_:
                                    raise _dafny.Break("0")
                                if d_10_stoppedOnOpenSpan_:
                                    d_13_enterGenerated_: _dafny.Seq
                                    d_14_enterInside_: bool
                                    d_15_enterCurrent_: _dafny.Seq
                                    out8_: _dafny.Seq
                                    out9_: bool
                                    out10_: _dafny.Seq
                                    out8_, out9_, out10_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                    d_13_enterGenerated_ = out8_
                                    d_14_enterInside_ = out9_
                                    d_15_enterCurrent_ = out10_
                                    generated = d_13_enterGenerated_
                                    insideConstrainedOut = d_14_enterInside_
                                    currentConstrainedOut = d_15_enterCurrent_
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_16_closedGenerated_: _dafny.Seq
                        d_17_closedInside_: bool
                        d_18_closedCurrent_: _dafny.Seq
                        out11_: _dafny.Seq
                        out12_: bool
                        out13_: _dafny.Seq
                        out11_, out12_, out13_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_16_closedGenerated_ = out11_
                        d_17_closedInside_ = out12_
                        d_18_closedCurrent_ = out13_
                        generated = d_16_closedGenerated_
                        insideConstrainedOut = d_17_closedInside_
                        currentConstrainedOut = d_18_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_2_doneConstrained_ = True
                    elif True:
                        d_19_constrainedPrompt_: _dafny.Seq
                        d_19_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_20_next_: _dafny.Seq
                        d_21_wasConstrained_: bool
                        out14_: _dafny.Seq
                        out15_: bool
                        out14_, out15_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, d_19_constrainedPrompt_, currentConstrainedOut, eosToken)
                        d_20_next_ = out14_
                        d_21_wasConstrained_ = out15_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_20_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_22_appendedGenerated_: _dafny.Seq
                            d_23_appendedInside_: bool
                            d_24_appendedCurrent_: _dafny.Seq
                            out16_: _dafny.Seq
                            out17_: bool
                            out18_: _dafny.Seq
                            out16_, out17_, out18_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_20_next_)
                            d_22_appendedGenerated_ = out16_
                            d_23_appendedInside_ = out17_
                            d_24_appendedCurrent_ = out18_
                            generated = d_22_appendedGenerated_
                            insideConstrainedOut = d_23_appendedInside_
                            currentConstrainedOut = d_24_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

