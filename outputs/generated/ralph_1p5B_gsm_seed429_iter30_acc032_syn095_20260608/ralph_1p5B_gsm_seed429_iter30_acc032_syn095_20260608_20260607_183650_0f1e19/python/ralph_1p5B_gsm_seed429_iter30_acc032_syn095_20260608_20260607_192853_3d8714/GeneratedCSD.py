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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve this math problem step by step. Use << >> to mark arithmetic expressions. Inside << >>, write ONLY valid Python arithmetic using the EXACT variable names that appear in the problem statement. Use only: +, -, *, **, //, %, (, ), and number literals. Do NOT use curly braces like {n1}. Do NOT call any functions like round(), int(), abs(), max(), min(), len(). Do NOT create your own variable names like 'remaining' or 'total' inside << >>. Use only the problem's own variable names. Example correct: <<n1 * p1 + n2 * p2>>. Example WRONG: <<round(n1 * p1)>> or <<remaining * p2 / 100>>.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_freeChunkSize_: int
        d_2_freeChunkSize_ = 25
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_3_remainingSteps_: int
                        d_3_remainingSteps_ = (maxSteps) - (d_1_steps_)
                        if ((d_3_remainingSteps_) >= (3)) and ((d_3_remainingSteps_) <= (50)):
                            d_4_openGenerated_: _dafny.Seq
                            d_5_openInside_: bool
                            d_6_openCurrent_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_4_openGenerated_ = out0_
                            d_5_openInside_ = out1_
                            d_6_openCurrent_ = out2_
                            generated = d_4_openGenerated_
                            insideConstrainedOut = d_5_openInside_
                            currentConstrainedOut = d_6_openCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_7_chunkSize_: int
                            d_7_chunkSize_ = d_2_freeChunkSize_
                            if (d_7_chunkSize_) > (d_3_remainingSteps_):
                                d_7_chunkSize_ = d_3_remainingSteps_
                            if (d_7_chunkSize_) == (0):
                                raise _dafny.Break("0")
                            d_8_chunkGenerated_: _dafny.Seq
                            d_9_stoppedOnOpenSpan_: bool
                            d_10_stoppedOnEos_: bool
                            d_11_stepsUsed_: int
                            out3_: _dafny.Seq
                            out4_: bool
                            out5_: bool
                            out6_: int
                            out3_, out4_, out5_, out6_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_7_chunkSize_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_8_chunkGenerated_ = out3_
                            d_9_stoppedOnOpenSpan_ = out4_
                            d_10_stoppedOnEos_ = out5_
                            d_11_stepsUsed_ = out6_
                            generated = d_8_chunkGenerated_
                            d_1_steps_ = (d_1_steps_) + (d_11_stepsUsed_)
                            if d_10_stoppedOnEos_:
                                raise _dafny.Break("0")
                            if d_9_stoppedOnOpenSpan_:
                                d_12_enterGenerated_: _dafny.Seq
                                d_13_enterInside_: bool
                                d_14_enterCurrent_: _dafny.Seq
                                out7_: _dafny.Seq
                                out8_: bool
                                out9_: _dafny.Seq
                                out7_, out8_, out9_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_12_enterGenerated_ = out7_
                                d_13_enterInside_ = out8_
                                d_14_enterCurrent_ = out9_
                                generated = d_12_enterGenerated_
                                insideConstrainedOut = d_13_enterInside_
                                currentConstrainedOut = d_14_enterCurrent_
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_15_closedGenerated_: _dafny.Seq
                        d_16_closedInside_: bool
                        d_17_closedCurrent_: _dafny.Seq
                        out10_: _dafny.Seq
                        out11_: bool
                        out12_: _dafny.Seq
                        out10_, out11_, out12_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_15_closedGenerated_ = out10_
                        d_16_closedInside_ = out11_
                        d_17_closedCurrent_ = out12_
                        generated = d_15_closedGenerated_
                        insideConstrainedOut = d_16_closedInside_
                        currentConstrainedOut = d_17_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_18_constrainedPrompt_: _dafny.Seq
                        d_18_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_19_next_: _dafny.Seq
                        d_20_wasConstrained_: bool
                        out13_: _dafny.Seq
                        out14_: bool
                        out13_, out14_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, d_18_constrainedPrompt_, currentConstrainedOut, eosToken)
                        d_19_next_ = out13_
                        d_20_wasConstrained_ = out14_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_19_next_) == (eosToken):
                            if (len(currentConstrainedOut)) > (0):
                                d_21_rolledGenerated_: _dafny.Seq
                                d_22_rolledCurrent_: _dafny.Seq
                                out15_: _dafny.Seq
                                out16_: _dafny.Seq
                                out15_, out16_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                d_21_rolledGenerated_ = out15_
                                d_22_rolledCurrent_ = out16_
                                generated = d_21_rolledGenerated_
                                currentConstrainedOut = d_22_rolledCurrent_
                                if not((parser).IsCompletePrefix(currentConstrainedOut)):
                                    raise _dafny.Break("0")
                            elif True:
                                raise _dafny.Break("0")
                        elif (((((d_19_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "{")))) or ((d_19_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "!"))))) or ((d_19_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "round"))))) or ((d_19_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "remaining"))))) or ((d_19_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "int")))):
                            if (parser).IsCompletePrefix(currentConstrainedOut):
                                pass
                            elif (len(currentConstrainedOut)) > (0):
                                d_23_rolledGenerated_: _dafny.Seq
                                d_24_rolledCurrent_: _dafny.Seq
                                out17_: _dafny.Seq
                                out18_: _dafny.Seq
                                out17_, out18_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                d_23_rolledGenerated_ = out17_
                                d_24_rolledCurrent_ = out18_
                                generated = d_23_rolledGenerated_
                                currentConstrainedOut = d_24_rolledCurrent_
                        elif True:
                            d_25_appendedGenerated_: _dafny.Seq
                            d_26_appendedInside_: bool
                            d_27_appendedCurrent_: _dafny.Seq
                            out19_: _dafny.Seq
                            out20_: bool
                            out21_: _dafny.Seq
                            out19_, out20_, out21_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_19_next_)
                            d_25_appendedGenerated_ = out19_
                            d_26_appendedInside_ = out20_
                            d_27_appendedCurrent_ = out21_
                            generated = d_25_appendedGenerated_
                            insideConstrainedOut = d_26_appendedInside_
                            currentConstrainedOut = d_27_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

