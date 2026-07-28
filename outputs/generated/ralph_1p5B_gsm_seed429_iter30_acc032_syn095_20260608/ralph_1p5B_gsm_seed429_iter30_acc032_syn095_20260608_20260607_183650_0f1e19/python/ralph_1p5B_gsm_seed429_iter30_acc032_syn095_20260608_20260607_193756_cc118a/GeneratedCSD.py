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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve this math problem step by step. When writing arithmetic inside << >>, use ONLY: variable names without curly braces, numbers, and operators +, -, *, **, //, %, (, ). Do NOT write {variable} inside << >>. Do NOT use ! or function calls like round(), int(), abs() inside << >>. Write the variable name directly without braces.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_freeChunkSize_: int
        d_2_freeChunkSize_ = 25
        d_3_eosCount_: int
        d_3_eosCount_ = 0
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_4_remainingSteps_: int
                        d_4_remainingSteps_ = (maxSteps) - (d_1_steps_)
                        if ((d_4_remainingSteps_) <= (50)) and ((d_4_remainingSteps_) >= (3)):
                            d_5_openGenerated_: _dafny.Seq
                            d_6_openInside_: bool
                            d_7_openCurrent_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_5_openGenerated_ = out0_
                            d_6_openInside_ = out1_
                            d_7_openCurrent_ = out2_
                            generated = d_5_openGenerated_
                            insideConstrainedOut = d_6_openInside_
                            currentConstrainedOut = d_7_openCurrent_
                            d_3_eosCount_ = 0
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif (d_4_remainingSteps_) == (0):
                            raise _dafny.Break("0")
                        elif True:
                            d_8_chunkSize_: int
                            d_8_chunkSize_ = d_2_freeChunkSize_
                            if (d_8_chunkSize_) > (d_4_remainingSteps_):
                                d_8_chunkSize_ = d_4_remainingSteps_
                            if (d_8_chunkSize_) == (0):
                                raise _dafny.Break("0")
                            d_9_chunkGenerated_: _dafny.Seq
                            d_10_stoppedOnOpenSpan_: bool
                            d_11_stoppedOnEos_: bool
                            d_12_stepsUsed_: int
                            out3_: _dafny.Seq
                            out4_: bool
                            out5_: bool
                            out6_: int
                            out3_, out4_, out5_, out6_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_8_chunkSize_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_9_chunkGenerated_ = out3_
                            d_10_stoppedOnOpenSpan_ = out4_
                            d_11_stoppedOnEos_ = out5_
                            d_12_stepsUsed_ = out6_
                            generated = d_9_chunkGenerated_
                            d_1_steps_ = (d_1_steps_) + (d_12_stepsUsed_)
                            if d_11_stoppedOnEos_:
                                raise _dafny.Break("0")
                            if d_10_stoppedOnOpenSpan_:
                                d_13_enterGenerated_: _dafny.Seq
                                d_14_enterInside_: bool
                                d_15_enterCurrent_: _dafny.Seq
                                out7_: _dafny.Seq
                                out8_: bool
                                out9_: _dafny.Seq
                                out7_, out8_, out9_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_13_enterGenerated_ = out7_
                                d_14_enterInside_ = out8_
                                d_15_enterCurrent_ = out9_
                                generated = d_13_enterGenerated_
                                insideConstrainedOut = d_14_enterInside_
                                currentConstrainedOut = d_15_enterCurrent_
                                d_3_eosCount_ = 0
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_16_closedGenerated_: _dafny.Seq
                        d_17_closedInside_: bool
                        d_18_closedCurrent_: _dafny.Seq
                        out10_: _dafny.Seq
                        out11_: bool
                        out12_: _dafny.Seq
                        out10_, out11_, out12_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_16_closedGenerated_ = out10_
                        d_17_closedInside_ = out11_
                        d_18_closedCurrent_ = out12_
                        generated = d_16_closedGenerated_
                        insideConstrainedOut = d_17_closedInside_
                        currentConstrainedOut = d_18_closedCurrent_
                        d_3_eosCount_ = 0
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_19_constrainedPrompt_: _dafny.Seq
                        d_19_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_20_next_: _dafny.Seq
                        d_21_wasConstrained_: bool
                        out13_: _dafny.Seq
                        out14_: bool
                        out13_, out14_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, d_19_constrainedPrompt_, currentConstrainedOut, eosToken)
                        d_20_next_ = out13_
                        d_21_wasConstrained_ = out14_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_20_next_) == (eosToken):
                            d_3_eosCount_ = (d_3_eosCount_) + (1)
                            if (d_3_eosCount_) >= (3):
                                raise _dafny.Break("0")
                            if (len(currentConstrainedOut)) > (0):
                                d_22_rolledGenerated_: _dafny.Seq
                                d_23_rolledCurrent_: _dafny.Seq
                                out15_: _dafny.Seq
                                out16_: _dafny.Seq
                                out15_, out16_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                d_22_rolledGenerated_ = out15_
                                d_23_rolledCurrent_ = out16_
                                generated = d_22_rolledGenerated_
                                currentConstrainedOut = d_23_rolledCurrent_
                        elif (((d_20_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "{")))) or ((d_20_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "!"))))) or ((d_20_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "round")))):
                            d_3_eosCount_ = 0
                            if (parser).IsCompletePrefix(currentConstrainedOut):
                                pass
                            elif (len(currentConstrainedOut)) > (0):
                                d_24_rolledGenerated_: _dafny.Seq
                                d_25_rolledCurrent_: _dafny.Seq
                                out17_: _dafny.Seq
                                out18_: _dafny.Seq
                                out17_, out18_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                d_24_rolledGenerated_ = out17_
                                d_25_rolledCurrent_ = out18_
                                generated = d_24_rolledGenerated_
                                currentConstrainedOut = d_25_rolledCurrent_
                        elif True:
                            d_3_eosCount_ = 0
                            d_26_appendedGenerated_: _dafny.Seq
                            d_27_appendedInside_: bool
                            d_28_appendedCurrent_: _dafny.Seq
                            out19_: _dafny.Seq
                            out20_: bool
                            out21_: _dafny.Seq
                            out19_, out20_, out21_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_20_next_)
                            d_26_appendedGenerated_ = out19_
                            d_27_appendedInside_ = out20_
                            d_28_appendedCurrent_ = out21_
                            generated = d_26_appendedGenerated_
                            insideConstrainedOut = d_27_appendedInside_
                            currentConstrainedOut = d_28_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

