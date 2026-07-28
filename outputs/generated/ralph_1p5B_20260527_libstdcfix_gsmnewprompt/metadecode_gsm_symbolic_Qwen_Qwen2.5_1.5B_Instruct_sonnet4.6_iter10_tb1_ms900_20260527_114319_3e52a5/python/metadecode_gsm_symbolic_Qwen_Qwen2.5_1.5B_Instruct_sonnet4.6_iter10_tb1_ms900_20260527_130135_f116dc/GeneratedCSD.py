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
        (d_0_helpers_).AppendTaskGuidance(lm, ((((((((((_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Read the ACTUAL problem above carefully and solve it step by step. "))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "For each arithmetic calculation, wrap ONLY the expression in << >> delimiters. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "IMPORTANT: Not every {variable} in the problem is a number. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Currency symbols like {cur} (which becomes $ or euros) are NOT numbers - do NOT include them in formulas. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Category names, labels, and unit strings (like {s1}, {s2}, {obj1}) are NOT numbers - do NOT use them in arithmetic. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Only use {variables} that represent actual numeric quantities (counts, prices, rates, times, distances, percentages). ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Use // for integer floor division when the result must be a whole number: <<total // n>>. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Use int() to convert a float to integer: <<int(n * frac)>>. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Use * for multiplication, + for addition, - for subtraction, / for exact division, ** for exponentiation. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "NEVER put an equals sign, assignment, or sentence text inside << >>. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Your final answer line must be exactly: #### <<expression>>"))))
        d_1_effectiveMax_: int
        if (maxSteps) > (850):
            d_1_effectiveMax_ = 850
        elif True:
            d_1_effectiveMax_ = maxSteps
        d_2_steps_: int
        d_2_steps_ = 0
        with _dafny.label("0"):
            while (d_2_steps_) < (d_1_effectiveMax_):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_3_remaining_: int
                        d_3_remaining_ = (d_1_effectiveMax_) - (d_2_steps_)
                        d_4_chunkSize_: int
                        if (d_3_remaining_) > (60):
                            d_4_chunkSize_ = 60
                        elif True:
                            d_4_chunkSize_ = d_3_remaining_
                        d_5_genOut_: _dafny.Seq
                        d_6_stoppedOnOpen_: bool
                        d_7_stoppedOnEos_: bool
                        d_8_stepsUsed_: int
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: bool
                        out3_: int
                        out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_4_chunkSize_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_5_genOut_ = out0_
                        d_6_stoppedOnOpen_ = out1_
                        d_7_stoppedOnEos_ = out2_
                        d_8_stepsUsed_ = out3_
                        d_2_steps_ = (d_2_steps_) + (d_8_stepsUsed_)
                        generated = d_5_genOut_
                        if d_7_stoppedOnEos_:
                            raise _dafny.Break("0")
                        elif d_6_stoppedOnOpen_:
                            d_9_newGen_: _dafny.Seq
                            d_10_newInside_: bool
                            d_11_newCurrent_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                            d_9_newGen_ = out4_
                            d_10_newInside_ = out5_
                            d_11_newCurrent_ = out6_
                            generated = d_9_newGen_
                            insideConstrainedOut = d_10_newInside_
                            currentConstrainedOut = d_11_newCurrent_
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
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
                        d_2_steps_ = (d_2_steps_) + (1)
                    elif True:
                        d_15_deadEnd_: bool
                        out10_: bool
                        out10_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 0)
                        d_15_deadEnd_ = out10_
                        if d_15_deadEnd_:
                            d_16_rolledGen_: _dafny.Seq
                            d_17_rolledCurrent_: _dafny.Seq
                            out11_: _dafny.Seq
                            out12_: _dafny.Seq
                            out11_, out12_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                            d_16_rolledGen_ = out11_
                            d_17_rolledCurrent_ = out12_
                            generated = d_16_rolledGen_
                            currentConstrainedOut = d_17_rolledCurrent_
                            d_2_steps_ = (d_2_steps_) + (1)
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
                            d_2_steps_ = (d_2_steps_) + (1)
                            if (d_19_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_21_appendedGenerated_: _dafny.Seq
                                d_22_appendedInside_: bool
                                d_23_appendedCurrent_: _dafny.Seq
                                out15_: _dafny.Seq
                                out16_: bool
                                out17_: _dafny.Seq
                                out15_, out16_, out17_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_19_next_)
                                d_21_appendedGenerated_ = out15_
                                d_22_appendedInside_ = out16_
                                d_23_appendedCurrent_ = out17_
                                generated = d_21_appendedGenerated_
                                insideConstrainedOut = d_22_appendedInside_
                                currentConstrainedOut = d_23_appendedCurrent_
                    pass
            pass
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

