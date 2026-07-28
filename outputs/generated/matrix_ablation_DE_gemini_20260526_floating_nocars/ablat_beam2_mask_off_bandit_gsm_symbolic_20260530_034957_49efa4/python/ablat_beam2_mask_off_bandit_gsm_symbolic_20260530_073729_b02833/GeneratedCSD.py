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
        d_1_guidance_: _dafny.Seq
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step. Wrap every intermediate symbolic arithmetic expression and the final answer inside visible << >> delimiters. Keep text outside delimiters concise. Inside each << >> span write only the symbolic expression or number: no words, no units, no LaTeX, no nested delimiters. Prefer plain arithmetic with digits, variables from the problem, +, -, *, /, //, and parentheses. Finish with the final answer in its own << >> span."))
        if (len(validTokenGroups)) > (0):
            d_1_guidance_ = (d_1_guidance_) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " Prefer caller-provided valid token groups when they match numbers, variables, or operators in the problem.")))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        if (maxSteps) == (0):
            cost = 0
            return generated, insideConstrainedOut, currentConstrainedOut, cost
        d_2_arithmeticGroups_: _dafny.Seq
        d_2_arithmeticGroups_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "0")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "1")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "2")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "3")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "4")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "5")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "6")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "7")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "8")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "9"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "+")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "-")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "*")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "/")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "//")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "(")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ")"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "n")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "x")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "y")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "m")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "k")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "t")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "d")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "p")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "r")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "total")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "answer"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "n1")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "n2")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "n3")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "n4")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "w1")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "w2")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "w3")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "w4")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "t1")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "t2")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "p1")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "r1"))])])
        d_3_penaltyTokens_: _dafny.Seq
        d_3_penaltyTokens_ = _dafny.SeqWithoutIsStrInference([eosToken, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<answer>")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "</answer>")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "int")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "%")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "ceil")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\\")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "$"))])
        d_4_reasonLimit_: int
        d_4_reasonLimit_ = 96
        d_5_reasonTokens_: int
        d_5_reasonTokens_ = 0
        d_6_sawAnswerCue_: bool
        d_6_sawAnswerCue_ = False
        d_7_forceFinal_: bool
        d_7_forceFinal_ = False
        d_8_done_: bool
        d_8_done_ = False
        d_9_steps_: int
        d_9_steps_ = 0
        if not(insideConstrainedOut):
            d_10_next0_: _dafny.Seq
            out0_: _dafny.Seq
            out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
            d_10_next0_ = out0_
            d_9_steps_ = 1
            if (d_10_next0_) == (eosToken):
                d_7_forceFinal_ = True
            elif (d_10_next0_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_10_next0_]))
                insideConstrainedOut = True
                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                d_7_forceFinal_ = False
            elif (((((d_10_next0_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")))) or ((d_10_next0_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<"))))) or ((d_10_next0_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">"))))) or ((d_10_next0_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<answer>"))))) or ((d_10_next0_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "</answer>")))):
                d_5_reasonTokens_ = (d_5_reasonTokens_) + (1)
            elif True:
                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_10_next0_]))
                d_5_reasonTokens_ = (d_5_reasonTokens_) + (1)
                if ((((((((d_10_next0_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Final")))) or ((d_10_next0_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "final"))))) or ((d_10_next0_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Answer"))))) or ((d_10_next0_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "answer"))))) or ((d_10_next0_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Therefore"))))) or ((d_10_next0_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "therefore"))))) or ((d_10_next0_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Thus"))))) or ((d_10_next0_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "thus")))):
                    d_6_sawAnswerCue_ = True
        elif (parser).IsCompletePrefix(currentConstrainedOut):
            d_11_closedGenerated0_: _dafny.Seq
            d_12_closedInside0_: bool
            d_13_closedCurrent0_: _dafny.Seq
            out1_: _dafny.Seq
            out2_: bool
            out3_: _dafny.Seq
            out1_, out2_, out3_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
            d_11_closedGenerated0_ = out1_
            d_12_closedInside0_ = out2_
            d_13_closedCurrent0_ = out3_
            generated = d_11_closedGenerated0_
            insideConstrainedOut = d_12_closedInside0_
            currentConstrainedOut = d_13_closedCurrent0_
            d_9_steps_ = 1
            d_8_done_ = True
        elif True:
            d_14_constrainedPrompt0_: _dafny.Seq
            d_14_constrainedPrompt0_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
            d_15_groups0_: _dafny.Seq
            d_15_groups0_ = (d_2_arithmeticGroups_) + (validTokenGroups)
            d_16_nextConstrained0_: _dafny.Seq
            out4_: _dafny.Seq
            out4_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_14_constrainedPrompt0_, currentConstrainedOut, d_15_groups0_, _dafny.BigRational('5e0'), d_3_penaltyTokens_, _dafny.BigRational('7e0'), 24, eosToken)
            d_16_nextConstrained0_ = out4_
            d_9_steps_ = 1
            if (d_16_nextConstrained0_) == (eosToken):
                d_7_forceFinal_ = True
            elif True:
                d_17_appendedGenerated0_: _dafny.Seq
                d_18_appendedInside0_: bool
                d_19_appendedCurrent0_: _dafny.Seq
                out5_: _dafny.Seq
                out6_: bool
                out7_: _dafny.Seq
                out5_, out6_, out7_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_16_nextConstrained0_)
                d_17_appendedGenerated0_ = out5_
                d_18_appendedInside0_ = out6_
                d_19_appendedCurrent0_ = out7_
                generated = d_17_appendedGenerated0_
                insideConstrainedOut = d_18_appendedInside0_
                currentConstrainedOut = d_19_appendedCurrent0_
        while ((d_9_steps_) < (maxSteps)) and (not(d_8_done_)):
            if not(insideConstrainedOut):
                d_20_shouldOpen_: bool
                d_20_shouldOpen_ = (((d_7_forceFinal_) or ((d_5_reasonTokens_) >= (d_4_reasonLimit_))) or ((d_6_sawAnswerCue_) and ((d_5_reasonTokens_) >= (16)))) or (((d_9_steps_) + (96)) >= (maxSteps))
                if d_20_shouldOpen_:
                    d_21_openedGenerated_: _dafny.Seq
                    d_22_openedInside_: bool
                    d_23_openedCurrent_: _dafny.Seq
                    out8_: _dafny.Seq
                    out9_: bool
                    out10_: _dafny.Seq
                    out8_, out9_, out10_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                    d_21_openedGenerated_ = out8_
                    d_22_openedInside_ = out9_
                    d_23_openedCurrent_ = out10_
                    generated = d_21_openedGenerated_
                    insideConstrainedOut = d_22_openedInside_
                    currentConstrainedOut = d_23_openedCurrent_
                    d_7_forceFinal_ = False
                    d_9_steps_ = (d_9_steps_) + (1)
                elif True:
                    d_24_next_: _dafny.Seq
                    out11_: _dafny.Seq
                    out11_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                    d_24_next_ = out11_
                    d_9_steps_ = (d_9_steps_) + (1)
                    if (d_24_next_) == (eosToken):
                        d_7_forceFinal_ = True
                    elif (d_24_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_24_next_]))
                        insideConstrainedOut = True
                        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                        d_7_forceFinal_ = False
                    elif (((((d_24_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")))) or ((d_24_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<"))))) or ((d_24_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">"))))) or ((d_24_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<answer>"))))) or ((d_24_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "</answer>")))):
                        d_5_reasonTokens_ = (d_5_reasonTokens_) + (1)
                    elif True:
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_24_next_]))
                        d_5_reasonTokens_ = (d_5_reasonTokens_) + (1)
                        if ((((((((d_24_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Final")))) or ((d_24_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "final"))))) or ((d_24_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Answer"))))) or ((d_24_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "answer"))))) or ((d_24_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Therefore"))))) or ((d_24_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "therefore"))))) or ((d_24_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Thus"))))) or ((d_24_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "thus")))):
                            d_6_sawAnswerCue_ = True
            elif (parser).IsCompletePrefix(currentConstrainedOut):
                d_25_closedGenerated_: _dafny.Seq
                d_26_closedInside_: bool
                d_27_closedCurrent_: _dafny.Seq
                out12_: _dafny.Seq
                out13_: bool
                out14_: _dafny.Seq
                out12_, out13_, out14_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_25_closedGenerated_ = out12_
                d_26_closedInside_ = out13_
                d_27_closedCurrent_ = out14_
                generated = d_25_closedGenerated_
                insideConstrainedOut = d_26_closedInside_
                currentConstrainedOut = d_27_closedCurrent_
                d_9_steps_ = (d_9_steps_) + (1)
                d_8_done_ = True
            elif True:
                d_28_constrainedPrompt_: _dafny.Seq
                d_28_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                d_29_groups_: _dafny.Seq
                d_29_groups_ = (d_2_arithmeticGroups_) + (validTokenGroups)
                d_30_nextConstrained_: _dafny.Seq
                out15_: _dafny.Seq
                out15_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_28_constrainedPrompt_, currentConstrainedOut, d_29_groups_, _dafny.BigRational('5e0'), d_3_penaltyTokens_, _dafny.BigRational('7e0'), 24, eosToken)
                d_30_nextConstrained_ = out15_
                d_9_steps_ = (d_9_steps_) + (1)
                if (d_30_nextConstrained_) == (eosToken):
                    d_7_forceFinal_ = True
                elif True:
                    d_31_appendedGenerated_: _dafny.Seq
                    d_32_appendedInside_: bool
                    d_33_appendedCurrent_: _dafny.Seq
                    out16_: _dafny.Seq
                    out17_: bool
                    out18_: _dafny.Seq
                    out16_, out17_, out18_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_30_nextConstrained_)
                    d_31_appendedGenerated_ = out16_
                    d_32_appendedInside_ = out17_
                    d_33_appendedCurrent_ = out18_
                    generated = d_31_appendedGenerated_
                    insideConstrainedOut = d_32_appendedInside_
                    currentConstrainedOut = d_33_appendedCurrent_
        cost = d_9_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

