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
        insideConstrainedOut = False
        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
        cost = 0
        d_1_guidance_: _dafny.Seq
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step, concisely. Do not type angle brackets during the prose reasoning; the decoder will add the final visible <<expression>> span. End your reasoning with a compact symbolic arithmetic answer. Inside the final span use only an arithmetic expression or number: no words, no units, no LaTeX, and no nested delimiters."))
        if (len(validTokenGroups)) > (0):
            d_1_guidance_ = (d_1_guidance_) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " Prefer the problem's variables, numbers, and arithmetic operators for the final expression.")))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_reasonTokens_: int
        d_2_reasonTokens_ = 0
        d_3_sawAnswerCue_: bool
        d_3_sawAnswerCue_ = False
        d_4_phase_: int
        d_4_phase_ = 0
        d_5_spanSteps_: int
        d_5_spanSteps_ = 0
        d_6_arithmeticGroups_: _dafny.Seq
        d_6_arithmeticGroups_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "+")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "-")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "*")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "/")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "//"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "(")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ")"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "n")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "m")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "k")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "x")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "y")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "t")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "d")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "r")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "w"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "n1")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "n2")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "n3")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "n4")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "m1")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "m2")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "w1")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "w2")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "w3")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "w4"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "c")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "c1")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "c2")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "c3")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "cn")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "cm")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "total")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "mult"))])])
        d_7_groups_: _dafny.Seq
        d_7_groups_ = (d_6_arithmeticGroups_) + (validTokenGroups)
        d_8_penaltyTokens_: _dafny.Seq
        d_8_penaltyTokens_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "max")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "min")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "ceil")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "floor")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "round")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "int")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\\")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\\(")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\\)")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\\[")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\\]")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "{")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "}")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "text")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "frac")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "cdot")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "times")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "if")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "otherwise")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "positive")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "minutes")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "dollars")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "pounds")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "answer")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Answer")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Final")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "final")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Therefore")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Thus"))])
        d_9_steps_: int
        d_9_steps_ = 0
        if (maxSteps) > (0):
            d_10_first_: _dafny.Seq
            out0_: _dafny.Seq
            out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
            d_10_first_ = out0_
            d_9_steps_ = 1
            d_2_reasonTokens_ = 1
            if (d_10_first_) == (eosToken):
                d_4_phase_ = 1
            elif ((VerifiedDecoderAgent.CSDHelpers.CountSubstring(d_10_first_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<")))) > (0)) or ((VerifiedDecoderAgent.CSDHelpers.CountSubstring(d_10_first_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">")))) > (0)):
                pass
            elif True:
                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_10_first_]))
                if (((((((((((d_10_first_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Final")))) or ((d_10_first_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "final"))))) or ((d_10_first_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Answer"))))) or ((d_10_first_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "answer"))))) or ((d_10_first_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Therefore"))))) or ((d_10_first_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "therefore"))))) or ((d_10_first_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Thus"))))) or ((d_10_first_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "thus"))))) or ((d_10_first_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Finally"))))) or ((d_10_first_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "finally"))))) or ((d_10_first_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "####")))):
                    d_3_sawAnswerCue_ = True
        with _dafny.label("0"):
            while ((d_9_steps_) < (maxSteps)) and ((d_4_phase_) < (2)):
                with _dafny.c_label("0"):
                    if (d_4_phase_) == (0):
                        d_11_shouldOpen_: bool
                        d_11_shouldOpen_ = (((d_3_sawAnswerCue_) and ((d_2_reasonTokens_) >= (45))) or ((d_2_reasonTokens_) >= (135))) or (((d_9_steps_) + (96)) >= (maxSteps))
                        if d_11_shouldOpen_:
                            d_12_openedGenerated_: _dafny.Seq
                            d_13_openedInside_: bool
                            d_14_openedCurrent_: _dafny.Seq
                            out1_: _dafny.Seq
                            out2_: bool
                            out3_: _dafny.Seq
                            out1_, out2_, out3_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_12_openedGenerated_ = out1_
                            d_13_openedInside_ = out2_
                            d_14_openedCurrent_ = out3_
                            generated = d_12_openedGenerated_
                            insideConstrainedOut = d_13_openedInside_
                            currentConstrainedOut = d_14_openedCurrent_
                            d_4_phase_ = 1
                            d_5_spanSteps_ = 0
                            d_9_steps_ = (d_9_steps_) + (1)
                        elif True:
                            d_15_next_: _dafny.Seq
                            out4_: _dafny.Seq
                            out4_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_15_next_ = out4_
                            d_9_steps_ = (d_9_steps_) + (1)
                            d_2_reasonTokens_ = (d_2_reasonTokens_) + (1)
                            if (d_15_next_) == (eosToken):
                                d_4_phase_ = 1
                            elif ((VerifiedDecoderAgent.CSDHelpers.CountSubstring(d_15_next_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<")))) > (0)) or ((VerifiedDecoderAgent.CSDHelpers.CountSubstring(d_15_next_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">")))) > (0)):
                                pass
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_15_next_]))
                                if (((((((((((d_15_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Final")))) or ((d_15_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "final"))))) or ((d_15_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Answer"))))) or ((d_15_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "answer"))))) or ((d_15_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Therefore"))))) or ((d_15_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "therefore"))))) or ((d_15_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Thus"))))) or ((d_15_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "thus"))))) or ((d_15_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Finally"))))) or ((d_15_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "finally"))))) or ((d_15_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "####")))):
                                    d_3_sawAnswerCue_ = True
                    elif not(insideConstrainedOut):
                        d_16_openedGenerated2_: _dafny.Seq
                        d_17_openedInside2_: bool
                        d_18_openedCurrent2_: _dafny.Seq
                        out5_: _dafny.Seq
                        out6_: bool
                        out7_: _dafny.Seq
                        out5_, out6_, out7_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                        d_16_openedGenerated2_ = out5_
                        d_17_openedInside2_ = out6_
                        d_18_openedCurrent2_ = out7_
                        generated = d_16_openedGenerated2_
                        insideConstrainedOut = d_17_openedInside2_
                        currentConstrainedOut = d_18_openedCurrent2_
                        d_5_spanSteps_ = 0
                        d_9_steps_ = (d_9_steps_) + (1)
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_19_closedGenerated_: _dafny.Seq
                        d_20_closedInside_: bool
                        d_21_closedCurrent_: _dafny.Seq
                        out8_: _dafny.Seq
                        out9_: bool
                        out10_: _dafny.Seq
                        out8_, out9_, out10_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_19_closedGenerated_ = out8_
                        d_20_closedInside_ = out9_
                        d_21_closedCurrent_ = out10_
                        generated = d_19_closedGenerated_
                        insideConstrainedOut = d_20_closedInside_
                        currentConstrainedOut = d_21_closedCurrent_
                        d_9_steps_ = (d_9_steps_) + (1)
                        d_4_phase_ = 2
                        raise _dafny.Break("0")
                    elif True:
                        d_22_constrainedPrompt_: _dafny.Seq
                        d_22_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_23_nextFinal_: _dafny.Seq
                        out11_: _dafny.Seq
                        out11_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_22_constrainedPrompt_, currentConstrainedOut, d_7_groups_, _dafny.BigRational('6e0'), d_8_penaltyTokens_, _dafny.BigRational('8e0'), 18, eosToken)
                        d_23_nextFinal_ = out11_
                        d_9_steps_ = (d_9_steps_) + (1)
                        d_5_spanSteps_ = (d_5_spanSteps_) + (1)
                        if (d_23_nextFinal_) == (eosToken):
                            pass
                        elif True:
                            d_24_appendedGenerated_: _dafny.Seq
                            d_25_appendedInside_: bool
                            d_26_appendedCurrent_: _dafny.Seq
                            out12_: _dafny.Seq
                            out13_: bool
                            out14_: _dafny.Seq
                            out12_, out13_, out14_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_23_nextFinal_)
                            d_24_appendedGenerated_ = out12_
                            d_25_appendedInside_ = out13_
                            d_26_appendedCurrent_ = out14_
                            generated = d_24_appendedGenerated_
                            insideConstrainedOut = d_25_appendedInside_
                            currentConstrainedOut = d_26_appendedCurrent_
                    pass
            pass
        cost = d_9_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

