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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step, concisely. Wrap every intermediate symbolic arithmetic expression and the final answer inside visible << >> delimiters. Inside each delimiter use only a compact arithmetic expression or number: no words, no units, no LaTeX, and no nested delimiters. End with a final answer span like <<w*r+x-w*n>>. Prefer *, +, -, /, //, and parentheses; use // when an integer whole-count quotient is required."))
        if (len(validTokenGroups)) > (0):
            d_1_guidance_ = (d_1_guidance_) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " Prefer the problem's variable names and numeric/operator tokens when valid.")))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_arithmeticGroups_: _dafny.Seq
        d_2_arithmeticGroups_ = (validTokenGroups) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "0")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "1")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "2")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "3")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "4")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "5")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "6")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "7")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "8")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "9"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "+")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "-")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "*")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "/")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "//"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "(")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ")"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "n")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "x")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "y")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "k")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "w")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "r")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "total")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "sum")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "answer"))])]))
        d_3_penaltyTokens_: _dafny.Seq
        d_3_penaltyTokens_ = _dafny.SeqWithoutIsStrInference([eosToken, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " ")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "  ")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\n")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\t")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " <<")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " >>"))])
        d_4_reasonLimit_: int
        d_4_reasonLimit_ = 85
        d_5_reasonTokens_: int
        d_5_reasonTokens_ = 0
        d_6_sawAnswerCue_: bool
        d_6_sawAnswerCue_ = False
        d_7_phase_: int
        d_7_phase_ = 0
        d_8_steps_: int
        d_8_steps_ = 0
        if (maxSteps) > (0):
            d_9_first_: _dafny.Seq
            out0_: _dafny.Seq
            out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
            d_9_first_ = out0_
            d_8_steps_ = 1
            if (d_9_first_) == (eosToken):
                d_7_phase_ = 1
            elif (d_9_first_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                d_5_reasonTokens_ = 1
            elif (d_9_first_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))):
                d_5_reasonTokens_ = 1
            elif True:
                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_9_first_]))
                d_5_reasonTokens_ = 1
                if ((((((d_9_first_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Final")))) or ((d_9_first_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "final"))))) or ((d_9_first_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Answer"))))) or ((d_9_first_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "answer"))))) or ((d_9_first_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Therefore"))))) or ((d_9_first_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "therefore")))):
                    d_6_sawAnswerCue_ = True
        with _dafny.label("0"):
            while ((d_8_steps_) < (maxSteps)) and ((d_7_phase_) < (2)):
                with _dafny.c_label("0"):
                    if (d_7_phase_) == (0):
                        d_10_shouldOpen_: bool
                        d_10_shouldOpen_ = (((d_5_reasonTokens_) >= (d_4_reasonLimit_)) or ((d_6_sawAnswerCue_) and ((d_5_reasonTokens_) >= (24)))) or (((d_8_steps_) + (48)) >= (maxSteps))
                        if d_10_shouldOpen_:
                            d_11_openedGenerated_: _dafny.Seq
                            d_12_openedInside_: bool
                            d_13_openedCurrent_: _dafny.Seq
                            out1_: _dafny.Seq
                            out2_: bool
                            out3_: _dafny.Seq
                            out1_, out2_, out3_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_11_openedGenerated_ = out1_
                            d_12_openedInside_ = out2_
                            d_13_openedCurrent_ = out3_
                            generated = d_11_openedGenerated_
                            insideConstrainedOut = d_12_openedInside_
                            currentConstrainedOut = d_13_openedCurrent_
                            d_7_phase_ = 1
                            d_8_steps_ = (d_8_steps_) + (1)
                        elif True:
                            d_14_nextFree_: _dafny.Seq
                            out4_: _dafny.Seq
                            out4_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_14_nextFree_ = out4_
                            d_8_steps_ = (d_8_steps_) + (1)
                            if (d_14_nextFree_) == (eosToken):
                                d_7_phase_ = 1
                            elif (d_14_nextFree_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                d_5_reasonTokens_ = (d_5_reasonTokens_) + (1)
                            elif (d_14_nextFree_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))):
                                d_5_reasonTokens_ = (d_5_reasonTokens_) + (1)
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_14_nextFree_]))
                                d_5_reasonTokens_ = (d_5_reasonTokens_) + (1)
                                if ((((((d_14_nextFree_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Final")))) or ((d_14_nextFree_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "final"))))) or ((d_14_nextFree_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Answer"))))) or ((d_14_nextFree_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "answer"))))) or ((d_14_nextFree_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Therefore"))))) or ((d_14_nextFree_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "therefore")))):
                                    d_6_sawAnswerCue_ = True
                    elif not(insideConstrainedOut):
                        d_15_openedGenerated2_: _dafny.Seq
                        d_16_openedInside2_: bool
                        d_17_openedCurrent2_: _dafny.Seq
                        out5_: _dafny.Seq
                        out6_: bool
                        out7_: _dafny.Seq
                        out5_, out6_, out7_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                        d_15_openedGenerated2_ = out5_
                        d_16_openedInside2_ = out6_
                        d_17_openedCurrent2_ = out7_
                        generated = d_15_openedGenerated2_
                        insideConstrainedOut = d_16_openedInside2_
                        currentConstrainedOut = d_17_openedCurrent2_
                        d_8_steps_ = (d_8_steps_) + (1)
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_18_closedGenerated_: _dafny.Seq
                        d_19_closedInside_: bool
                        d_20_closedCurrent_: _dafny.Seq
                        out8_: _dafny.Seq
                        out9_: bool
                        out10_: _dafny.Seq
                        out8_, out9_, out10_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_18_closedGenerated_ = out8_
                        d_19_closedInside_ = out9_
                        d_20_closedCurrent_ = out10_
                        generated = d_18_closedGenerated_
                        insideConstrainedOut = d_19_closedInside_
                        currentConstrainedOut = d_20_closedCurrent_
                        d_8_steps_ = (d_8_steps_) + (1)
                        d_7_phase_ = 2
                        raise _dafny.Break("0")
                    elif True:
                        if ((maxSteps) - (d_8_steps_)) <= (1):
                            raise _dafny.Break("0")
                        elif True:
                            d_21_constrainedPrompt_: _dafny.Seq
                            d_21_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_22_next_: _dafny.Seq
                            out11_: _dafny.Seq
                            out11_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_21_constrainedPrompt_, currentConstrainedOut, d_2_arithmeticGroups_, _dafny.BigRational('7e0'), d_3_penaltyTokens_, _dafny.BigRational('12e0'), 64, eosToken)
                            d_22_next_ = out11_
                            d_8_steps_ = (d_8_steps_) + (1)
                            if (d_22_next_) == (eosToken):
                                pass
                            elif ((((((((((d_22_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " ")))) or ((d_22_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "  "))))) or ((d_22_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\n"))))) or ((d_22_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\t"))))) or ((d_22_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))))) or ((d_22_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))))) or ((d_22_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<"))))) or ((d_22_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">"))))) or ((d_22_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " <<"))))) or ((d_22_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " >>")))):
                                pass
                            elif True:
                                d_23_appendedGenerated_: _dafny.Seq
                                d_24_appendedInside_: bool
                                d_25_appendedCurrent_: _dafny.Seq
                                out12_: _dafny.Seq
                                out13_: bool
                                out14_: _dafny.Seq
                                out12_, out13_, out14_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_22_next_)
                                d_23_appendedGenerated_ = out12_
                                d_24_appendedInside_ = out13_
                                d_25_appendedCurrent_ = out14_
                                generated = d_23_appendedGenerated_
                                insideConstrainedOut = d_24_appendedInside_
                                currentConstrainedOut = d_25_appendedCurrent_
                    pass
            pass
        cost = d_8_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

