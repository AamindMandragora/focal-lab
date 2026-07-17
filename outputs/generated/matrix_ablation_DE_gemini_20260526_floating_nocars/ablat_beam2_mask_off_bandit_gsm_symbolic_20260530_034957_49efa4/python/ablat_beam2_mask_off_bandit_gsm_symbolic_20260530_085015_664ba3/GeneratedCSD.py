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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step. Use visible delimiters exactly like <<expression>> for symbolic arithmetic expressions and for the final answer. Inside delimiters use only compact arithmetic expressions or numbers: no words, no units, no LaTeX, and no nested delimiters. Prefer exact integer arithmetic with // when division is integral, and finish with a final answer span."))
        if (len(validTokenGroups)) > (0):
            d_1_guidance_ = (d_1_guidance_) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " Prefer the problem's given variable names and numeric symbols when forming expressions.")))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_reasonLimit_: int
        d_2_reasonLimit_ = 96
        d_3_reasonTokens_: int
        d_3_reasonTokens_ = 0
        d_4_sawAnswerCue_: bool
        d_4_sawAnswerCue_ = False
        d_5_phase_: int
        d_5_phase_ = 0
        d_6_spanSteps_: int
        d_6_spanSteps_ = 0
        d_7_penaltyTokens_: _dafny.Seq
        d_7_penaltyTokens_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "{")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "}")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "max")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "min")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "if")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "else")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "%")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "!=")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "/")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "ceil")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "round")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "approximately")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "about"))])
        d_8_steps_: int
        d_8_steps_ = 0
        if (maxSteps) > (0):
            d_9_first_: _dafny.Seq
            out0_: _dafny.Seq
            out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
            d_9_first_ = out0_
            d_8_steps_ = 1
            if (d_9_first_) == (eosToken):
                d_5_phase_ = 1
            elif ((d_9_first_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))) or ((d_9_first_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")))):
                d_3_reasonTokens_ = 1
            elif True:
                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_9_first_]))
                d_3_reasonTokens_ = 1
                if (((((((((((d_9_first_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Final")))) or ((d_9_first_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "final"))))) or ((d_9_first_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Answer"))))) or ((d_9_first_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "answer"))))) or ((d_9_first_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Therefore"))))) or ((d_9_first_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "therefore"))))) or ((d_9_first_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Thus"))))) or ((d_9_first_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "thus"))))) or ((d_9_first_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Finally"))))) or ((d_9_first_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "finally"))))) or ((d_9_first_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "####")))):
                    d_4_sawAnswerCue_ = True
        while ((d_8_steps_) < (maxSteps)) and ((d_5_phase_) < (2)):
            if (d_5_phase_) == (0):
                d_10_shouldOpen_: bool
                d_10_shouldOpen_ = (((d_3_reasonTokens_) >= (d_2_reasonLimit_)) or ((d_4_sawAnswerCue_) and ((d_3_reasonTokens_) >= (24)))) or (((d_8_steps_) + (140)) >= (maxSteps))
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
                    d_5_phase_ = 1
                    d_6_spanSteps_ = 0
                    d_8_steps_ = (d_8_steps_) + (1)
                elif True:
                    d_14_next_: _dafny.Seq
                    out4_: _dafny.Seq
                    out4_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                    d_14_next_ = out4_
                    d_8_steps_ = (d_8_steps_) + (1)
                    if (d_14_next_) == (eosToken):
                        d_5_phase_ = 1
                    elif ((d_14_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))) or ((d_14_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")))):
                        d_3_reasonTokens_ = (d_3_reasonTokens_) + (1)
                    elif True:
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_14_next_]))
                        d_3_reasonTokens_ = (d_3_reasonTokens_) + (1)
                        if (((((((((((d_14_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Final")))) or ((d_14_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "final"))))) or ((d_14_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Answer"))))) or ((d_14_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "answer"))))) or ((d_14_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Therefore"))))) or ((d_14_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "therefore"))))) or ((d_14_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Thus"))))) or ((d_14_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "thus"))))) or ((d_14_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Finally"))))) or ((d_14_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "finally"))))) or ((d_14_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "####")))):
                            d_4_sawAnswerCue_ = True
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
                d_6_spanSteps_ = 0
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
                d_5_phase_ = 2
            elif True:
                d_21_constrainedPrompt_: _dafny.Seq
                d_21_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                d_22_nextFinal_: _dafny.Seq
                out11_: _dafny.Seq
                out11_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_21_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('3e0'), d_7_penaltyTokens_, _dafny.BigRational('6e0'), 18, eosToken)
                d_22_nextFinal_ = out11_
                d_8_steps_ = (d_8_steps_) + (1)
                d_6_spanSteps_ = (d_6_spanSteps_) + (1)
                if (d_22_nextFinal_) == (eosToken):
                    if (d_6_spanSteps_) >= (64):
                        d_5_phase_ = 2
                elif True:
                    d_23_appendedGenerated_: _dafny.Seq
                    d_24_appendedInside_: bool
                    d_25_appendedCurrent_: _dafny.Seq
                    out12_: _dafny.Seq
                    out13_: bool
                    out14_: _dafny.Seq
                    out12_, out13_, out14_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_22_nextFinal_)
                    d_23_appendedGenerated_ = out12_
                    d_24_appendedInside_ = out13_
                    d_25_appendedCurrent_ = out14_
                    generated = d_23_appendedGenerated_
                    insideConstrainedOut = d_24_appendedInside_
                    currentConstrainedOut = d_25_appendedCurrent_
        cost = d_8_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

