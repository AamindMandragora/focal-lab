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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step. Wrap every intermediate symbolic arithmetic expression and the final answer in visible delimiters exactly as <<expression>>. Outside the delimiters, write concise prose. Inside << >> write only a compact arithmetic expression or number: no words, units, LaTeX, nested delimiters, or unmatched delimiters. End with a final <<answer expression>> span."))
        if (len(validTokenGroups)) > (0):
            d_1_guidance_ = (d_1_guidance_) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " Prefer numbers, variables, and arithmetic operators that occur in the problem context.")))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_reasonTokens_: int
        d_3_reasonTokens_ = 0
        d_4_closedSpans_: int
        d_4_closedSpans_ = 0
        d_5_spanSteps_: int
        d_5_spanSteps_ = 0
        d_6_sawAnswerCue_: bool
        d_6_sawAnswerCue_ = False
        d_7_spanIsFinal_: bool
        d_7_spanIsFinal_ = False
        d_8_phase_: int
        d_8_phase_ = 0
        if insideConstrainedOut:
            d_8_phase_ = 1
        elif True:
            d_8_phase_ = 0
        d_9_reasonLimit_: int
        d_9_reasonLimit_ = 72
        d_10_spanSoftLimit_: int
        d_10_spanSoftLimit_ = 96
        d_11_penaltyTokens_: _dafny.Seq
        d_11_penaltyTokens_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\\")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\\[")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\\]")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "{")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "}")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "text")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Text")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<|endoftext|>")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<|im_start|>")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<|im_end|>")), eosToken])
        while ((d_2_steps_) < (maxSteps)) and ((d_8_phase_) < (2)):
            if (d_8_phase_) == (0):
                d_12_shouldOpen_: bool
                d_12_shouldOpen_ = ((((d_3_reasonTokens_) >= (d_9_reasonLimit_)) or ((d_6_sawAnswerCue_) and ((d_3_reasonTokens_) >= (18)))) or ((d_4_closedSpans_) >= (4))) or (((d_2_steps_) + (96)) >= (maxSteps))
                if d_12_shouldOpen_:
                    d_13_openedGenerated_: _dafny.Seq
                    d_14_openedInside_: bool
                    d_15_openedCurrent_: _dafny.Seq
                    out0_: _dafny.Seq
                    out1_: bool
                    out2_: _dafny.Seq
                    out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                    d_13_openedGenerated_ = out0_
                    d_14_openedInside_ = out1_
                    d_15_openedCurrent_ = out2_
                    generated = d_13_openedGenerated_
                    insideConstrainedOut = d_14_openedInside_
                    currentConstrainedOut = d_15_openedCurrent_
                    d_8_phase_ = 1
                    d_7_spanIsFinal_ = True
                    d_5_spanSteps_ = 0
                    d_2_steps_ = (d_2_steps_) + (1)
                elif True:
                    d_16_next_: _dafny.Seq
                    out3_: _dafny.Seq
                    out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                    d_16_next_ = out3_
                    d_2_steps_ = (d_2_steps_) + (1)
                    if (d_16_next_) == (eosToken):
                        if (d_4_closedSpans_) > (0):
                            d_8_phase_ = 2
                        elif True:
                            d_8_phase_ = 1
                            d_7_spanIsFinal_ = True
                            d_5_spanSteps_ = 0
                    elif (d_16_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_16_next_]))
                        d_3_reasonTokens_ = (d_3_reasonTokens_) + (1)
                        insideConstrainedOut = True
                        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                        d_8_phase_ = 1
                        d_7_spanIsFinal_ = (((d_6_sawAnswerCue_) or ((d_3_reasonTokens_) >= (32))) or ((d_4_closedSpans_) >= (3))) or (((d_2_steps_) + (96)) >= (maxSteps))
                        d_5_spanSteps_ = 0
                    elif (d_16_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))):
                        d_3_reasonTokens_ = (d_3_reasonTokens_) + (1)
                    elif True:
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_16_next_]))
                        d_3_reasonTokens_ = (d_3_reasonTokens_) + (1)
                        if ((((((((d_16_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Final")))) or ((d_16_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "final"))))) or ((d_16_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Answer"))))) or ((d_16_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "answer"))))) or ((d_16_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Therefore"))))) or ((d_16_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "therefore"))))) or ((d_16_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Thus"))))) or ((d_16_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "thus")))):
                            d_6_sawAnswerCue_ = True
            elif not(insideConstrainedOut):
                d_17_openedGenerated2_: _dafny.Seq
                d_18_openedInside2_: bool
                d_19_openedCurrent2_: _dafny.Seq
                out4_: _dafny.Seq
                out5_: bool
                out6_: _dafny.Seq
                out4_, out5_, out6_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                d_17_openedGenerated2_ = out4_
                d_18_openedInside2_ = out5_
                d_19_openedCurrent2_ = out6_
                generated = d_17_openedGenerated2_
                insideConstrainedOut = d_18_openedInside2_
                currentConstrainedOut = d_19_openedCurrent2_
                d_7_spanIsFinal_ = True
                d_5_spanSteps_ = 0
                d_2_steps_ = (d_2_steps_) + (1)
            elif (parser).IsCompletePrefix(currentConstrainedOut):
                d_20_wasFinal_: bool
                d_20_wasFinal_ = d_7_spanIsFinal_
                d_21_closedGenerated_: _dafny.Seq
                d_22_closedInside_: bool
                d_23_closedCurrent_: _dafny.Seq
                out7_: _dafny.Seq
                out8_: bool
                out9_: _dafny.Seq
                out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_21_closedGenerated_ = out7_
                d_22_closedInside_ = out8_
                d_23_closedCurrent_ = out9_
                generated = d_21_closedGenerated_
                insideConstrainedOut = d_22_closedInside_
                currentConstrainedOut = d_23_closedCurrent_
                d_2_steps_ = (d_2_steps_) + (1)
                d_4_closedSpans_ = (d_4_closedSpans_) + (1)
                d_5_spanSteps_ = 0
                if (((d_20_wasFinal_) or ((d_6_sawAnswerCue_) and ((d_4_closedSpans_) >= (1)))) or ((d_4_closedSpans_) >= (5))) or (((d_2_steps_) + (32)) >= (maxSteps)):
                    d_8_phase_ = 2
                elif True:
                    d_8_phase_ = 0
                    if (d_4_closedSpans_) >= (3):
                        d_6_sawAnswerCue_ = True
            elif True:
                d_24_constrainedPrompt_: _dafny.Seq
                d_24_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                d_25_nextConstrained_: _dafny.Seq
                out10_: _dafny.Seq
                out10_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_24_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('5e0'), d_11_penaltyTokens_, _dafny.BigRational('7e0'), 14, eosToken)
                d_25_nextConstrained_ = out10_
                d_2_steps_ = (d_2_steps_) + (1)
                d_5_spanSteps_ = (d_5_spanSteps_) + (1)
                if (d_25_nextConstrained_) == (eosToken):
                    if (d_5_spanSteps_) >= (d_10_spanSoftLimit_):
                        d_7_spanIsFinal_ = True
                elif True:
                    d_26_appendedGenerated_: _dafny.Seq
                    d_27_appendedInside_: bool
                    d_28_appendedCurrent_: _dafny.Seq
                    out11_: _dafny.Seq
                    out12_: bool
                    out13_: _dafny.Seq
                    out11_, out12_, out13_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_25_nextConstrained_)
                    d_26_appendedGenerated_ = out11_
                    d_27_appendedInside_ = out12_
                    d_28_appendedCurrent_ = out13_
                    generated = d_26_appendedGenerated_
                    insideConstrainedOut = d_27_appendedInside_
                    currentConstrainedOut = d_28_appendedCurrent_
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

