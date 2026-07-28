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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step, concisely. Wrap arithmetic expressions and the final answer in visible << >> delimiters. Avoid typing stray delimiter tokens outside a span. Inside each span use only a compact arithmetic expression or number: no words, no units, no Markdown, no LaTeX, and no nested delimiters."))
        if (len(validTokenGroups)) > (0):
            d_1_guidance_ = (d_1_guidance_) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " Prefer numbers and symbols from the problem statement when forming the constrained arithmetic expression.")))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_phase_: int
        d_2_phase_ = 0
        if insideConstrainedOut:
            d_2_phase_ = 1
        d_3_reasonLimit_: int
        d_3_reasonLimit_ = 28
        d_4_freeTokens_: int
        d_4_freeTokens_ = 0
        d_5_sawAnswerCue_: bool
        d_5_sawAnswerCue_ = False
        d_6_spanSteps_: int
        d_6_spanSteps_ = 0
        d_7_penaltyTokens_: _dafny.Seq
        d_7_penaltyTokens_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<|im_start|>")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<|endoftext|>")), eosToken])
        d_8_narrowThreshold_: int
        d_8_narrowThreshold_ = 18
        d_9_steps_: int
        d_9_steps_ = 0
        while ((d_9_steps_) < (maxSteps)) and ((d_2_phase_) < (2)):
            if (d_2_phase_) == (0):
                d_10_shouldOpen_: bool
                d_10_shouldOpen_ = (((d_4_freeTokens_) >= (d_3_reasonLimit_)) or ((d_5_sawAnswerCue_) and ((d_4_freeTokens_) >= (8)))) or (((d_9_steps_) + (80)) >= (maxSteps))
                if d_10_shouldOpen_:
                    d_11_openedGenerated_: _dafny.Seq
                    d_12_openedInside_: bool
                    d_13_openedCurrent_: _dafny.Seq
                    out0_: _dafny.Seq
                    out1_: bool
                    out2_: _dafny.Seq
                    out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                    d_11_openedGenerated_ = out0_
                    d_12_openedInside_ = out1_
                    d_13_openedCurrent_ = out2_
                    generated = d_11_openedGenerated_
                    insideConstrainedOut = d_12_openedInside_
                    currentConstrainedOut = d_13_openedCurrent_
                    d_2_phase_ = 1
                    d_6_spanSteps_ = 0
                    d_9_steps_ = (d_9_steps_) + (1)
                elif True:
                    d_14_next_: _dafny.Seq
                    out3_: _dafny.Seq
                    out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                    d_14_next_ = out3_
                    d_9_steps_ = (d_9_steps_) + (1)
                    if (d_14_next_) == (eosToken):
                        d_4_freeTokens_ = d_3_reasonLimit_
                        d_5_sawAnswerCue_ = True
                    elif (d_14_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_14_next_]))
                        insideConstrainedOut = True
                        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                        d_2_phase_ = 1
                        d_6_spanSteps_ = 0
                    elif (((d_14_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")))) or ((d_14_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<|im_start|>"))))) or ((d_14_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<|endoftext|>")))):
                        d_4_freeTokens_ = (d_4_freeTokens_) + (1)
                    elif True:
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_14_next_]))
                        d_4_freeTokens_ = (d_4_freeTokens_) + (1)
                        if (((((((((((d_14_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Final")))) or ((d_14_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "final"))))) or ((d_14_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Answer"))))) or ((d_14_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "answer"))))) or ((d_14_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Therefore"))))) or ((d_14_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "therefore"))))) or ((d_14_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Thus"))))) or ((d_14_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "thus"))))) or ((d_14_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Finally"))))) or ((d_14_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "finally"))))) or ((d_14_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "####")))):
                            d_5_sawAnswerCue_ = True
            elif not(insideConstrainedOut):
                d_15_openedGenerated2_: _dafny.Seq
                d_16_openedInside2_: bool
                d_17_openedCurrent2_: _dafny.Seq
                out4_: _dafny.Seq
                out5_: bool
                out6_: _dafny.Seq
                out4_, out5_, out6_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                d_15_openedGenerated2_ = out4_
                d_16_openedInside2_ = out5_
                d_17_openedCurrent2_ = out6_
                generated = d_15_openedGenerated2_
                insideConstrainedOut = d_16_openedInside2_
                currentConstrainedOut = d_17_openedCurrent2_
                d_2_phase_ = 1
                d_6_spanSteps_ = 0
                d_9_steps_ = (d_9_steps_) + (1)
            elif (parser).IsCompletePrefix(currentConstrainedOut):
                d_18_closedGenerated_: _dafny.Seq
                d_19_closedInside_: bool
                d_20_closedCurrent_: _dafny.Seq
                out7_: _dafny.Seq
                out8_: bool
                out9_: _dafny.Seq
                out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_18_closedGenerated_ = out7_
                d_19_closedInside_ = out8_
                d_20_closedCurrent_ = out9_
                generated = d_18_closedGenerated_
                insideConstrainedOut = d_19_closedInside_
                currentConstrainedOut = d_20_closedCurrent_
                d_9_steps_ = (d_9_steps_) + (1)
                d_2_phase_ = 2
            elif True:
                d_21_constrainedPrompt_: _dafny.Seq
                d_21_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                d_22_nextConstrained_: _dafny.Seq
                out10_: _dafny.Seq
                out10_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_21_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('5e0'), d_7_penaltyTokens_, _dafny.BigRational('7e0'), d_8_narrowThreshold_, eosToken)
                d_22_nextConstrained_ = out10_
                d_9_steps_ = (d_9_steps_) + (1)
                d_6_spanSteps_ = (d_6_spanSteps_) + (1)
                if (d_22_nextConstrained_) == (eosToken):
                    d_2_phase_ = 2
                elif True:
                    d_23_appendedGenerated_: _dafny.Seq
                    d_24_appendedInside_: bool
                    d_25_appendedCurrent_: _dafny.Seq
                    out11_: _dafny.Seq
                    out12_: bool
                    out13_: _dafny.Seq
                    out11_, out12_, out13_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_22_nextConstrained_)
                    d_23_appendedGenerated_ = out11_
                    d_24_appendedInside_ = out12_
                    d_25_appendedCurrent_ = out13_
                    generated = d_23_appendedGenerated_
                    insideConstrainedOut = d_24_appendedInside_
                    currentConstrainedOut = d_25_appendedCurrent_
        cost = d_9_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

