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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step. Wrap every intermediate symbolic arithmetic expression and the final answer inside visible << >> delimiters. Outside delimiters, explain concisely in words. Inside delimiters, write only a compact arithmetic expression or number: no words, no units, no Markdown, no LaTeX, and no nested delimiters. Close each span before continuing."))
        if (len(validTokenGroups)) > (0):
            d_1_guidance_ = (d_1_guidance_) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " Prefer numbers, variables, and operators that appear in the problem when forming expressions.")))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_reasonTokens_: int
        d_3_reasonTokens_ = 0
        d_4_sawAnswerCue_: bool
        d_4_sawAnswerCue_ = False
        d_5_spanIsFinal_: bool
        d_5_spanIsFinal_ = False
        d_6_completeSpans_: int
        d_6_completeSpans_ = 0
        d_7_finished_: bool
        d_7_finished_ = False
        d_8_forceAfter_: int
        d_8_forceAfter_ = 28
        d_9_answerForceAfter_: int
        d_9_answerForceAfter_ = 6
        d_10_narrowThreshold_: int
        d_10_narrowThreshold_ = 14
        while ((d_2_steps_) < (maxSteps)) and (not(d_7_finished_)):
            if not(insideConstrainedOut):
                d_11_nearBudget_: bool
                d_11_nearBudget_ = ((d_2_steps_) + (80)) >= (maxSteps)
                d_12_shouldOpen_: bool
                d_12_shouldOpen_ = (((d_3_reasonTokens_) >= (d_8_forceAfter_)) or ((d_4_sawAnswerCue_) and ((d_3_reasonTokens_) >= (d_9_answerForceAfter_)))) or (d_11_nearBudget_)
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
                    d_2_steps_ = (d_2_steps_) + (1)
                    d_3_reasonTokens_ = 0
                    d_5_spanIsFinal_ = (d_4_sawAnswerCue_) or (d_11_nearBudget_)
                elif True:
                    d_16_next_: _dafny.Seq
                    out3_: _dafny.Seq
                    out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                    d_16_next_ = out3_
                    d_2_steps_ = (d_2_steps_) + (1)
                    if (d_16_next_) == (eosToken):
                        d_7_finished_ = True
                    elif (d_16_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_16_next_]))
                        insideConstrainedOut = True
                        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                        d_3_reasonTokens_ = 0
                        d_5_spanIsFinal_ = (d_4_sawAnswerCue_) or (((d_2_steps_) + (80)) >= (maxSteps))
                    elif (d_16_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))):
                        d_3_reasonTokens_ = (d_3_reasonTokens_) + (1)
                    elif True:
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_16_next_]))
                        d_3_reasonTokens_ = (d_3_reasonTokens_) + (1)
                        if (((((((((((d_16_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Final")))) or ((d_16_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "final"))))) or ((d_16_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Answer"))))) or ((d_16_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "answer"))))) or ((d_16_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Therefore"))))) or ((d_16_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "therefore"))))) or ((d_16_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Thus"))))) or ((d_16_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "thus"))))) or ((d_16_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Finally"))))) or ((d_16_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "finally"))))) or ((d_16_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "####")))):
                            d_4_sawAnswerCue_ = True
            elif (parser).IsCompletePrefix(currentConstrainedOut):
                d_17_closedGenerated_: _dafny.Seq
                d_18_closedInside_: bool
                d_19_closedCurrent_: _dafny.Seq
                out4_: _dafny.Seq
                out5_: bool
                out6_: _dafny.Seq
                out4_, out5_, out6_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_17_closedGenerated_ = out4_
                d_18_closedInside_ = out5_
                d_19_closedCurrent_ = out6_
                generated = d_17_closedGenerated_
                insideConstrainedOut = d_18_closedInside_
                currentConstrainedOut = d_19_closedCurrent_
                d_2_steps_ = (d_2_steps_) + (1)
                d_6_completeSpans_ = (d_6_completeSpans_) + (1)
                d_3_reasonTokens_ = 0
                if d_5_spanIsFinal_:
                    d_7_finished_ = True
                elif (d_6_completeSpans_) >= (2):
                    d_4_sawAnswerCue_ = True
                d_5_spanIsFinal_ = False
            elif True:
                d_20_constrainedPrompt_: _dafny.Seq
                d_20_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                d_21_nextIn_: _dafny.Seq
                out7_: _dafny.Seq
                out7_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_20_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_10_narrowThreshold_, eosToken)
                d_21_nextIn_ = out7_
                d_2_steps_ = (d_2_steps_) + (1)
                if (d_21_nextIn_) == (eosToken):
                    d_7_finished_ = True
                elif True:
                    d_22_appendedGenerated_: _dafny.Seq
                    d_23_appendedInside_: bool
                    d_24_appendedCurrent_: _dafny.Seq
                    out8_: _dafny.Seq
                    out9_: bool
                    out10_: _dafny.Seq
                    out8_, out9_, out10_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_21_nextIn_)
                    d_22_appendedGenerated_ = out8_
                    d_23_appendedInside_ = out9_
                    d_24_appendedCurrent_ = out10_
                    generated = d_22_appendedGenerated_
                    insideConstrainedOut = d_23_appendedInside_
                    currentConstrainedOut = d_24_appendedCurrent_
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

