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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step, concisely. Do the reasoning in plain text, but do not type delimiter tokens during the reasoning. Finish with exactly one final visible span of the form <<expression>>. Inside the final span put only a compact Python-style arithmetic expression or number: no words, no units, no Markdown, no LaTeX, and no nested delimiters. Use // or int(...) when integer division/rounding is needed."))
        if (len(validTokenGroups)) > (0):
            d_1_guidance_ = (d_1_guidance_) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " Prefer the problem's variable names, numbers, fractions, and arithmetic operators in the final expression.")))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_reasonLimit_: int
        d_2_reasonLimit_ = 118
        d_3_minAfterCue_: int
        d_3_minAfterCue_ = 28
        d_4_reasonTokens_: int
        d_4_reasonTokens_ = 0
        d_5_sawAnswerCue_: bool
        d_5_sawAnswerCue_ = False
        d_6_sawThereforeCue_: bool
        d_6_sawThereforeCue_ = False
        d_7_phase_: int
        d_7_phase_ = 0
        d_8_spanSteps_: int
        d_8_spanSteps_ = 0
        d_9_steps_: int
        d_9_steps_ = 0
        if (maxSteps) > (0):
            d_10_first_: _dafny.Seq
            out0_: _dafny.Seq
            out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
            d_10_first_ = out0_
            d_9_steps_ = 1
            if (d_10_first_) == (eosToken):
                d_7_phase_ = 1
            elif ((d_10_first_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))) or ((d_10_first_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")))):
                d_4_reasonTokens_ = 1
            elif True:
                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_10_first_]))
                d_4_reasonTokens_ = 1
                if (((((d_10_first_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Final")))) or ((d_10_first_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "final"))))) or ((d_10_first_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Answer"))))) or ((d_10_first_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "answer"))))) or ((d_10_first_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "####")))):
                    d_5_sawAnswerCue_ = True
                if ((((((d_10_first_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Therefore")))) or ((d_10_first_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "therefore"))))) or ((d_10_first_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Thus"))))) or ((d_10_first_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "thus"))))) or ((d_10_first_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Finally"))))) or ((d_10_first_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "finally")))):
                    d_6_sawThereforeCue_ = True
        with _dafny.label("0"):
            while ((d_9_steps_) < (maxSteps)) and ((d_7_phase_) < (2)):
                with _dafny.c_label("0"):
                    if (d_7_phase_) == (0):
                        d_11_shouldOpen_: bool
                        d_11_shouldOpen_ = ((((d_4_reasonTokens_) >= (d_2_reasonLimit_)) or ((d_5_sawAnswerCue_) and ((d_4_reasonTokens_) >= (d_3_minAfterCue_)))) or ((d_6_sawThereforeCue_) and ((d_4_reasonTokens_) >= (44)))) or (((d_9_steps_) + (90)) >= (maxSteps))
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
                            d_7_phase_ = 1
                            d_8_spanSteps_ = 0
                            d_9_steps_ = (d_9_steps_) + (1)
                        elif True:
                            d_15_next_: _dafny.Seq
                            out4_: _dafny.Seq
                            out4_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_15_next_ = out4_
                            d_9_steps_ = (d_9_steps_) + (1)
                            if (d_15_next_) == (eosToken):
                                d_7_phase_ = 1
                            elif ((d_15_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))) or ((d_15_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")))):
                                d_4_reasonTokens_ = (d_4_reasonTokens_) + (1)
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_15_next_]))
                                d_4_reasonTokens_ = (d_4_reasonTokens_) + (1)
                                if (((((d_15_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Final")))) or ((d_15_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "final"))))) or ((d_15_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Answer"))))) or ((d_15_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "answer"))))) or ((d_15_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "####")))):
                                    d_5_sawAnswerCue_ = True
                                if ((((((d_15_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Therefore")))) or ((d_15_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "therefore"))))) or ((d_15_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Thus"))))) or ((d_15_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "thus"))))) or ((d_15_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Finally"))))) or ((d_15_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "finally")))):
                                    d_6_sawThereforeCue_ = True
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
                        d_8_spanSteps_ = 0
                        d_9_steps_ = (d_9_steps_) + (1)
                    elif ((parser).IsCompletePrefix(currentConstrainedOut)) and ((len(currentConstrainedOut)) > (0)):
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
                        d_8_spanSteps_ = 0
                        d_7_phase_ = 2
                        raise _dafny.Break("0")
                    elif True:
                        d_22_remaining_: int
                        d_22_remaining_ = (maxSteps) - (d_9_steps_)
                        if (d_22_remaining_) <= (1):
                            raise _dafny.Break("0")
                        elif True:
                            d_23_chunkBudget_: int
                            d_23_chunkBudget_ = 14
                            if (d_8_spanSteps_) >= (20):
                                d_23_chunkBudget_ = 4
                            if (d_8_spanSteps_) >= (32):
                                d_23_chunkBudget_ = 1
                            d_24_availableForChunk_: int
                            d_24_availableForChunk_ = (d_22_remaining_) - (1)
                            if (d_24_availableForChunk_) < (d_23_chunkBudget_):
                                d_23_chunkBudget_ = d_24_availableForChunk_
                            d_25_constrainedPrompt_: _dafny.Seq
                            d_25_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_26_generatedOut_: _dafny.Seq
                            d_27_currentOut_: _dafny.Seq
                            d_28_hitEos_: bool
                            d_29_stepsUsed_: int
                            out11_: _dafny.Seq
                            out12_: _dafny.Seq
                            out13_: bool
                            out14_: int
                            out11_, out12_, out13_, out14_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_25_constrainedPrompt_, generated, currentConstrainedOut, d_23_chunkBudget_, eosToken)
                            d_26_generatedOut_ = out11_
                            d_27_currentOut_ = out12_
                            d_28_hitEos_ = out13_
                            d_29_stepsUsed_ = out14_
                            generated = d_26_generatedOut_
                            currentConstrainedOut = d_27_currentOut_
                            d_9_steps_ = (d_9_steps_) + (d_29_stepsUsed_)
                            d_8_spanSteps_ = (d_8_spanSteps_) + (d_29_stepsUsed_)
                            if (d_29_stepsUsed_) == (0):
                                raise _dafny.Break("0")
                    pass
            pass
        cost = d_9_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

