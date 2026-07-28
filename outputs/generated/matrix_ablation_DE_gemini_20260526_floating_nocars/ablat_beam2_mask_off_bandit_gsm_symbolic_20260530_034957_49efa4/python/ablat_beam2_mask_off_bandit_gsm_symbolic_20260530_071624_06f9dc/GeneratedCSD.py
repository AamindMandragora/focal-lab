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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step, but keep the reasoning concise. Do not type unmatched << or >> in prose; the decoder will force the final visible <<expression>> span. The final span must contain only the compact symbolic arithmetic expression or number for the answer: no words, no units, no LaTeX, no nested delimiters."))
        if (len(validTokenGroups)) > (0):
            d_1_guidance_ = (d_1_guidance_) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " Prefer variables, numbers, and arithmetic operators from the problem when writing the final expression.")))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_reasonLimit_: int
        d_2_reasonLimit_ = 36
        d_3_reasonTokens_: int
        d_3_reasonTokens_ = 0
        d_4_sawAnswerCue_: bool
        d_4_sawAnswerCue_ = False
        d_5_phase_: int
        d_5_phase_ = 0
        d_6_spanSteps_: int
        d_6_spanSteps_ = 0
        d_7_steps_: int
        d_7_steps_ = 0
        if (maxSteps) > (0):
            d_8_first_: _dafny.Seq
            out0_: _dafny.Seq
            out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
            d_8_first_ = out0_
            d_7_steps_ = 1
            if (d_8_first_) == (eosToken):
                d_5_phase_ = 1
            elif ((d_8_first_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))) or ((d_8_first_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")))):
                d_3_reasonTokens_ = 1
            elif True:
                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_8_first_]))
                d_3_reasonTokens_ = 1
                if ((((((((d_8_first_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Final")))) or ((d_8_first_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "final"))))) or ((d_8_first_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Answer"))))) or ((d_8_first_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "answer"))))) or ((d_8_first_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Therefore"))))) or ((d_8_first_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "therefore"))))) or ((d_8_first_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Thus"))))) or ((d_8_first_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "thus")))):
                    d_4_sawAnswerCue_ = True
        with _dafny.label("0"):
            while ((d_7_steps_) < (maxSteps)) and ((d_5_phase_) < (2)):
                with _dafny.c_label("0"):
                    if (d_5_phase_) == (0):
                        d_9_shouldOpen_: bool
                        d_9_shouldOpen_ = (((d_3_reasonTokens_) >= (d_2_reasonLimit_)) or ((d_4_sawAnswerCue_) and ((d_3_reasonTokens_) >= (14)))) or (((d_7_steps_) + (96)) >= (maxSteps))
                        if d_9_shouldOpen_:
                            d_10_openedGenerated_: _dafny.Seq
                            d_11_openedInside_: bool
                            d_12_openedCurrent_: _dafny.Seq
                            out1_: _dafny.Seq
                            out2_: bool
                            out3_: _dafny.Seq
                            out1_, out2_, out3_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_10_openedGenerated_ = out1_
                            d_11_openedInside_ = out2_
                            d_12_openedCurrent_ = out3_
                            generated = d_10_openedGenerated_
                            insideConstrainedOut = d_11_openedInside_
                            currentConstrainedOut = d_12_openedCurrent_
                            d_5_phase_ = 1
                            d_6_spanSteps_ = 0
                            d_7_steps_ = (d_7_steps_) + (1)
                        elif True:
                            d_13_next_: _dafny.Seq
                            out4_: _dafny.Seq
                            out4_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_13_next_ = out4_
                            d_7_steps_ = (d_7_steps_) + (1)
                            if (d_13_next_) == (eosToken):
                                d_5_phase_ = 1
                            elif ((d_13_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))) or ((d_13_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")))):
                                d_3_reasonTokens_ = (d_3_reasonTokens_) + (1)
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_13_next_]))
                                d_3_reasonTokens_ = (d_3_reasonTokens_) + (1)
                                if ((((((((d_13_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Final")))) or ((d_13_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "final"))))) or ((d_13_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Answer"))))) or ((d_13_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "answer"))))) or ((d_13_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Therefore"))))) or ((d_13_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "therefore"))))) or ((d_13_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Thus"))))) or ((d_13_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "thus")))):
                                    d_4_sawAnswerCue_ = True
                    elif not(insideConstrainedOut):
                        d_14_openedGenerated2_: _dafny.Seq
                        d_15_openedInside2_: bool
                        d_16_openedCurrent2_: _dafny.Seq
                        out5_: _dafny.Seq
                        out6_: bool
                        out7_: _dafny.Seq
                        out5_, out6_, out7_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                        d_14_openedGenerated2_ = out5_
                        d_15_openedInside2_ = out6_
                        d_16_openedCurrent2_ = out7_
                        generated = d_14_openedGenerated2_
                        insideConstrainedOut = d_15_openedInside2_
                        currentConstrainedOut = d_16_openedCurrent2_
                        d_6_spanSteps_ = 0
                        d_7_steps_ = (d_7_steps_) + (1)
                    elif ((parser).IsCompletePrefix(currentConstrainedOut)) and ((len(currentConstrainedOut)) > (0)):
                        d_17_closedGenerated_: _dafny.Seq
                        d_18_closedInside_: bool
                        d_19_closedCurrent_: _dafny.Seq
                        out8_: _dafny.Seq
                        out9_: bool
                        out10_: _dafny.Seq
                        out8_, out9_, out10_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_17_closedGenerated_ = out8_
                        d_18_closedInside_ = out9_
                        d_19_closedCurrent_ = out10_
                        generated = d_17_closedGenerated_
                        insideConstrainedOut = d_18_closedInside_
                        currentConstrainedOut = d_19_closedCurrent_
                        d_7_steps_ = (d_7_steps_) + (1)
                        d_5_phase_ = 2
                        raise _dafny.Break("0")
                    elif True:
                        d_20_remaining_: int
                        d_20_remaining_ = (maxSteps) - (d_7_steps_)
                        if ((d_20_remaining_) <= (1)) or ((d_6_spanSteps_) >= (120)):
                            raise _dafny.Break("0")
                        elif True:
                            d_21_chunkBudget_: int
                            d_21_chunkBudget_ = 24
                            if (d_6_spanSteps_) >= (48):
                                d_21_chunkBudget_ = 12
                            if (d_6_spanSteps_) >= (84):
                                d_21_chunkBudget_ = 8
                            d_22_availableForChunk_: int
                            d_22_availableForChunk_ = (d_20_remaining_) - (1)
                            if (d_22_availableForChunk_) < (d_21_chunkBudget_):
                                d_21_chunkBudget_ = d_22_availableForChunk_
                            d_23_constrainedPrompt_: _dafny.Seq
                            d_23_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_24_generatedOut_: _dafny.Seq
                            d_25_currentOut_: _dafny.Seq
                            d_26_hitEos_: bool
                            d_27_stepsUsed_: int
                            out11_: _dafny.Seq
                            out12_: _dafny.Seq
                            out13_: bool
                            out14_: int
                            out11_, out12_, out13_, out14_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_23_constrainedPrompt_, generated, currentConstrainedOut, d_21_chunkBudget_, eosToken)
                            d_24_generatedOut_ = out11_
                            d_25_currentOut_ = out12_
                            d_26_hitEos_ = out13_
                            d_27_stepsUsed_ = out14_
                            generated = d_24_generatedOut_
                            currentConstrainedOut = d_25_currentOut_
                            d_7_steps_ = (d_7_steps_) + (d_27_stepsUsed_)
                            d_6_spanSteps_ = (d_6_spanSteps_) + (d_27_stepsUsed_)
                            if (d_27_stepsUsed_) == (0):
                                raise _dafny.Break("0")
                            if ((d_26_hitEos_) and ((d_25_currentOut_) == (currentConstrainedOut))) and (not((parser).IsCompletePrefix(currentConstrainedOut))):
                                raise _dafny.Break("0")
                    pass
            pass
        cost = d_7_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

