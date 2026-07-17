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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step, concisely. Do ordinary reasoning first if useful, then give the final symbolic answer in one visible <<expression>> span. Inside << >> write only a compact arithmetic expression or number: no words, no units, no LaTeX, no nested delimiters. Use integer arithmetic forms such as // or int(...) when the problem asks for whole-number counts or percentages."))
        if (len(validTokenGroups)) > (0):
            d_1_guidance_ = (d_1_guidance_) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " Prefer the problem's variable names, numbers, and arithmetic operators when forming the final expression.")))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_reasonLimit_: int
        d_2_reasonLimit_ = 60
        d_3_cueDelay_: int
        d_3_cueDelay_ = 18
        d_4_spanCap_: int
        d_4_spanCap_ = 72
        d_5_freeTokens_: int
        d_5_freeTokens_ = 0
        d_6_sawAnswerCue_: bool
        d_6_sawAnswerCue_ = False
        d_7_phase_: int
        d_7_phase_ = 0
        d_8_spanSteps_: int
        d_8_spanSteps_ = 0
        d_9_spanBase_: _dafny.Seq
        d_9_spanBase_ = generated
        d_10_steps_: int
        d_10_steps_ = 0
        if (maxSteps) > (0):
            d_11_first_: _dafny.Seq
            out0_: _dafny.Seq
            out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
            d_11_first_ = out0_
            d_10_steps_ = 1
            d_5_freeTokens_ = 1
            if (d_11_first_) == (eosToken):
                d_7_phase_ = 1
            elif ((d_11_first_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))) or ((d_11_first_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")))):
                d_7_phase_ = 0
            elif True:
                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_11_first_]))
                if (((((((((d_11_first_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Final")))) or ((d_11_first_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "final"))))) or ((d_11_first_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Answer"))))) or ((d_11_first_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "answer"))))) or ((d_11_first_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Therefore"))))) or ((d_11_first_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "therefore"))))) or ((d_11_first_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Thus"))))) or ((d_11_first_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "thus"))))) or ((d_11_first_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "####")))):
                    d_6_sawAnswerCue_ = True
        with _dafny.label("0"):
            while ((d_10_steps_) < (maxSteps)) and ((d_7_phase_) < (2)):
                with _dafny.c_label("0"):
                    if (d_7_phase_) == (0):
                        d_12_shouldOpen_: bool
                        d_12_shouldOpen_ = (((d_5_freeTokens_) >= (d_2_reasonLimit_)) or ((d_6_sawAnswerCue_) and ((d_5_freeTokens_) >= (d_3_cueDelay_)))) or (((d_10_steps_) + (80)) >= (maxSteps))
                        if d_12_shouldOpen_:
                            d_9_spanBase_ = generated
                            d_13_openedGenerated_: _dafny.Seq
                            d_14_openedInside_: bool
                            d_15_openedCurrent_: _dafny.Seq
                            out1_: _dafny.Seq
                            out2_: bool
                            out3_: _dafny.Seq
                            out1_, out2_, out3_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_13_openedGenerated_ = out1_
                            d_14_openedInside_ = out2_
                            d_15_openedCurrent_ = out3_
                            generated = d_13_openedGenerated_
                            insideConstrainedOut = d_14_openedInside_
                            currentConstrainedOut = d_15_openedCurrent_
                            d_7_phase_ = 1
                            d_8_spanSteps_ = 0
                            d_10_steps_ = (d_10_steps_) + (1)
                        elif True:
                            d_16_next_: _dafny.Seq
                            out4_: _dafny.Seq
                            out4_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_16_next_ = out4_
                            d_10_steps_ = (d_10_steps_) + (1)
                            d_5_freeTokens_ = (d_5_freeTokens_) + (1)
                            if (d_16_next_) == (eosToken):
                                d_7_phase_ = 1
                            elif ((d_16_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))) or ((d_16_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")))):
                                pass
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_16_next_]))
                                if (((((((((d_16_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Final")))) or ((d_16_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "final"))))) or ((d_16_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Answer"))))) or ((d_16_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "answer"))))) or ((d_16_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Therefore"))))) or ((d_16_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "therefore"))))) or ((d_16_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Thus"))))) or ((d_16_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "thus"))))) or ((d_16_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "####")))):
                                    d_6_sawAnswerCue_ = True
                    elif not(insideConstrainedOut):
                        d_9_spanBase_ = generated
                        d_17_openedGenerated2_: _dafny.Seq
                        d_18_openedInside2_: bool
                        d_19_openedCurrent2_: _dafny.Seq
                        out5_: _dafny.Seq
                        out6_: bool
                        out7_: _dafny.Seq
                        out5_, out6_, out7_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                        d_17_openedGenerated2_ = out5_
                        d_18_openedInside2_ = out6_
                        d_19_openedCurrent2_ = out7_
                        generated = d_17_openedGenerated2_
                        insideConstrainedOut = d_18_openedInside2_
                        currentConstrainedOut = d_19_openedCurrent2_
                        d_8_spanSteps_ = 0
                        d_10_steps_ = (d_10_steps_) + (1)
                    elif ((parser).IsCompletePrefix(currentConstrainedOut)) and ((len(currentConstrainedOut)) > (0)):
                        d_20_closedGenerated_: _dafny.Seq
                        d_21_closedInside_: bool
                        d_22_closedCurrent_: _dafny.Seq
                        out8_: _dafny.Seq
                        out9_: bool
                        out10_: _dafny.Seq
                        out8_, out9_, out10_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_20_closedGenerated_ = out8_
                        d_21_closedInside_ = out9_
                        d_22_closedCurrent_ = out10_
                        generated = d_20_closedGenerated_
                        insideConstrainedOut = d_21_closedInside_
                        currentConstrainedOut = d_22_closedCurrent_
                        d_10_steps_ = (d_10_steps_) + (1)
                        d_7_phase_ = 2
                        raise _dafny.Break("0")
                    elif True:
                        d_23_remaining_: int
                        d_23_remaining_ = (maxSteps) - (d_10_steps_)
                        if (d_23_remaining_) <= (1):
                            raise _dafny.Break("0")
                        elif (d_8_spanSteps_) >= (d_4_spanCap_):
                            raise _dafny.Break("0")
                        elif True:
                            d_24_chunkBudget_: int
                            d_24_chunkBudget_ = 16
                            if (len(currentConstrainedOut)) == (0):
                                d_24_chunkBudget_ = 24
                            if (d_8_spanSteps_) >= (32):
                                d_24_chunkBudget_ = 8
                            d_25_availableForChunk_: int
                            d_25_availableForChunk_ = (d_23_remaining_) - (1)
                            if (d_25_availableForChunk_) < (d_24_chunkBudget_):
                                d_24_chunkBudget_ = d_25_availableForChunk_
                            d_26_constrainedPrompt_: _dafny.Seq
                            d_26_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_27_generatedOut_: _dafny.Seq
                            d_28_currentOut_: _dafny.Seq
                            d_29_hitEos_: bool
                            d_30_stepsUsed_: int
                            out11_: _dafny.Seq
                            out12_: _dafny.Seq
                            out13_: bool
                            out14_: int
                            out11_, out12_, out13_, out14_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_26_constrainedPrompt_, generated, currentConstrainedOut, d_24_chunkBudget_, eosToken)
                            d_27_generatedOut_ = out11_
                            d_28_currentOut_ = out12_
                            d_29_hitEos_ = out13_
                            d_30_stepsUsed_ = out14_
                            generated = d_27_generatedOut_
                            currentConstrainedOut = d_28_currentOut_
                            d_10_steps_ = (d_10_steps_) + (d_30_stepsUsed_)
                            d_8_spanSteps_ = (d_8_spanSteps_) + (d_30_stepsUsed_)
                            if (d_30_stepsUsed_) == (0):
                                raise _dafny.Break("0")
                    pass
            pass
        cost = d_10_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

