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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step, concisely. Use visible delimiters exactly like <<expression>> for symbolic arithmetic expressions and for the final answer. When you open <<, immediately write only one compact arithmetic expression or number and close it with >>. Do not nest delimiters and do not put words, units, Markdown, or LaTeX inside delimiters. Use Python-style arithmetic, especially // for exact integer division. End with a final answer span containing the answer expression."))
        if (len(validTokenGroups)) > (0):
            d_1_guidance_ = (d_1_guidance_) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " Prefer the problem's variable names, numbers, and arithmetic operators when forming expressions.")))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_reasonLimit_: int
        d_2_reasonLimit_ = 40
        d_3_reasonTokens_: int
        d_3_reasonTokens_ = 0
        d_4_sawAnswerCue_: bool
        d_4_sawAnswerCue_ = False
        d_5_phase_: int
        d_5_phase_ = 0
        d_6_spanIsFinal_: bool
        d_6_spanIsFinal_ = False
        d_7_spanSteps_: int
        d_7_spanSteps_ = 0
        d_8_closedSpans_: int
        d_8_closedSpans_ = 0
        d_9_steps_: int
        d_9_steps_ = 0
        if (maxSteps) > (0):
            d_10_first_: _dafny.Seq
            out0_: _dafny.Seq
            out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
            d_10_first_ = out0_
            d_9_steps_ = 1
            if (d_10_first_) == (eosToken):
                d_5_phase_ = 1
                d_6_spanIsFinal_ = True
            elif (d_10_first_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_10_first_]))
                d_3_reasonTokens_ = 1
                d_11_enteredGenerated0_: _dafny.Seq
                d_12_enteredInside0_: bool
                d_13_enteredCurrent0_: _dafny.Seq
                out1_: _dafny.Seq
                out2_: bool
                out3_: _dafny.Seq
                out1_, out2_, out3_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                d_11_enteredGenerated0_ = out1_
                d_12_enteredInside0_ = out2_
                d_13_enteredCurrent0_ = out3_
                generated = d_11_enteredGenerated0_
                insideConstrainedOut = d_12_enteredInside0_
                currentConstrainedOut = d_13_enteredCurrent0_
                d_5_phase_ = 1
                d_6_spanIsFinal_ = False
                d_7_spanSteps_ = 0
            elif (d_10_first_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))):
                d_3_reasonTokens_ = 1
            elif True:
                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_10_first_]))
                d_3_reasonTokens_ = 1
                if (((((((((((d_10_first_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Final")))) or ((d_10_first_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "final"))))) or ((d_10_first_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Answer"))))) or ((d_10_first_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "answer"))))) or ((d_10_first_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Therefore"))))) or ((d_10_first_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "therefore"))))) or ((d_10_first_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Thus"))))) or ((d_10_first_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "thus"))))) or ((d_10_first_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Finally"))))) or ((d_10_first_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "finally"))))) or ((d_10_first_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "####")))):
                    d_4_sawAnswerCue_ = True
        with _dafny.label("0"):
            while ((d_9_steps_) < (maxSteps)) and ((d_5_phase_) < (2)):
                with _dafny.c_label("0"):
                    if (d_5_phase_) == (0):
                        d_14_shouldOpen_: bool
                        d_14_shouldOpen_ = (((d_3_reasonTokens_) >= (d_2_reasonLimit_)) or ((d_4_sawAnswerCue_) and ((d_3_reasonTokens_) >= (14)))) or (((d_9_steps_) + (76)) >= (maxSteps))
                        if d_14_shouldOpen_:
                            d_15_openedGenerated_: _dafny.Seq
                            d_16_openedInside_: bool
                            d_17_openedCurrent_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_15_openedGenerated_ = out4_
                            d_16_openedInside_ = out5_
                            d_17_openedCurrent_ = out6_
                            generated = d_15_openedGenerated_
                            insideConstrainedOut = d_16_openedInside_
                            currentConstrainedOut = d_17_openedCurrent_
                            d_5_phase_ = 1
                            d_6_spanIsFinal_ = True
                            d_7_spanSteps_ = 0
                            d_9_steps_ = (d_9_steps_) + (1)
                        elif True:
                            d_18_next_: _dafny.Seq
                            out7_: _dafny.Seq
                            out7_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_18_next_ = out7_
                            d_9_steps_ = (d_9_steps_) + (1)
                            if (d_18_next_) == (eosToken):
                                d_5_phase_ = 1
                                d_6_spanIsFinal_ = True
                            elif (d_18_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_18_next_]))
                                d_3_reasonTokens_ = (d_3_reasonTokens_) + (1)
                                d_19_enteredGenerated_: _dafny.Seq
                                d_20_enteredInside_: bool
                                d_21_enteredCurrent_: _dafny.Seq
                                out8_: _dafny.Seq
                                out9_: bool
                                out10_: _dafny.Seq
                                out8_, out9_, out10_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_19_enteredGenerated_ = out8_
                                d_20_enteredInside_ = out9_
                                d_21_enteredCurrent_ = out10_
                                generated = d_19_enteredGenerated_
                                insideConstrainedOut = d_20_enteredInside_
                                currentConstrainedOut = d_21_enteredCurrent_
                                d_5_phase_ = 1
                                d_6_spanIsFinal_ = ((d_4_sawAnswerCue_) or ((d_3_reasonTokens_) >= (18))) or (((d_9_steps_) + (76)) >= (maxSteps))
                                d_7_spanSteps_ = 0
                            elif (d_18_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))):
                                d_3_reasonTokens_ = (d_3_reasonTokens_) + (1)
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_18_next_]))
                                d_3_reasonTokens_ = (d_3_reasonTokens_) + (1)
                                if (((((((((((d_18_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Final")))) or ((d_18_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "final"))))) or ((d_18_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Answer"))))) or ((d_18_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "answer"))))) or ((d_18_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Therefore"))))) or ((d_18_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "therefore"))))) or ((d_18_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Thus"))))) or ((d_18_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "thus"))))) or ((d_18_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Finally"))))) or ((d_18_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "finally"))))) or ((d_18_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "####")))):
                                    d_4_sawAnswerCue_ = True
                    elif not(insideConstrainedOut):
                        d_22_openedGenerated2_: _dafny.Seq
                        d_23_openedInside2_: bool
                        d_24_openedCurrent2_: _dafny.Seq
                        out11_: _dafny.Seq
                        out12_: bool
                        out13_: _dafny.Seq
                        out11_, out12_, out13_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                        d_22_openedGenerated2_ = out11_
                        d_23_openedInside2_ = out12_
                        d_24_openedCurrent2_ = out13_
                        generated = d_22_openedGenerated2_
                        insideConstrainedOut = d_23_openedInside2_
                        currentConstrainedOut = d_24_openedCurrent2_
                        d_6_spanIsFinal_ = True
                        d_7_spanSteps_ = 0
                        d_9_steps_ = (d_9_steps_) + (1)
                    elif ((parser).IsCompletePrefix(currentConstrainedOut)) and ((len(currentConstrainedOut)) > (0)):
                        d_25_wasFinal_: bool
                        d_25_wasFinal_ = d_6_spanIsFinal_
                        d_26_closedGenerated_: _dafny.Seq
                        d_27_closedInside_: bool
                        d_28_closedCurrent_: _dafny.Seq
                        out14_: _dafny.Seq
                        out15_: bool
                        out16_: _dafny.Seq
                        out14_, out15_, out16_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_26_closedGenerated_ = out14_
                        d_27_closedInside_ = out15_
                        d_28_closedCurrent_ = out16_
                        generated = d_26_closedGenerated_
                        insideConstrainedOut = d_27_closedInside_
                        currentConstrainedOut = d_28_closedCurrent_
                        d_9_steps_ = (d_9_steps_) + (1)
                        d_8_closedSpans_ = (d_8_closedSpans_) + (1)
                        d_7_spanSteps_ = 0
                        if d_25_wasFinal_:
                            d_5_phase_ = 2
                            raise _dafny.Break("0")
                        elif True:
                            d_5_phase_ = 0
                            if (d_8_closedSpans_) >= (2):
                                d_4_sawAnswerCue_ = True
                    elif True:
                        d_29_remaining_: int
                        d_29_remaining_ = (maxSteps) - (d_9_steps_)
                        if ((d_7_spanSteps_) >= (34)) or ((d_29_remaining_) <= (1)):
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]))
                            insideConstrainedOut = False
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                            d_9_steps_ = (d_9_steps_) + (1)
                            d_8_closedSpans_ = (d_8_closedSpans_) + (1)
                            d_7_spanSteps_ = 0
                            if d_6_spanIsFinal_:
                                d_5_phase_ = 2
                                raise _dafny.Break("0")
                            elif True:
                                d_5_phase_ = 0
                                d_4_sawAnswerCue_ = True
                        elif True:
                            d_30_chunkBudget_: int
                            d_30_chunkBudget_ = 8
                            if not(d_6_spanIsFinal_):
                                d_30_chunkBudget_ = 5
                            if (d_7_spanSteps_) >= (16):
                                d_30_chunkBudget_ = 1
                            d_31_availableForChunk_: int
                            d_31_availableForChunk_ = (d_29_remaining_) - (1)
                            if (d_31_availableForChunk_) < (d_30_chunkBudget_):
                                d_30_chunkBudget_ = d_31_availableForChunk_
                            d_32_constrainedPrompt_: _dafny.Seq
                            d_32_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_33_generatedOut_: _dafny.Seq
                            d_34_currentOut_: _dafny.Seq
                            d_35_hitEos_: bool
                            d_36_stepsUsed_: int
                            out17_: _dafny.Seq
                            out18_: _dafny.Seq
                            out19_: bool
                            out20_: int
                            out17_, out18_, out19_, out20_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_32_constrainedPrompt_, generated, currentConstrainedOut, d_30_chunkBudget_, eosToken)
                            d_33_generatedOut_ = out17_
                            d_34_currentOut_ = out18_
                            d_35_hitEos_ = out19_
                            d_36_stepsUsed_ = out20_
                            generated = d_33_generatedOut_
                            currentConstrainedOut = d_34_currentOut_
                            d_9_steps_ = (d_9_steps_) + (d_36_stepsUsed_)
                            d_7_spanSteps_ = (d_7_spanSteps_) + (d_36_stepsUsed_)
                            if (d_36_stepsUsed_) == (0):
                                raise _dafny.Break("0")
                    pass
            pass
        cost = d_9_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

