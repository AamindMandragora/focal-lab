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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step, but be concise. Use visible delimiters exactly like <<expression>> for symbolic arithmetic expressions and the final answer. In ordinary prose, do not type raw << or >>. Inside << >> put only a short algebraic expression or number: no words, no units, no LaTeX, and no nested delimiters. End with a final answer span."))
        if (len(validTokenGroups)) > (0):
            d_1_guidance_ = (d_1_guidance_) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " Prefer the problem's variable names and arithmetic operators.")))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_reasonLimit_: int
        d_2_reasonLimit_ = 70
        d_3_reasonTokens_: int
        d_3_reasonTokens_ = 0
        d_4_sawAnswerCue_: bool
        d_4_sawAnswerCue_ = False
        d_5_phase_: int
        d_5_phase_ = 0
        d_6_steps_: int
        d_6_steps_ = 0
        if (maxSteps) > (0):
            d_7_first_: _dafny.Seq
            out0_: _dafny.Seq
            out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
            d_7_first_ = out0_
            d_6_steps_ = 1
            if (d_7_first_) == (eosToken):
                d_5_phase_ = 1
            elif (d_7_first_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                d_3_reasonTokens_ = 1
            elif (d_7_first_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))):
                d_3_reasonTokens_ = 1
            elif True:
                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_7_first_]))
                d_3_reasonTokens_ = 1
                if ((((d_7_first_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Final")))) or ((d_7_first_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "final"))))) or ((d_7_first_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Answer"))))) or ((d_7_first_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "answer")))):
                    d_4_sawAnswerCue_ = True
        with _dafny.label("0"):
            while ((d_6_steps_) < (maxSteps)) and ((d_5_phase_) < (2)):
                with _dafny.c_label("0"):
                    if (d_5_phase_) == (0):
                        d_8_shouldOpen_: bool
                        d_8_shouldOpen_ = (((d_3_reasonTokens_) >= (d_2_reasonLimit_)) or ((d_4_sawAnswerCue_) and ((d_3_reasonTokens_) >= (24)))) or (((d_6_steps_) + (42)) >= (maxSteps))
                        if d_8_shouldOpen_:
                            d_9_openedGenerated_: _dafny.Seq
                            d_10_openedInside_: bool
                            d_11_openedCurrent_: _dafny.Seq
                            out1_: _dafny.Seq
                            out2_: bool
                            out3_: _dafny.Seq
                            out1_, out2_, out3_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_9_openedGenerated_ = out1_
                            d_10_openedInside_ = out2_
                            d_11_openedCurrent_ = out3_
                            generated = d_9_openedGenerated_
                            insideConstrainedOut = d_10_openedInside_
                            currentConstrainedOut = d_11_openedCurrent_
                            d_5_phase_ = 1
                            d_6_steps_ = (d_6_steps_) + (1)
                        elif True:
                            d_12_next_: _dafny.Seq
                            out4_: _dafny.Seq
                            out4_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_12_next_ = out4_
                            d_6_steps_ = (d_6_steps_) + (1)
                            if (d_12_next_) == (eosToken):
                                d_5_phase_ = 1
                            elif (d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                d_3_reasonTokens_ = (d_3_reasonTokens_) + (1)
                            elif (d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))):
                                d_3_reasonTokens_ = (d_3_reasonTokens_) + (1)
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_12_next_]))
                                d_3_reasonTokens_ = (d_3_reasonTokens_) + (1)
                                if ((((d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Final")))) or ((d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "final"))))) or ((d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Answer"))))) or ((d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "answer")))):
                                    d_4_sawAnswerCue_ = True
                    elif not(insideConstrainedOut):
                        d_13_openedGenerated2_: _dafny.Seq
                        d_14_openedInside2_: bool
                        d_15_openedCurrent2_: _dafny.Seq
                        out5_: _dafny.Seq
                        out6_: bool
                        out7_: _dafny.Seq
                        out5_, out6_, out7_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                        d_13_openedGenerated2_ = out5_
                        d_14_openedInside2_ = out6_
                        d_15_openedCurrent2_ = out7_
                        generated = d_13_openedGenerated2_
                        insideConstrainedOut = d_14_openedInside2_
                        currentConstrainedOut = d_15_openedCurrent2_
                        d_6_steps_ = (d_6_steps_) + (1)
                    elif ((parser).IsCompletePrefix(currentConstrainedOut)) and ((len(currentConstrainedOut)) > (0)):
                        d_16_closedGenerated_: _dafny.Seq
                        d_17_closedInside_: bool
                        d_18_closedCurrent_: _dafny.Seq
                        out8_: _dafny.Seq
                        out9_: bool
                        out10_: _dafny.Seq
                        out8_, out9_, out10_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_16_closedGenerated_ = out8_
                        d_17_closedInside_ = out9_
                        d_18_closedCurrent_ = out10_
                        generated = d_16_closedGenerated_
                        insideConstrainedOut = d_17_closedInside_
                        currentConstrainedOut = d_18_closedCurrent_
                        d_6_steps_ = (d_6_steps_) + (1)
                        d_5_phase_ = 2
                        raise _dafny.Break("0")
                    elif True:
                        d_19_remaining_: int
                        d_19_remaining_ = (maxSteps) - (d_6_steps_)
                        if (d_19_remaining_) <= (1):
                            raise _dafny.Break("0")
                        elif True:
                            d_20_chunkBudget_: int
                            d_20_chunkBudget_ = 32
                            d_21_availableForChunk_: int
                            d_21_availableForChunk_ = (d_19_remaining_) - (1)
                            if (d_21_availableForChunk_) < (d_20_chunkBudget_):
                                d_20_chunkBudget_ = d_21_availableForChunk_
                            d_22_constrainedPrompt_: _dafny.Seq
                            d_22_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_23_generatedOut_: _dafny.Seq
                            d_24_currentOut_: _dafny.Seq
                            d_25_hitEos_: bool
                            d_26_stepsUsed_: int
                            out11_: _dafny.Seq
                            out12_: _dafny.Seq
                            out13_: bool
                            out14_: int
                            out11_, out12_, out13_, out14_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_22_constrainedPrompt_, generated, currentConstrainedOut, d_20_chunkBudget_, eosToken)
                            d_23_generatedOut_ = out11_
                            d_24_currentOut_ = out12_
                            d_25_hitEos_ = out13_
                            d_26_stepsUsed_ = out14_
                            generated = d_23_generatedOut_
                            currentConstrainedOut = d_24_currentOut_
                            d_6_steps_ = (d_6_steps_) + (d_26_stepsUsed_)
                            if (d_26_stepsUsed_) == (0):
                                raise _dafny.Break("0")
                    pass
            pass
        cost = d_6_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

