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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step, concisely. Wrap every intermediate symbolic arithmetic expression and the final answer inside visible << >> delimiters. In ordinary prose, do not type raw << or >> except as these delimiters. Inside each span use only a concise arithmetic expression or number: no words, no units, no LaTeX, and no nested delimiters."))
        if (len(validTokenGroups)) > (0):
            d_1_guidance_ = (d_1_guidance_) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " Prefer the problem's variable names and arithmetic operators when they are valid.")))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_reasonLimit_: int
        d_2_reasonLimit_ = 60
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
        d_8_penaltyTokens_: _dafny.Seq
        d_8_penaltyTokens_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "$")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\\")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\\frac")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "frac")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "times")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "minutes")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "miles")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "dollars")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "units")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "answer")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Answer")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "/")), eosToken])
        d_9_symbolicGroups_: _dafny.Seq
        d_9_symbolicGroups_ = (validTokenGroups) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "//")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "*")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "+")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "-")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "(")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ")")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "int")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "max")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "min"))])]))
        with _dafny.label("0"):
            while ((d_6_steps_) < (maxSteps)) and ((d_5_phase_) < (2)):
                with _dafny.c_label("0"):
                    if (d_5_phase_) == (0):
                        d_10_shouldOpen_: bool
                        d_10_shouldOpen_ = (((d_3_reasonTokens_) >= (d_2_reasonLimit_)) or ((d_4_sawAnswerCue_) and ((d_3_reasonTokens_) >= (18)))) or (((d_6_steps_) + (40)) >= (maxSteps))
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
                            d_6_steps_ = (d_6_steps_) + (1)
                        elif True:
                            d_14_next_: _dafny.Seq
                            out4_: _dafny.Seq
                            out4_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_14_next_ = out4_
                            d_6_steps_ = (d_6_steps_) + (1)
                            if (d_14_next_) == (eosToken):
                                d_5_phase_ = 1
                            elif (d_14_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                d_3_reasonTokens_ = (d_3_reasonTokens_) + (1)
                            elif (d_14_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))):
                                d_3_reasonTokens_ = (d_3_reasonTokens_) + (1)
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_14_next_]))
                                d_3_reasonTokens_ = (d_3_reasonTokens_) + (1)
                                if ((((d_14_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Final")))) or ((d_14_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "final"))))) or ((d_14_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Answer"))))) or ((d_14_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "answer")))):
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
                        d_6_steps_ = (d_6_steps_) + (1)
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
                        d_6_steps_ = (d_6_steps_) + (1)
                        d_5_phase_ = 2
                        raise _dafny.Break("0")
                    elif True:
                        d_21_constrainedPrompt_: _dafny.Seq
                        d_21_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_22_nextConstrained_: _dafny.Seq
                        out11_: _dafny.Seq
                        out11_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_21_constrainedPrompt_, currentConstrainedOut, d_9_symbolicGroups_, _dafny.BigRational('5e0'), d_8_penaltyTokens_, _dafny.BigRational('7e0'), 18, eosToken)
                        d_22_nextConstrained_ = out11_
                        d_6_steps_ = (d_6_steps_) + (1)
                        if (d_22_nextConstrained_) == (eosToken):
                            if ((len(currentConstrainedOut)) > (0)) and ((d_6_steps_) < (maxSteps)):
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]))
                                insideConstrainedOut = False
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                d_6_steps_ = (d_6_steps_) + (1)
                                d_5_phase_ = 2
                            raise _dafny.Break("0")
                        elif (d_22_nextConstrained_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                            pass
                        elif (d_22_nextConstrained_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))):
                            if (len(currentConstrainedOut)) > (0):
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_22_nextConstrained_]))
                                insideConstrainedOut = False
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                d_5_phase_ = 2
                                raise _dafny.Break("0")
                        elif True:
                            d_23_appendedGenerated_: _dafny.Seq
                            d_24_appendedInside_: bool
                            d_25_appendedCurrent_: _dafny.Seq
                            out12_: _dafny.Seq
                            out13_: bool
                            out14_: _dafny.Seq
                            out12_, out13_, out14_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_22_nextConstrained_)
                            d_23_appendedGenerated_ = out12_
                            d_24_appendedInside_ = out13_
                            d_25_appendedCurrent_ = out14_
                            generated = d_23_appendedGenerated_
                            insideConstrainedOut = d_24_appendedInside_
                            currentConstrainedOut = d_25_appendedCurrent_
                    pass
            pass
        cost = d_6_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

