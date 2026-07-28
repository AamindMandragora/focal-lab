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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step in concise prose. Wrap every intermediate symbolic arithmetic expression and the final answer inside visible << >> delimiters. Inside each span, write only compact arithmetic syntax: numbers, variables, parentheses, and operators such as +, -, *, /, //, =. Do not put words, units, explanations, or delimiter tokens inside a constrained span. After the final answer span, stop."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_mathGroups_: _dafny.Seq
        d_2_mathGroups_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "0")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "1")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "2")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "3")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "4")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "5")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "6")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "7")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "8")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "9"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "+")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "-")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "*")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "/")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "//")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "="))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "(")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ")")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ".")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "$")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "%"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "n")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "x")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "y")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "m")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "k")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "s")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "w")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "r")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "t")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "d")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "total"))])])
        d_3_delimiterPenalty_: _dafny.Seq
        d_3_delimiterPenalty_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\n")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\n\n")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " "))])
        d_4_steps_: int
        d_4_steps_ = 0
        d_5_forceAfter_: int
        d_5_forceAfter_ = 120
        d_6_minReasoningBeforeCue_: int
        d_6_minReasoningBeforeCue_ = 24
        d_7_seenAnswerCue_: bool
        d_7_seenAnswerCue_ = False
        d_8_forcedFinalSpan_: bool
        d_8_forcedFinalSpan_ = False
        with _dafny.label("0"):
            while (d_4_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if ((d_4_steps_) >= (d_5_forceAfter_)) or ((d_7_seenAnswerCue_) and ((d_4_steps_) >= (d_6_minReasoningBeforeCue_))):
                            d_9_openedGenerated_: _dafny.Seq
                            d_10_openedInside_: bool
                            d_11_openedCurrent_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_9_openedGenerated_ = out0_
                            d_10_openedInside_ = out1_
                            d_11_openedCurrent_ = out2_
                            generated = d_9_openedGenerated_
                            insideConstrainedOut = d_10_openedInside_
                            currentConstrainedOut = d_11_openedCurrent_
                            d_8_forcedFinalSpan_ = True
                            d_4_steps_ = (d_4_steps_) + (1)
                        elif True:
                            d_12_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_12_next_ = out3_
                            d_4_steps_ = (d_4_steps_) + (1)
                            if (d_12_next_) == (eosToken):
                                if (d_4_steps_) < (maxSteps):
                                    d_13_eosOpenedGenerated_: _dafny.Seq
                                    d_14_eosOpenedInside_: bool
                                    d_15_eosOpenedCurrent_: _dafny.Seq
                                    out4_: _dafny.Seq
                                    out5_: bool
                                    out6_: _dafny.Seq
                                    out4_, out5_, out6_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                    d_13_eosOpenedGenerated_ = out4_
                                    d_14_eosOpenedInside_ = out5_
                                    d_15_eosOpenedCurrent_ = out6_
                                    generated = d_13_eosOpenedGenerated_
                                    insideConstrainedOut = d_14_eosOpenedInside_
                                    currentConstrainedOut = d_15_eosOpenedCurrent_
                                    d_8_forcedFinalSpan_ = True
                                    d_4_steps_ = (d_4_steps_) + (1)
                                elif True:
                                    raise _dafny.Break("0")
                            elif (d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_12_next_]))
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                            elif (d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))):
                                pass
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_12_next_]))
                                if ((((d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "answer")))) or ((d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Answer"))))) or ((d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "ANSWER"))))) or ((d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "####")))):
                                    d_7_seenAnswerCue_ = True
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_16_closedGenerated_: _dafny.Seq
                        d_17_closedInside_: bool
                        d_18_closedCurrent_: _dafny.Seq
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: _dafny.Seq
                        out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_16_closedGenerated_ = out7_
                        d_17_closedInside_ = out8_
                        d_18_closedCurrent_ = out9_
                        generated = d_16_closedGenerated_
                        insideConstrainedOut = d_17_closedInside_
                        currentConstrainedOut = d_18_closedCurrent_
                        d_4_steps_ = (d_4_steps_) + (1)
                        if ((d_8_forcedFinalSpan_) or (d_7_seenAnswerCue_)) or ((d_4_steps_) >= (d_5_forceAfter_)):
                            raise _dafny.Break("0")
                    elif True:
                        d_19_constrainedPrompt_: _dafny.Seq
                        d_19_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_20_nextConstrained_: _dafny.Seq
                        d_20_nextConstrained_ = eosToken
                        if (len(currentConstrainedOut)) >= (24):
                            out10_: _dafny.Seq
                            out10_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_19_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('25e-1'), eosToken)
                            d_20_nextConstrained_ = out10_
                        elif True:
                            d_21_combinedGroups_: _dafny.Seq
                            d_21_combinedGroups_ = (d_2_mathGroups_) + (validTokenGroups)
                            out11_: _dafny.Seq
                            out11_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_19_constrainedPrompt_, currentConstrainedOut, d_21_combinedGroups_, _dafny.BigRational('5e0'), d_3_delimiterPenalty_, _dafny.BigRational('8e0'), 64, eosToken)
                            d_20_nextConstrained_ = out11_
                        d_4_steps_ = (d_4_steps_) + (1)
                        if (d_20_nextConstrained_) == (eosToken):
                            raise _dafny.Break("0")
                        elif (d_20_nextConstrained_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                            pass
                        elif (d_20_nextConstrained_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))):
                            pass
                        elif True:
                            d_22_appendedGenerated_: _dafny.Seq
                            d_23_appendedInside_: bool
                            d_24_appendedCurrent_: _dafny.Seq
                            out12_: _dafny.Seq
                            out13_: bool
                            out14_: _dafny.Seq
                            out12_, out13_, out14_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_20_nextConstrained_)
                            d_22_appendedGenerated_ = out12_
                            d_23_appendedInside_ = out13_
                            d_24_appendedCurrent_ = out14_
                            generated = d_22_appendedGenerated_
                            insideConstrainedOut = d_23_appendedInside_
                            currentConstrainedOut = d_24_appendedCurrent_
                    pass
            pass
        cost = d_4_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

