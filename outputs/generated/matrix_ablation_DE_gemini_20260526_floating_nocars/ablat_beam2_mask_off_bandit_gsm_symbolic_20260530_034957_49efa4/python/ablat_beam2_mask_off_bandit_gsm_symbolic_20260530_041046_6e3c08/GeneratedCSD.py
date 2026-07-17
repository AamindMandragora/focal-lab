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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step. Put every intermediate symbolic arithmetic expression inside visible << >> delimiters, and put the final answer inside << >> as well. Outside delimiters, write concise explanatory prose. Inside << >> output only a symbolic expression or numeric answer, with no prose, no units, no equals sign, and no extra delimiter tokens."))
        if (len(validTokenGroups)) > (0):
            d_1_guidance_ = (d_1_guidance_) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " Prefer parser-valid caller-provided number, variable, and operator tokens when they fit the expression.")))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_digitGroups_: _dafny.Seq
        d_2_digitGroups_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "0")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "1")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "2")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "3")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "4"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "5")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "6")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "7")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "8")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "9"))])])
        d_3_operatorGroups_: _dafny.Seq
        d_3_operatorGroups_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "+")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "-")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "*")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "/")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "//"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "(")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ")"))])])
        d_4_variableGroups_: _dafny.Seq
        d_4_variableGroups_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "x")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "y")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "z")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "n")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "m")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "k")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "t")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "r")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "w")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "p")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "s"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "total")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "amount")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "rate")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "time")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "cost")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "price"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "n1")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "n2")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "n3")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "n4")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "w1")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "w2")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "w3")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "w4")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "t1")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "t2"))])])
        d_5_boostedGroups_: _dafny.Seq
        d_5_boostedGroups_ = (((d_2_digitGroups_) + (d_3_operatorGroups_)) + (d_4_variableGroups_)) + (validTokenGroups)
        d_6_penaltyTokens_: _dafny.Seq
        d_6_penaltyTokens_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "####")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "{")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "}")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "!")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "!!")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "!!!")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ".")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ",")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ":")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ";")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\n")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "seconds")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "second")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "minutes")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "minute")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "hours")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "hour")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "feet")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "foot")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "inch")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "inches")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "pounds")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "pound")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "cups")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "cup")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "teaspoons")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "teaspoon")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "toys")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "blocks")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "kernels")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "pieces")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "plants")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "windows")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "bags")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "boxes")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "is")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "are")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "was")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "were")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "the")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "of")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "and")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "in")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "to")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "from")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "answer")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Answer")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "final")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Final"))])
        d_7_proseGapLimit_: int
        d_7_proseGapLimit_ = 64
        d_8_proseSinceSpan_: int
        d_8_proseSinceSpan_ = 0
        d_9_spanSteps_: int
        d_9_spanSteps_ = 0
        d_10_narrowThreshold_: int
        d_10_narrowThreshold_ = 100000
        d_11_steps_: int
        d_11_steps_ = 0
        with _dafny.label("0"):
            while (d_11_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if (d_8_proseSinceSpan_) >= (d_7_proseGapLimit_):
                            d_12_openedGenerated_: _dafny.Seq
                            d_13_openedInside_: bool
                            d_14_openedCurrent_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_12_openedGenerated_ = out0_
                            d_13_openedInside_ = out1_
                            d_14_openedCurrent_ = out2_
                            generated = d_12_openedGenerated_
                            insideConstrainedOut = d_13_openedInside_
                            currentConstrainedOut = d_14_openedCurrent_
                            d_8_proseSinceSpan_ = 0
                            d_9_spanSteps_ = 0
                            d_11_steps_ = (d_11_steps_) + (1)
                        elif True:
                            d_15_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_15_next_ = out3_
                            d_11_steps_ = (d_11_steps_) + (1)
                            if (d_15_next_) == (eosToken):
                                if (d_11_steps_) < (maxSteps):
                                    d_16_openedGenerated2_: _dafny.Seq
                                    d_17_openedInside2_: bool
                                    d_18_openedCurrent2_: _dafny.Seq
                                    out4_: _dafny.Seq
                                    out5_: bool
                                    out6_: _dafny.Seq
                                    out4_, out5_, out6_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                    d_16_openedGenerated2_ = out4_
                                    d_17_openedInside2_ = out5_
                                    d_18_openedCurrent2_ = out6_
                                    generated = d_16_openedGenerated2_
                                    insideConstrainedOut = d_17_openedInside2_
                                    currentConstrainedOut = d_18_openedCurrent2_
                                    d_8_proseSinceSpan_ = 0
                                    d_9_spanSteps_ = 0
                                    d_11_steps_ = (d_11_steps_) + (1)
                                elif True:
                                    raise _dafny.Break("0")
                            elif (d_15_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))):
                                d_8_proseSinceSpan_ = (d_8_proseSinceSpan_) + (1)
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_15_next_]))
                                d_8_proseSinceSpan_ = (d_8_proseSinceSpan_) + (1)
                                if (d_15_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    insideConstrainedOut = True
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                    d_8_proseSinceSpan_ = 0
                                    d_9_spanSteps_ = 0
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_19_closedGenerated_: _dafny.Seq
                        d_20_closedInside_: bool
                        d_21_closedCurrent_: _dafny.Seq
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: _dafny.Seq
                        out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_19_closedGenerated_ = out7_
                        d_20_closedInside_ = out8_
                        d_21_closedCurrent_ = out9_
                        generated = d_19_closedGenerated_
                        insideConstrainedOut = d_20_closedInside_
                        currentConstrainedOut = d_21_closedCurrent_
                        d_8_proseSinceSpan_ = 0
                        d_9_spanSteps_ = 0
                        d_11_steps_ = (d_11_steps_) + (1)
                    elif True:
                        d_22_constrainedPrompt_: _dafny.Seq
                        d_22_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_23_nextIn_: _dafny.Seq
                        out10_: _dafny.Seq
                        out10_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_22_constrainedPrompt_, currentConstrainedOut, d_5_boostedGroups_, _dafny.BigRational('8e0'), d_6_penaltyTokens_, _dafny.BigRational('1e1'), d_10_narrowThreshold_, eosToken)
                        d_23_nextIn_ = out10_
                        d_11_steps_ = (d_11_steps_) + (1)
                        d_9_spanSteps_ = (d_9_spanSteps_) + (1)
                        if (d_23_nextIn_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_24_appendedGenerated_: _dafny.Seq
                            d_25_appendedInside_: bool
                            d_26_appendedCurrent_: _dafny.Seq
                            out11_: _dafny.Seq
                            out12_: bool
                            out13_: _dafny.Seq
                            out11_, out12_, out13_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_23_nextIn_)
                            d_24_appendedGenerated_ = out11_
                            d_25_appendedInside_ = out12_
                            d_26_appendedCurrent_ = out13_
                            generated = d_24_appendedGenerated_
                            insideConstrainedOut = d_25_appendedInside_
                            currentConstrainedOut = d_26_appendedCurrent_
                            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_11_steps_) < (maxSteps)):
                                d_27_closedGenerated2_: _dafny.Seq
                                d_28_closedInside2_: bool
                                d_29_closedCurrent2_: _dafny.Seq
                                out14_: _dafny.Seq
                                out15_: bool
                                out16_: _dafny.Seq
                                out14_, out15_, out16_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_27_closedGenerated2_ = out14_
                                d_28_closedInside2_ = out15_
                                d_29_closedCurrent2_ = out16_
                                generated = d_27_closedGenerated2_
                                insideConstrainedOut = d_28_closedInside2_
                                currentConstrainedOut = d_29_closedCurrent2_
                                d_8_proseSinceSpan_ = 0
                                d_9_spanSteps_ = 0
                                d_11_steps_ = (d_11_steps_) + (1)
                    pass
            pass
        cost = d_11_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

