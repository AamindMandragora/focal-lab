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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step. Put every intermediate symbolic arithmetic expression inside visible << >> delimiters, and put the final numeric answer inside << >> as well. Outside delimiters, write concise explanatory prose. Inside delimiters, output only the symbolic expression or final answer, with no prose."))
        if (len(validTokenGroups)) > (0):
            d_1_guidance_ = (d_1_guidance_) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " Prefer caller-provided arithmetic/context token groups when they are parser-valid.")))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_arithmeticGroups_: _dafny.Seq
        d_2_arithmeticGroups_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "0")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "1")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "2")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "3")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "4"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "5")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "6")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "7")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "8")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "9"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "+")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "-")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "*")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "/")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "="))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "(")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ")")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ".")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ","))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "####"))])])
        d_3_boostedGroups_: _dafny.Seq
        d_3_boostedGroups_ = (d_2_arithmeticGroups_) + (validTokenGroups)
        d_4_steps_: int
        d_4_steps_ = 0
        d_5_freeSinceSpan_: int
        d_5_freeSinceSpan_ = 0
        d_6_forceAfter_: int
        d_6_forceAfter_ = 48
        d_7_narrowThreshold_: int
        d_7_narrowThreshold_ = 12
        with _dafny.label("0"):
            while (d_4_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if (d_5_freeSinceSpan_) >= (d_6_forceAfter_):
                            d_8_openedGenerated_: _dafny.Seq
                            d_9_openedInside_: bool
                            d_10_openedCurrent_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_8_openedGenerated_ = out0_
                            d_9_openedInside_ = out1_
                            d_10_openedCurrent_ = out2_
                            generated = d_8_openedGenerated_
                            insideConstrainedOut = d_9_openedInside_
                            currentConstrainedOut = d_10_openedCurrent_
                            d_4_steps_ = (d_4_steps_) + (1)
                            d_5_freeSinceSpan_ = 0
                        elif True:
                            d_11_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_11_next_ = out3_
                            d_4_steps_ = (d_4_steps_) + (1)
                            if (d_11_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_11_next_]))
                                if (d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    d_12_enteredGenerated_: _dafny.Seq
                                    d_13_enteredInside_: bool
                                    d_14_enteredCurrent_: _dafny.Seq
                                    out4_: _dafny.Seq
                                    out5_: bool
                                    out6_: _dafny.Seq
                                    out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                    d_12_enteredGenerated_ = out4_
                                    d_13_enteredInside_ = out5_
                                    d_14_enteredCurrent_ = out6_
                                    generated = d_12_enteredGenerated_
                                    insideConstrainedOut = d_13_enteredInside_
                                    currentConstrainedOut = d_14_enteredCurrent_
                                    d_5_freeSinceSpan_ = 0
                                elif True:
                                    d_5_freeSinceSpan_ = (d_5_freeSinceSpan_) + (1)
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_15_closedGenerated_: _dafny.Seq
                        d_16_closedInside_: bool
                        d_17_closedCurrent_: _dafny.Seq
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: _dafny.Seq
                        out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_15_closedGenerated_ = out7_
                        d_16_closedInside_ = out8_
                        d_17_closedCurrent_ = out9_
                        generated = d_15_closedGenerated_
                        insideConstrainedOut = d_16_closedInside_
                        currentConstrainedOut = d_17_closedCurrent_
                        d_4_steps_ = (d_4_steps_) + (1)
                        d_5_freeSinceSpan_ = 0
                    elif True:
                        d_18_constrainedPrompt_: _dafny.Seq
                        d_18_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_19_validCount_: int
                        out10_: int
                        out10_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                        d_19_validCount_ = out10_
                        d_20_next_: _dafny.Seq
                        d_20_next_ = eosToken
                        if (d_19_validCount_) <= (d_7_narrowThreshold_):
                            out11_: _dafny.Seq
                            out11_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_18_constrainedPrompt_, currentConstrainedOut, d_3_boostedGroups_, _dafny.BigRational('5e0'), d_7_narrowThreshold_, eosToken)
                            d_20_next_ = out11_
                        elif True:
                            out12_: _dafny.Seq
                            out12_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_18_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('2e0'), eosToken)
                            d_20_next_ = out12_
                        d_4_steps_ = (d_4_steps_) + (1)
                        if (d_20_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_21_appendedGenerated_: _dafny.Seq
                            d_22_appendedInside_: bool
                            d_23_appendedCurrent_: _dafny.Seq
                            out13_: _dafny.Seq
                            out14_: bool
                            out15_: _dafny.Seq
                            out13_, out14_, out15_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_20_next_)
                            d_21_appendedGenerated_ = out13_
                            d_22_appendedInside_ = out14_
                            d_23_appendedCurrent_ = out15_
                            generated = d_21_appendedGenerated_
                            insideConstrainedOut = d_22_appendedInside_
                            currentConstrainedOut = d_23_appendedCurrent_
                    pass
            pass
        cost = d_4_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

