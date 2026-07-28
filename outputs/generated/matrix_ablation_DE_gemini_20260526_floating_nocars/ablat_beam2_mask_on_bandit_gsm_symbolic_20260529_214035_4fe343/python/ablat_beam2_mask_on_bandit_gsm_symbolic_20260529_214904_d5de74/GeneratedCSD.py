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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step in concise prose. Put the final symbolic answer expression inside visible << >> delimiters. Do not put a closing delimiter inside the expression; after the expression, close exactly once with >>. Prefer compact valid arithmetic syntax such as a*b+c, n-k*x, (a+b)//c, or int(expr) when needed."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_mathGroups_: _dafny.Seq
        d_2_mathGroups_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "0")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "1")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "2")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "3")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "4")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "5")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "6")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "7")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "8")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "9"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "+")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "-")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "*")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "/")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "//")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "×")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "÷")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "="))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "(")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ")")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ".")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "$")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "%"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "int")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "ceil")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "floor"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "n")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "x")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "y")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "t")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "d")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "w")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "r")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "k")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "total"))])])
        d_3_steps_: int
        d_3_steps_ = 0
        d_4_forceAfter_: int
        d_4_forceAfter_ = 48
        with _dafny.label("0"):
            while (d_3_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_5_openCount_: int
                        out0_: int
                        out0_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))
                        d_5_openCount_ = out0_
                        d_6_closeCount_: int
                        out1_: int
                        out1_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")))
                        d_6_closeCount_ = out1_
                        if (((d_5_openCount_) == (d_6_closeCount_)) and ((d_5_openCount_) == (0))) and ((d_3_steps_) >= (d_4_forceAfter_)):
                            d_7_openedGenerated_: _dafny.Seq
                            d_8_openedInside_: bool
                            d_9_openedCurrent_: _dafny.Seq
                            out2_: _dafny.Seq
                            out3_: bool
                            out4_: _dafny.Seq
                            out2_, out3_, out4_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_7_openedGenerated_ = out2_
                            d_8_openedInside_ = out3_
                            d_9_openedCurrent_ = out4_
                            generated = d_7_openedGenerated_
                            insideConstrainedOut = d_8_openedInside_
                            currentConstrainedOut = d_9_openedCurrent_
                            d_3_steps_ = (d_3_steps_) + (1)
                        elif True:
                            d_10_next_: _dafny.Seq
                            out5_: _dafny.Seq
                            out5_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_10_next_ = out5_
                            d_3_steps_ = (d_3_steps_) + (1)
                            if (d_10_next_) == (eosToken):
                                if (((d_5_openCount_) == (d_6_closeCount_)) and ((d_5_openCount_) == (0))) and ((d_3_steps_) < (maxSteps)):
                                    d_11_eosOpenedGenerated_: _dafny.Seq
                                    d_12_eosOpenedInside_: bool
                                    d_13_eosOpenedCurrent_: _dafny.Seq
                                    out6_: _dafny.Seq
                                    out7_: bool
                                    out8_: _dafny.Seq
                                    out6_, out7_, out8_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                    d_11_eosOpenedGenerated_ = out6_
                                    d_12_eosOpenedInside_ = out7_
                                    d_13_eosOpenedCurrent_ = out8_
                                    generated = d_11_eosOpenedGenerated_
                                    insideConstrainedOut = d_12_eosOpenedInside_
                                    currentConstrainedOut = d_13_eosOpenedCurrent_
                                    d_3_steps_ = (d_3_steps_) + (1)
                                elif True:
                                    raise _dafny.Break("0")
                            elif (d_10_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_10_next_]))
                                d_14_observedGenerated_: _dafny.Seq
                                d_15_observedInside_: bool
                                d_16_observedCurrent_: _dafny.Seq
                                out9_: _dafny.Seq
                                out10_: bool
                                out11_: _dafny.Seq
                                out9_, out10_, out11_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_14_observedGenerated_ = out9_
                                d_15_observedInside_ = out10_
                                d_16_observedCurrent_ = out11_
                                generated = d_14_observedGenerated_
                                insideConstrainedOut = d_15_observedInside_
                                currentConstrainedOut = d_16_observedCurrent_
                            elif (d_10_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))):
                                pass
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_10_next_]))
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_17_closedGenerated_: _dafny.Seq
                        d_18_closedInside_: bool
                        d_19_closedCurrent_: _dafny.Seq
                        out12_: _dafny.Seq
                        out13_: bool
                        out14_: _dafny.Seq
                        out12_, out13_, out14_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_17_closedGenerated_ = out12_
                        d_18_closedInside_ = out13_
                        d_19_closedCurrent_ = out14_
                        generated = d_17_closedGenerated_
                        insideConstrainedOut = d_18_closedInside_
                        currentConstrainedOut = d_19_closedCurrent_
                        d_3_steps_ = (d_3_steps_) + (1)
                    elif True:
                        d_20_constrainedPrompt_: _dafny.Seq
                        d_20_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_21_combinedGroups_: _dafny.Seq
                        d_21_combinedGroups_ = (d_2_mathGroups_) + (validTokenGroups)
                        d_22_penaltyTokens_: _dafny.Seq
                        d_22_penaltyTokens_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\n")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\n\n")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " "))])
                        d_23_nextConstrained_: _dafny.Seq
                        out15_: _dafny.Seq
                        out15_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_20_constrainedPrompt_, currentConstrainedOut, d_21_combinedGroups_, _dafny.BigRational('5e0'), d_22_penaltyTokens_, _dafny.BigRational('6e0'), 64, eosToken)
                        d_23_nextConstrained_ = out15_
                        d_3_steps_ = (d_3_steps_) + (1)
                        if (d_23_nextConstrained_) == (eosToken):
                            if (d_3_steps_) < (maxSteps):
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]))
                                insideConstrainedOut = False
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                d_3_steps_ = (d_3_steps_) + (1)
                            elif True:
                                raise _dafny.Break("0")
                        elif (d_23_nextConstrained_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))):
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_23_nextConstrained_]))
                            insideConstrainedOut = False
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                        elif (d_23_nextConstrained_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                            pass
                        elif True:
                            d_24_appendedGenerated_: _dafny.Seq
                            d_25_appendedInside_: bool
                            d_26_appendedCurrent_: _dafny.Seq
                            out16_: _dafny.Seq
                            out17_: bool
                            out18_: _dafny.Seq
                            out16_, out17_, out18_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_23_nextConstrained_)
                            d_24_appendedGenerated_ = out16_
                            d_25_appendedInside_ = out17_
                            d_26_appendedCurrent_ = out18_
                            generated = d_24_appendedGenerated_
                            insideConstrainedOut = d_25_appendedInside_
                            currentConstrainedOut = d_26_appendedCurrent_
                    pass
            pass
        cost = d_3_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

