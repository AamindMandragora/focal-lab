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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. Put useful symbolic calculations and especially the final answer inside visible << >> delimiters. Inside delimiters write only compact valid arithmetic, e.g. k*y/(x*12), int((k*y)/(x*12)*100), n-k*x, or (a+b)//c."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_mathGroups_: _dafny.Seq
        d_2_mathGroups_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "0")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "1")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "2")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "3")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "4")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "5")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "6")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "7")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "8")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "9"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "+")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "-")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "*")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "/")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "//")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "(")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ")")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ".")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "%"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "int")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "ceil")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "floor"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "n")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "x")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "y")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "t")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "d")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "w")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "r")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "k")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "m")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "total")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "target")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "sides"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "n1")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "n2")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "n3")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "n4")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "w1")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "w2")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "w3")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "w4")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "t1")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "t2"))])])
        d_3_penaltyTokens_: _dafny.Seq
        d_3_penaltyTokens_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\n")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\n\n")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " ")), eosToken])
        d_4_steps_: int
        d_4_steps_ = 0
        d_5_forceAfter_: int
        d_5_forceAfter_ = 48
        d_6_localLimit_: int
        d_6_localLimit_ = 84
        d_7_forcedFinal_: bool
        d_7_forcedFinal_ = False
        with _dafny.label("0"):
            while ((d_4_steps_) < (maxSteps)) and ((d_4_steps_) < (d_6_localLimit_)):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_8_openCount_: int
                        out0_: int
                        out0_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))
                        d_8_openCount_ = out0_
                        d_9_closeCount_: int
                        out1_: int
                        out1_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")))
                        d_9_closeCount_ = out1_
                        if ((((d_8_openCount_) == (d_9_closeCount_)) and ((d_8_openCount_) == (0))) and (not(d_7_forcedFinal_))) and ((d_4_steps_) >= (d_5_forceAfter_)):
                            d_10_openedGenerated_: _dafny.Seq
                            d_11_openedInside_: bool
                            d_12_openedCurrent_: _dafny.Seq
                            out2_: _dafny.Seq
                            out3_: bool
                            out4_: _dafny.Seq
                            out2_, out3_, out4_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_10_openedGenerated_ = out2_
                            d_11_openedInside_ = out3_
                            d_12_openedCurrent_ = out4_
                            generated = d_10_openedGenerated_
                            insideConstrainedOut = d_11_openedInside_
                            currentConstrainedOut = d_12_openedCurrent_
                            d_7_forcedFinal_ = True
                            d_4_steps_ = (d_4_steps_) + (1)
                        elif True:
                            d_13_nextFree_: _dafny.Seq
                            out5_: _dafny.Seq
                            out5_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_13_nextFree_ = out5_
                            d_4_steps_ = (d_4_steps_) + (1)
                            if (d_13_nextFree_) == (eosToken):
                                if ((((d_8_openCount_) == (d_9_closeCount_)) and (not(d_7_forcedFinal_))) and ((d_4_steps_) < (maxSteps))) and ((d_4_steps_) < (d_6_localLimit_)):
                                    d_14_eosOpenedGenerated_: _dafny.Seq
                                    d_15_eosOpenedInside_: bool
                                    d_16_eosOpenedCurrent_: _dafny.Seq
                                    out6_: _dafny.Seq
                                    out7_: bool
                                    out8_: _dafny.Seq
                                    out6_, out7_, out8_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                    d_14_eosOpenedGenerated_ = out6_
                                    d_15_eosOpenedInside_ = out7_
                                    d_16_eosOpenedCurrent_ = out8_
                                    generated = d_14_eosOpenedGenerated_
                                    insideConstrainedOut = d_15_eosOpenedInside_
                                    currentConstrainedOut = d_16_eosOpenedCurrent_
                                    d_7_forcedFinal_ = True
                                    d_4_steps_ = (d_4_steps_) + (1)
                                elif True:
                                    raise _dafny.Break("0")
                            elif (d_13_nextFree_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_13_nextFree_]))
                                d_17_observedGenerated_: _dafny.Seq
                                d_18_observedInside_: bool
                                d_19_observedCurrent_: _dafny.Seq
                                out9_: _dafny.Seq
                                out10_: bool
                                out11_: _dafny.Seq
                                out9_, out10_, out11_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_17_observedGenerated_ = out9_
                                d_18_observedInside_ = out10_
                                d_19_observedCurrent_ = out11_
                                generated = d_17_observedGenerated_
                                insideConstrainedOut = d_18_observedInside_
                                currentConstrainedOut = d_19_observedCurrent_
                            elif (d_13_nextFree_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))):
                                pass
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_13_nextFree_]))
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_20_closedGenerated_: _dafny.Seq
                        d_21_closedInside_: bool
                        d_22_closedCurrent_: _dafny.Seq
                        out12_: _dafny.Seq
                        out13_: bool
                        out14_: _dafny.Seq
                        out12_, out13_, out14_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_20_closedGenerated_ = out12_
                        d_21_closedInside_ = out13_
                        d_22_closedCurrent_ = out14_
                        generated = d_20_closedGenerated_
                        insideConstrainedOut = d_21_closedInside_
                        currentConstrainedOut = d_22_closedCurrent_
                        d_4_steps_ = (d_4_steps_) + (1)
                    elif True:
                        d_23_constrainedPrompt_: _dafny.Seq
                        d_23_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_24_combinedGroups_: _dafny.Seq
                        d_24_combinedGroups_ = (d_2_mathGroups_) + (validTokenGroups)
                        d_25_nextConstrained_: _dafny.Seq
                        out15_: _dafny.Seq
                        out15_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_23_constrainedPrompt_, currentConstrainedOut, d_24_combinedGroups_, _dafny.BigRational('5e0'), d_3_penaltyTokens_, _dafny.BigRational('5e0'), 64, eosToken)
                        d_25_nextConstrained_ = out15_
                        d_4_steps_ = (d_4_steps_) + (1)
                        if (d_25_nextConstrained_) == (eosToken):
                            if (((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_4_steps_) < (maxSteps))) and ((d_4_steps_) < (d_6_localLimit_)):
                                d_26_eosClosedGenerated_: _dafny.Seq
                                d_27_eosClosedInside_: bool
                                d_28_eosClosedCurrent_: _dafny.Seq
                                out16_: _dafny.Seq
                                out17_: bool
                                out18_: _dafny.Seq
                                out16_, out17_, out18_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_26_eosClosedGenerated_ = out16_
                                d_27_eosClosedInside_ = out17_
                                d_28_eosClosedCurrent_ = out18_
                                generated = d_26_eosClosedGenerated_
                                insideConstrainedOut = d_27_eosClosedInside_
                                currentConstrainedOut = d_28_eosClosedCurrent_
                                d_4_steps_ = (d_4_steps_) + (1)
                        elif (d_25_nextConstrained_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                            pass
                        elif (d_25_nextConstrained_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))):
                            if (((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_4_steps_) < (maxSteps))) and ((d_4_steps_) < (d_6_localLimit_)):
                                d_29_delimClosedGenerated_: _dafny.Seq
                                d_30_delimClosedInside_: bool
                                d_31_delimClosedCurrent_: _dafny.Seq
                                out19_: _dafny.Seq
                                out20_: bool
                                out21_: _dafny.Seq
                                out19_, out20_, out21_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_29_delimClosedGenerated_ = out19_
                                d_30_delimClosedInside_ = out20_
                                d_31_delimClosedCurrent_ = out21_
                                generated = d_29_delimClosedGenerated_
                                insideConstrainedOut = d_30_delimClosedInside_
                                currentConstrainedOut = d_31_delimClosedCurrent_
                                d_4_steps_ = (d_4_steps_) + (1)
                        elif True:
                            d_32_appendedGenerated_: _dafny.Seq
                            d_33_appendedInside_: bool
                            d_34_appendedCurrent_: _dafny.Seq
                            out22_: _dafny.Seq
                            out23_: bool
                            out24_: _dafny.Seq
                            out22_, out23_, out24_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_25_nextConstrained_)
                            d_32_appendedGenerated_ = out22_
                            d_33_appendedInside_ = out23_
                            d_34_appendedCurrent_ = out24_
                            generated = d_32_appendedGenerated_
                            insideConstrainedOut = d_33_appendedInside_
                            currentConstrainedOut = d_34_appendedCurrent_
                            if (((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_4_steps_) < (maxSteps))) and ((d_4_steps_) < (d_6_localLimit_)):
                                d_35_justClosedGenerated_: _dafny.Seq
                                d_36_justClosedInside_: bool
                                d_37_justClosedCurrent_: _dafny.Seq
                                out25_: _dafny.Seq
                                out26_: bool
                                out27_: _dafny.Seq
                                out25_, out26_, out27_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_35_justClosedGenerated_ = out25_
                                d_36_justClosedInside_ = out26_
                                d_37_justClosedCurrent_ = out27_
                                generated = d_35_justClosedGenerated_
                                insideConstrainedOut = d_36_justClosedInside_
                                currentConstrainedOut = d_37_justClosedCurrent_
                                d_4_steps_ = (d_4_steps_) + (1)
                    pass
            pass
        cost = d_4_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

