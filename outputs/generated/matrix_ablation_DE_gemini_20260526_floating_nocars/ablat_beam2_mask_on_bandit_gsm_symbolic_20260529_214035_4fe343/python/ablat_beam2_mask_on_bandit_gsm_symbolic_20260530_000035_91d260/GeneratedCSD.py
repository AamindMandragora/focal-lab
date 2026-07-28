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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step in concise prose. Do not open any << >> span until the final answer. At the end, give exactly one final symbolic arithmetic expression inside visible << >> delimiters. Inside the delimiters write only valid compact arithmetic, for example a*b+c, t-d, (n1*w1+n2*w2)//total, or int((a/b)*100)."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_mathGroups_: _dafny.Seq
        d_2_mathGroups_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "0")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "1")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "2")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "3")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "4")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "5")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "6")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "7")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "8")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "9"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "+")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "-")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "*")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "/")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "//")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "(")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ")")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ".")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "%"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "int")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "ceil")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "floor"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "n")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "n1")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "n2")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "n3")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "n4")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "x")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "y")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "t")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "t1")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "t2")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "d")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "w")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "w1")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "w2")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "w3")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "w4")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "total")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "target")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "sides")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "m"))])])
        d_3_penaltyTokens_: _dafny.Seq
        d_3_penaltyTokens_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\n")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\n\n")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " ")), eosToken])
        d_4_steps_: int
        d_4_steps_ = 0
        d_5_forceAfter_: int
        d_5_forceAfter_ = 120
        with _dafny.label("0"):
            while (d_4_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_6_openCount_: int
                        out0_: int
                        out0_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))
                        d_6_openCount_ = out0_
                        d_7_closeCount_: int
                        out1_: int
                        out1_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")))
                        d_7_closeCount_ = out1_
                        if (((d_6_openCount_) == (d_7_closeCount_)) and ((d_6_openCount_) == (0))) and ((d_4_steps_) >= (d_5_forceAfter_)):
                            d_8_openedGenerated_: _dafny.Seq
                            d_9_openedInside_: bool
                            d_10_openedCurrent_: _dafny.Seq
                            out2_: _dafny.Seq
                            out3_: bool
                            out4_: _dafny.Seq
                            out2_, out3_, out4_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_8_openedGenerated_ = out2_
                            d_9_openedInside_ = out3_
                            d_10_openedCurrent_ = out4_
                            generated = d_8_openedGenerated_
                            insideConstrainedOut = d_9_openedInside_
                            currentConstrainedOut = d_10_openedCurrent_
                            d_4_steps_ = (d_4_steps_) + (1)
                        elif True:
                            d_11_nextFree_: _dafny.Seq
                            out5_: _dafny.Seq
                            out5_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_11_nextFree_ = out5_
                            d_4_steps_ = (d_4_steps_) + (1)
                            if (d_11_nextFree_) == (eosToken):
                                if (((d_6_openCount_) == (d_7_closeCount_)) and ((d_6_openCount_) == (0))) and ((d_4_steps_) < (maxSteps)):
                                    d_12_eosOpenedGenerated_: _dafny.Seq
                                    d_13_eosOpenedInside_: bool
                                    d_14_eosOpenedCurrent_: _dafny.Seq
                                    out6_: _dafny.Seq
                                    out7_: bool
                                    out8_: _dafny.Seq
                                    out6_, out7_, out8_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                    d_12_eosOpenedGenerated_ = out6_
                                    d_13_eosOpenedInside_ = out7_
                                    d_14_eosOpenedCurrent_ = out8_
                                    generated = d_12_eosOpenedGenerated_
                                    insideConstrainedOut = d_13_eosOpenedInside_
                                    currentConstrainedOut = d_14_eosOpenedCurrent_
                                    d_4_steps_ = (d_4_steps_) + (1)
                                elif True:
                                    raise _dafny.Break("0")
                            elif (d_11_nextFree_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                pass
                            elif (d_11_nextFree_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))):
                                pass
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_11_nextFree_]))
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_15_closedGenerated_: _dafny.Seq
                        d_16_closedInside_: bool
                        d_17_closedCurrent_: _dafny.Seq
                        out9_: _dafny.Seq
                        out10_: bool
                        out11_: _dafny.Seq
                        out9_, out10_, out11_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_15_closedGenerated_ = out9_
                        d_16_closedInside_ = out10_
                        d_17_closedCurrent_ = out11_
                        generated = d_15_closedGenerated_
                        insideConstrainedOut = d_16_closedInside_
                        currentConstrainedOut = d_17_closedCurrent_
                        d_4_steps_ = (d_4_steps_) + (1)
                        raise _dafny.Break("0")
                    elif True:
                        d_18_constrainedPrompt_: _dafny.Seq
                        d_18_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_19_combinedGroups_: _dafny.Seq
                        d_19_combinedGroups_ = (d_2_mathGroups_) + (validTokenGroups)
                        d_20_nextConstrained_: _dafny.Seq
                        out12_: _dafny.Seq
                        out12_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_18_constrainedPrompt_, currentConstrainedOut, d_19_combinedGroups_, _dafny.BigRational('5e0'), d_3_penaltyTokens_, _dafny.BigRational('8e0'), 64, eosToken)
                        d_20_nextConstrained_ = out12_
                        d_4_steps_ = (d_4_steps_) + (1)
                        if (d_20_nextConstrained_) == (eosToken):
                            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_4_steps_) < (maxSteps)):
                                d_21_eosClosedGenerated_: _dafny.Seq
                                d_22_eosClosedInside_: bool
                                d_23_eosClosedCurrent_: _dafny.Seq
                                out13_: _dafny.Seq
                                out14_: bool
                                out15_: _dafny.Seq
                                out13_, out14_, out15_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_21_eosClosedGenerated_ = out13_
                                d_22_eosClosedInside_ = out14_
                                d_23_eosClosedCurrent_ = out15_
                                generated = d_21_eosClosedGenerated_
                                insideConstrainedOut = d_22_eosClosedInside_
                                currentConstrainedOut = d_23_eosClosedCurrent_
                                d_4_steps_ = (d_4_steps_) + (1)
                                raise _dafny.Break("0")
                        elif (d_20_nextConstrained_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                            pass
                        elif (d_20_nextConstrained_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))):
                            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_4_steps_) < (maxSteps)):
                                d_24_delimClosedGenerated_: _dafny.Seq
                                d_25_delimClosedInside_: bool
                                d_26_delimClosedCurrent_: _dafny.Seq
                                out16_: _dafny.Seq
                                out17_: bool
                                out18_: _dafny.Seq
                                out16_, out17_, out18_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_24_delimClosedGenerated_ = out16_
                                d_25_delimClosedInside_ = out17_
                                d_26_delimClosedCurrent_ = out18_
                                generated = d_24_delimClosedGenerated_
                                insideConstrainedOut = d_25_delimClosedInside_
                                currentConstrainedOut = d_26_delimClosedCurrent_
                                d_4_steps_ = (d_4_steps_) + (1)
                                raise _dafny.Break("0")
                        elif True:
                            d_27_appendedGenerated_: _dafny.Seq
                            d_28_appendedInside_: bool
                            d_29_appendedCurrent_: _dafny.Seq
                            out19_: _dafny.Seq
                            out20_: bool
                            out21_: _dafny.Seq
                            out19_, out20_, out21_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_20_nextConstrained_)
                            d_27_appendedGenerated_ = out19_
                            d_28_appendedInside_ = out20_
                            d_29_appendedCurrent_ = out21_
                            generated = d_27_appendedGenerated_
                            insideConstrainedOut = d_28_appendedInside_
                            currentConstrainedOut = d_29_appendedCurrent_
                    pass
            pass
        cost = d_4_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

