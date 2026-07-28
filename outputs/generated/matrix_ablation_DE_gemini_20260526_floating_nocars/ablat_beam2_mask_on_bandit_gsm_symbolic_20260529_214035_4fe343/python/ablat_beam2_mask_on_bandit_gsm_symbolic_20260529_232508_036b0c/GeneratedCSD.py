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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step. Wrap intermediate symbolic expressions and the final answer in visible << >> delimiters. Inside each << >> span, use only compact symbolic arithmetic or a numeric answer, with no words or units."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_mathGroups_: _dafny.Seq
        d_2_mathGroups_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "0")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "1")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "2")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "3")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "4")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "5")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "6")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "7")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "8")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "9"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "+")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "-")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "*")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "/")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "//")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "(")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ")")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ".")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "%")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "="))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "int")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "ceil")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "floor")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "min")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "max"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "n")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "x")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "y")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "k")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "r")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "w")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "t")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "d")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "total"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "n1")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "n2")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "n3")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "n4")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "w1")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "w2")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "w3")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "w4")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "k_2")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "k_3"))])])
        d_3_penaltyTokens_: _dafny.Seq
        d_3_penaltyTokens_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\n\n")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "  "))])
        d_4_steps_: int
        d_4_steps_ = 0
        d_5_forceAfter_: int
        d_5_forceAfter_ = 64
        d_6_forcedFinal_: bool
        d_6_forcedFinal_ = False
        with _dafny.label("0"):
            while (d_4_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if (d_4_steps_) >= (d_5_forceAfter_):
                            d_7_openedGenerated_: _dafny.Seq
                            d_8_openedInside_: bool
                            d_9_openedCurrent_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_7_openedGenerated_ = out0_
                            d_8_openedInside_ = out1_
                            d_9_openedCurrent_ = out2_
                            generated = d_7_openedGenerated_
                            insideConstrainedOut = d_8_openedInside_
                            currentConstrainedOut = d_9_openedCurrent_
                            d_6_forcedFinal_ = True
                            d_4_steps_ = (d_4_steps_) + (1)
                        elif True:
                            d_10_nextFree_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_10_nextFree_ = out3_
                            d_4_steps_ = (d_4_steps_) + (1)
                            if (d_10_nextFree_) == (eosToken):
                                if (d_4_steps_) < (maxSteps):
                                    d_11_eosOpenedGenerated_: _dafny.Seq
                                    d_12_eosOpenedInside_: bool
                                    d_13_eosOpenedCurrent_: _dafny.Seq
                                    out4_: _dafny.Seq
                                    out5_: bool
                                    out6_: _dafny.Seq
                                    out4_, out5_, out6_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                    d_11_eosOpenedGenerated_ = out4_
                                    d_12_eosOpenedInside_ = out5_
                                    d_13_eosOpenedCurrent_ = out6_
                                    generated = d_11_eosOpenedGenerated_
                                    insideConstrainedOut = d_12_eosOpenedInside_
                                    currentConstrainedOut = d_13_eosOpenedCurrent_
                                    d_6_forcedFinal_ = True
                                    d_4_steps_ = (d_4_steps_) + (1)
                                elif True:
                                    raise _dafny.Break("0")
                            elif (d_10_nextFree_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_10_nextFree_]))
                                d_14_observedGenerated_: _dafny.Seq
                                d_15_observedInside_: bool
                                d_16_observedCurrent_: _dafny.Seq
                                out7_: _dafny.Seq
                                out8_: bool
                                out9_: _dafny.Seq
                                out7_, out8_, out9_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_14_observedGenerated_ = out7_
                                d_15_observedInside_ = out8_
                                d_16_observedCurrent_ = out9_
                                generated = d_14_observedGenerated_
                                insideConstrainedOut = d_15_observedInside_
                                currentConstrainedOut = d_16_observedCurrent_
                                d_6_forcedFinal_ = False
                            elif (d_10_nextFree_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))):
                                pass
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_10_nextFree_]))
                    elif True:
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            d_17_closedGenerated_: _dafny.Seq
                            d_18_closedInside_: bool
                            d_19_closedCurrent_: _dafny.Seq
                            out10_: _dafny.Seq
                            out11_: bool
                            out12_: _dafny.Seq
                            out10_, out11_, out12_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_17_closedGenerated_ = out10_
                            d_18_closedInside_ = out11_
                            d_19_closedCurrent_ = out12_
                            generated = d_17_closedGenerated_
                            insideConstrainedOut = d_18_closedInside_
                            currentConstrainedOut = d_19_closedCurrent_
                            d_4_steps_ = (d_4_steps_) + (1)
                            if d_6_forcedFinal_:
                                raise _dafny.Break("0")
                        elif True:
                            d_20_constrainedPrompt_: _dafny.Seq
                            d_20_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_21_combinedGroups_: _dafny.Seq
                            d_21_combinedGroups_ = (d_2_mathGroups_) + (validTokenGroups)
                            d_22_nextConstrained_: _dafny.Seq
                            out13_: _dafny.Seq
                            out13_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_20_constrainedPrompt_, currentConstrainedOut, d_21_combinedGroups_, _dafny.BigRational('4e0'), d_3_penaltyTokens_, _dafny.BigRational('4e0'), 12, eosToken)
                            d_22_nextConstrained_ = out13_
                            d_4_steps_ = (d_4_steps_) + (1)
                            if (d_22_nextConstrained_) == (eosToken):
                                raise _dafny.Break("0")
                            elif (d_22_nextConstrained_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                pass
                            elif (d_22_nextConstrained_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))):
                                pass
                            elif True:
                                d_23_valid_: bool
                                out14_: bool
                                out14_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_22_nextConstrained_)
                                d_23_valid_ = out14_
                                if d_23_valid_:
                                    d_24_appendedGenerated_: _dafny.Seq
                                    d_25_appendedInside_: bool
                                    d_26_appendedCurrent_: _dafny.Seq
                                    out15_: _dafny.Seq
                                    out16_: bool
                                    out17_: _dafny.Seq
                                    out15_, out16_, out17_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_22_nextConstrained_)
                                    d_24_appendedGenerated_ = out15_
                                    d_25_appendedInside_ = out16_
                                    d_26_appendedCurrent_ = out17_
                                    generated = d_24_appendedGenerated_
                                    insideConstrainedOut = d_25_appendedInside_
                                    currentConstrainedOut = d_26_appendedCurrent_
                    pass
            pass
        cost = d_4_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

