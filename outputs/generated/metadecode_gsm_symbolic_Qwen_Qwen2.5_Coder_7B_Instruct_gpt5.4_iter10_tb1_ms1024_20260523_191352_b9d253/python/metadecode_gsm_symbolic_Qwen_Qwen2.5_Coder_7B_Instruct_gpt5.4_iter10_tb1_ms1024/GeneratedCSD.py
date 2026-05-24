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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step. Put each arithmetic computation inside visible << and >> delimiters.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_eqToken_: _dafny.Seq
        d_2_eqToken_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "="))
        d_3_colonToken_: _dafny.Seq
        d_3_colonToken_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ":"))
        d_4_plusToken_: _dafny.Seq
        d_4_plusToken_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "+"))
        d_5_minusToken_: _dafny.Seq
        d_5_minusToken_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "-"))
        d_6_timesToken_: _dafny.Seq
        d_6_timesToken_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "*"))
        d_7_divToken_: _dafny.Seq
        d_7_divToken_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "/"))
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_8_openCount_: int
                        out0_: int
                        out0_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))
                        d_8_openCount_ = out0_
                        d_9_sinceEq_: int
                        out1_: int
                        out1_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, d_2_eqToken_)
                        d_9_sinceEq_ = out1_
                        d_10_sinceColon_: int
                        out2_: int
                        out2_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, d_3_colonToken_)
                        d_10_sinceColon_ = out2_
                        if (((d_8_openCount_) == (0)) and ((len(generated)) > (len(generatedPrefix)))) and (((d_9_sinceEq_) <= (2)) or ((d_10_sinceColon_) <= (2))):
                            d_11_openedGenerated_: _dafny.Seq
                            d_12_openedInside_: bool
                            d_13_openedCurrent_: _dafny.Seq
                            out3_: _dafny.Seq
                            out4_: bool
                            out5_: _dafny.Seq
                            out3_, out4_, out5_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_11_openedGenerated_ = out3_
                            d_12_openedInside_ = out4_
                            d_13_openedCurrent_ = out5_
                            generated = d_11_openedGenerated_
                            insideConstrainedOut = d_12_openedInside_
                            currentConstrainedOut = d_13_openedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_14_next_: _dafny.Seq
                            out6_: _dafny.Seq
                            out6_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_14_next_ = out6_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_14_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_14_next_]))
                                if (d_14_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    d_15_enteredGenerated_: _dafny.Seq
                                    d_16_enteredInside_: bool
                                    d_17_enteredCurrent_: _dafny.Seq
                                    out7_: _dafny.Seq
                                    out8_: bool
                                    out9_: _dafny.Seq
                                    out7_, out8_, out9_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                    d_15_enteredGenerated_ = out7_
                                    d_16_enteredInside_ = out8_
                                    d_17_enteredCurrent_ = out9_
                                    generated = d_15_enteredGenerated_
                                    insideConstrainedOut = d_16_enteredInside_
                                    currentConstrainedOut = d_17_enteredCurrent_
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_18_closedGenerated_: _dafny.Seq
                        d_19_closedInside_: bool
                        d_20_closedCurrent_: _dafny.Seq
                        out10_: _dafny.Seq
                        out11_: bool
                        out12_: _dafny.Seq
                        out10_, out11_, out12_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_18_closedGenerated_ = out10_
                        d_19_closedInside_ = out11_
                        d_20_closedCurrent_ = out12_
                        generated = d_18_closedGenerated_
                        insideConstrainedOut = d_19_closedInside_
                        currentConstrainedOut = d_20_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_21_constrainedPrompt_: _dafny.Seq
                        d_21_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_22_validCount_: int
                        out13_: int
                        out13_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                        d_22_validCount_ = out13_
                        if ((d_22_validCount_) <= (12)) or ((stepTokenBudget) <= (1)):
                            d_23_penaltyTokens_: _dafny.Seq
                            d_23_penaltyTokens_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))])
                            if (len(currentConstrainedOut)) >= (1):
                                d_23_penaltyTokens_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), d_2_eqToken_])
                            d_24_next_: _dafny.Seq
                            out14_: _dafny.Seq
                            out14_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_21_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_23_penaltyTokens_, _dafny.BigRational('3e0'), 12, eosToken)
                            d_24_next_ = out14_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_24_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_25_appendedGenerated_: _dafny.Seq
                                d_26_appendedInside_: bool
                                d_27_appendedCurrent_: _dafny.Seq
                                out15_: _dafny.Seq
                                out16_: bool
                                out17_: _dafny.Seq
                                out15_, out16_, out17_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_24_next_)
                                d_25_appendedGenerated_ = out15_
                                d_26_appendedInside_ = out16_
                                d_27_appendedCurrent_ = out17_
                                generated = d_25_appendedGenerated_
                                insideConstrainedOut = d_26_appendedInside_
                                currentConstrainedOut = d_27_appendedCurrent_
                        elif True:
                            d_28_remaining_: int
                            d_28_remaining_ = (maxSteps) - (d_1_steps_)
                            d_29_symbolBudget_: int
                            if ((stepTokenBudget) == (0)) or ((stepTokenBudget) > (d_28_remaining_)):
                                d_29_symbolBudget_ = d_28_remaining_
                            elif True:
                                d_29_symbolBudget_ = stepTokenBudget
                            d_30_symbolGenerated_: _dafny.Seq
                            d_31_symbolOut_: _dafny.Seq
                            d_32_hitEos_: bool
                            d_33_stepsUsed_: int
                            out18_: _dafny.Seq
                            out19_: _dafny.Seq
                            out20_: bool
                            out21_: int
                            out18_, out19_, out20_, out21_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_21_constrainedPrompt_, generated, currentConstrainedOut, d_29_symbolBudget_, eosToken)
                            d_30_symbolGenerated_ = out18_
                            d_31_symbolOut_ = out19_
                            d_32_hitEos_ = out20_
                            d_33_stepsUsed_ = out21_
                            generated = d_30_symbolGenerated_
                            insideConstrainedOut = True
                            currentConstrainedOut = d_31_symbolOut_
                            d_1_steps_ = (d_1_steps_) + (d_33_stepsUsed_)
                            if d_32_hitEos_:
                                raise _dafny.Break("0")
                            elif ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                d_34_closedGenerated2_: _dafny.Seq
                                d_35_closedInside2_: bool
                                d_36_closedCurrent2_: _dafny.Seq
                                out22_: _dafny.Seq
                                out23_: bool
                                out24_: _dafny.Seq
                                out22_, out23_, out24_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_34_closedGenerated2_ = out22_
                                d_35_closedInside2_ = out23_
                                d_36_closedCurrent2_ = out24_
                                generated = d_34_closedGenerated2_
                                insideConstrainedOut = d_35_closedInside2_
                                currentConstrainedOut = d_36_closedCurrent2_
                                d_1_steps_ = (d_1_steps_) + (1)
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

