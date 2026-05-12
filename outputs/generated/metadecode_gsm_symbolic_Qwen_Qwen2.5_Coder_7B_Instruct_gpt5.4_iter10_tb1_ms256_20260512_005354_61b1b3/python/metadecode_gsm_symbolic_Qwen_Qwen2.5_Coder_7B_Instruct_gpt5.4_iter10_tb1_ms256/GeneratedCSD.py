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
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_arithmeticArmed_: bool
        d_2_arithmeticArmed_ = False
        d_3_rollbackLimit_: int
        d_3_rollbackLimit_ = 48
        d_4_narrowThreshold_: int
        d_4_narrowThreshold_ = 8
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if d_2_arithmeticArmed_:
                            d_5_openedGenerated_: _dafny.Seq
                            d_6_openedInside_: bool
                            d_7_openedCurrent_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_5_openedGenerated_ = out0_
                            d_6_openedInside_ = out1_
                            d_7_openedCurrent_ = out2_
                            generated = d_5_openedGenerated_
                            insideConstrainedOut = d_6_openedInside_
                            currentConstrainedOut = d_7_openedCurrent_
                            d_2_arithmeticArmed_ = False
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_8_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_8_next_ = out3_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_8_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_8_next_]))
                                if (d_8_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    d_9_enteredGenerated_: _dafny.Seq
                                    d_10_enteredInside_: bool
                                    d_11_enteredCurrent_: _dafny.Seq
                                    out4_: _dafny.Seq
                                    out5_: bool
                                    out6_: _dafny.Seq
                                    out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                    d_9_enteredGenerated_ = out4_
                                    d_10_enteredInside_ = out5_
                                    d_11_enteredCurrent_ = out6_
                                    generated = d_9_enteredGenerated_
                                    insideConstrainedOut = d_10_enteredInside_
                                    currentConstrainedOut = d_11_enteredCurrent_
                                    d_2_arithmeticArmed_ = False
                                elif True:
                                    if (((((d_8_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")))) or ((d_8_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "+"))))) or ((d_8_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "-"))))) or ((d_8_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "*"))))) or ((d_8_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "/")))):
                                        d_2_arithmeticArmed_ = True
                                    elif (((d_8_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ".")))) or ((d_8_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ","))))) or ((d_8_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")))):
                                        d_2_arithmeticArmed_ = False
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_12_closedGenerated_: _dafny.Seq
                        d_13_closedInside_: bool
                        d_14_closedCurrent_: _dafny.Seq
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: _dafny.Seq
                        out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_12_closedGenerated_ = out7_
                        d_13_closedInside_ = out8_
                        d_14_closedCurrent_ = out9_
                        generated = d_12_closedGenerated_
                        insideConstrainedOut = d_13_closedInside_
                        currentConstrainedOut = d_14_closedCurrent_
                        d_2_arithmeticArmed_ = False
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif (len(currentConstrainedOut)) >= (d_3_rollbackLimit_):
                        d_15_rolledGenerated_: _dafny.Seq
                        d_16_rolledCurrent_: _dafny.Seq
                        out10_: _dafny.Seq
                        out11_: _dafny.Seq
                        out10_, out11_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                        d_15_rolledGenerated_ = out10_
                        d_16_rolledCurrent_ = out11_
                        generated = d_15_rolledGenerated_
                        insideConstrainedOut = True
                        currentConstrainedOut = d_16_rolledCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_17_stablePrefix_: _dafny.Seq
                        d_17_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_18_constrainedPrompt_: _dafny.Seq
                        d_18_constrainedPrompt_ = (prompt) + (d_17_stablePrefix_)
                        d_19_validCount_: int
                        out12_: int
                        out12_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                        d_19_validCount_ = out12_
                        d_20_deadEndNear_: bool
                        out13_: bool
                        out13_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                        d_20_deadEndNear_ = out13_
                        if (((stepTokenBudget) > (1)) and ((d_19_validCount_) > (d_4_narrowThreshold_))) and (not(d_20_deadEndNear_)):
                            d_21_remaining_: int
                            d_21_remaining_ = (maxSteps) - (d_1_steps_)
                            d_22_symbolBudget_: int
                            if (stepTokenBudget) > (d_21_remaining_):
                                d_22_symbolBudget_ = d_21_remaining_
                            elif True:
                                d_22_symbolBudget_ = stepTokenBudget
                            d_23_symbolGenerated_: _dafny.Seq
                            d_24_symbolOut_: _dafny.Seq
                            d_25_hitEos_: bool
                            d_26_stepsUsed_: int
                            out14_: _dafny.Seq
                            out15_: _dafny.Seq
                            out16_: bool
                            out17_: int
                            out14_, out15_, out16_, out17_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_18_constrainedPrompt_, generated, currentConstrainedOut, d_22_symbolBudget_, eosToken)
                            d_23_symbolGenerated_ = out14_
                            d_24_symbolOut_ = out15_
                            d_25_hitEos_ = out16_
                            d_26_stepsUsed_ = out17_
                            generated = d_23_symbolGenerated_
                            currentConstrainedOut = d_24_symbolOut_
                            d_1_steps_ = (d_1_steps_) + (d_26_stepsUsed_)
                            if d_25_hitEos_:
                                raise _dafny.Break("0")
                        elif True:
                            d_27_nextIn_: _dafny.Seq
                            d_27_nextIn_ = eosToken
                            d_28_sinceEq_: int
                            d_28_sinceEq_ = 0
                            d_29_eqCount_: int
                            d_29_eqCount_ = 0
                            if (len(currentConstrainedOut)) > (0):
                                out18_: int
                                out18_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")))
                                d_28_sinceEq_ = out18_
                            out19_: int
                            out19_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")))
                            d_29_eqCount_ = out19_
                            if (len(currentConstrainedOut)) == (0):
                                out20_: _dafny.Seq
                                out20_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_18_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                                d_27_nextIn_ = out20_
                            elif (len(currentConstrainedOut)) < (2):
                                out21_: _dafny.Seq
                                out21_ = (d_0_helpers_).SafePenalizedConstrainedStep(lm, parser, d_18_constrainedPrompt_, currentConstrainedOut, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]), _dafny.BigRational('6e0'), eosToken)
                                d_27_nextIn_ = out21_
                            elif (d_28_sinceEq_) < (len(currentConstrainedOut)):
                                out22_: _dafny.Seq
                                out22_ = (d_0_helpers_).SafeBoostedConstrainedStep(lm, parser, d_18_constrainedPrompt_, currentConstrainedOut, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "+")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "-")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "*")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "/")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "0")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "1")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "2")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "3")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "4")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "5")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "6")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "7")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "8")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "9"))]), _dafny.BigRational('2e0'), eosToken)
                                d_27_nextIn_ = out22_
                            elif (d_29_eqCount_) > (0):
                                out23_: _dafny.Seq
                                out23_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_18_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('2e0'), eosToken)
                                d_27_nextIn_ = out23_
                            elif True:
                                d_30_gatedTok_: _dafny.Seq
                                d_31_wasConstrained_: bool
                                out24_: _dafny.Seq
                                out25_: bool
                                out24_, out25_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, d_18_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_30_gatedTok_ = out24_
                                d_31_wasConstrained_ = out25_
                                d_27_nextIn_ = d_30_gatedTok_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_27_nextIn_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_32_appendedGenerated_: _dafny.Seq
                                d_33_appendedInside_: bool
                                d_34_appendedCurrent_: _dafny.Seq
                                out26_: _dafny.Seq
                                out27_: bool
                                out28_: _dafny.Seq
                                out26_, out27_, out28_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_27_nextIn_)
                                d_32_appendedGenerated_ = out26_
                                d_33_appendedInside_ = out27_
                                d_34_appendedCurrent_ = out28_
                                generated = d_32_appendedGenerated_
                                insideConstrainedOut = d_33_appendedInside_
                                currentConstrainedOut = d_34_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

