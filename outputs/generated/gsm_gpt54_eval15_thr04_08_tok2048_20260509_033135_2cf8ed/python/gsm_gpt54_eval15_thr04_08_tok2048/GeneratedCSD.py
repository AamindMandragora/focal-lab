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
        d_2_cueArmed_: bool
        d_2_cueArmed_ = False
        d_3_recentAfterAnswer_: _dafny.Seq
        out0_: _dafny.Seq
        out0_ = VerifiedDecoderAgent.CSDHelpers.ExtractAfterKeyword(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "answer")))
        d_3_recentAfterAnswer_ = out0_
        d_4_flatGroups_: _dafny.Seq
        out1_: _dafny.Seq
        out1_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
        d_4_flatGroups_ = out1_
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if d_2_cueArmed_:
                            d_5_openedGenerated_: _dafny.Seq
                            d_6_openedInside_: bool
                            d_7_openedCurrent_: _dafny.Seq
                            out2_: _dafny.Seq
                            out3_: bool
                            out4_: _dafny.Seq
                            out2_, out3_, out4_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_5_openedGenerated_ = out2_
                            d_6_openedInside_ = out3_
                            d_7_openedCurrent_ = out4_
                            generated = d_5_openedGenerated_
                            insideConstrainedOut = d_6_openedInside_
                            currentConstrainedOut = d_7_openedCurrent_
                            d_2_cueArmed_ = False
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_8_remainingChunk_: int
                            d_8_remainingChunk_ = (maxSteps) - (d_1_steps_)
                            d_9_chunkBudget_: int
                            if (d_8_remainingChunk_) <= (2):
                                d_9_chunkBudget_ = d_8_remainingChunk_
                            elif True:
                                d_9_chunkBudget_ = 2
                            d_10_chunkedGenerated_: _dafny.Seq
                            d_11_stoppedOnOpenSpan_: bool
                            d_12_stoppedOnEos_: bool
                            d_13_stepsUsed_: int
                            out5_: _dafny.Seq
                            out6_: bool
                            out7_: bool
                            out8_: int
                            out5_, out6_, out7_, out8_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_9_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_10_chunkedGenerated_ = out5_
                            d_11_stoppedOnOpenSpan_ = out6_
                            d_12_stoppedOnEos_ = out7_
                            d_13_stepsUsed_ = out8_
                            generated = d_10_chunkedGenerated_
                            d_1_steps_ = (d_1_steps_) + (d_13_stepsUsed_)
                            out9_: _dafny.Seq
                            out9_ = VerifiedDecoderAgent.CSDHelpers.ExtractAfterKeyword(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "answer")))
                            d_3_recentAfterAnswer_ = out9_
                            if d_12_stoppedOnEos_:
                                raise _dafny.Break("0")
                            elif d_11_stoppedOnOpenSpan_:
                                d_14_enteredGenerated_: _dafny.Seq
                                d_15_enteredInside_: bool
                                d_16_enteredCurrent_: _dafny.Seq
                                out10_: _dafny.Seq
                                out11_: bool
                                out12_: _dafny.Seq
                                out10_, out11_, out12_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_14_enteredGenerated_ = out10_
                                d_15_enteredInside_ = out11_
                                d_16_enteredCurrent_ = out12_
                                generated = d_14_enteredGenerated_
                                insideConstrainedOut = d_15_enteredInside_
                                currentConstrainedOut = d_16_enteredCurrent_
                                d_2_cueArmed_ = False
                            elif True:
                                d_17_prevTok_: _dafny.Seq
                                d_18_foundPrev_: bool
                                out13_: _dafny.Seq
                                out14_: bool
                                out13_, out14_ = (d_0_helpers_).LastTokenBefore(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")))
                                d_17_prevTok_ = out13_
                                d_18_foundPrev_ = out14_
                                d_2_cueArmed_ = False
                                if (d_18_foundPrev_) and ((((d_17_prevTok_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")))) or ((d_17_prevTok_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ":"))))) or ((d_17_prevTok_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "is"))))):
                                    d_2_cueArmed_ = True
                                elif (len(d_3_recentAfterAnswer_)) > (0):
                                    d_2_cueArmed_ = True
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_19_closedGenerated_: _dafny.Seq
                        d_20_closedInside_: bool
                        d_21_closedCurrent_: _dafny.Seq
                        out15_: _dafny.Seq
                        out16_: bool
                        out17_: _dafny.Seq
                        out15_, out16_, out17_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_19_closedGenerated_ = out15_
                        d_20_closedInside_ = out16_
                        d_21_closedCurrent_ = out17_
                        generated = d_19_closedGenerated_
                        insideConstrainedOut = d_20_closedInside_
                        currentConstrainedOut = d_21_closedCurrent_
                        out18_: _dafny.Seq
                        out18_ = VerifiedDecoderAgent.CSDHelpers.ExtractAfterKeyword(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "answer")))
                        d_3_recentAfterAnswer_ = out18_
                        d_2_cueArmed_ = False
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_22_shouldRollback_: bool
                        d_22_shouldRollback_ = False
                        if (len(currentConstrainedOut)) >= (24):
                            d_23_deadEnd_: bool
                            out19_: bool
                            out19_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                            d_23_deadEnd_ = out19_
                            if d_23_deadEnd_:
                                d_22_shouldRollback_ = True
                        if d_22_shouldRollback_:
                            d_24_rolledGenerated_: _dafny.Seq
                            d_25_rolledCurrent_: _dafny.Seq
                            out20_: _dafny.Seq
                            out21_: _dafny.Seq
                            out20_, out21_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                            d_24_rolledGenerated_ = out20_
                            d_25_rolledCurrent_ = out21_
                            generated = d_24_rolledGenerated_
                            insideConstrainedOut = True
                            currentConstrainedOut = d_25_rolledCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_26_stablePrefix_: _dafny.Seq
                            d_26_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_27_constrainedPrompt_: _dafny.Seq
                            d_27_constrainedPrompt_ = (prompt) + (d_26_stablePrefix_)
                            d_28_validCount_: int
                            out22_: int
                            out22_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                            d_28_validCount_ = out22_
                            d_29_remaining_: int
                            d_29_remaining_ = (maxSteps) - (d_1_steps_)
                            if (((d_29_remaining_) >= (2)) and ((stepTokenBudget) > (1))) and ((d_28_validCount_) > (8)):
                                d_30_symbolBudget_: int
                                if (stepTokenBudget) > (d_29_remaining_):
                                    d_30_symbolBudget_ = d_29_remaining_
                                elif True:
                                    d_30_symbolBudget_ = stepTokenBudget
                                d_31_symbolGenerated_: _dafny.Seq
                                d_32_symbolOut_: _dafny.Seq
                                d_33_hitEos_: bool
                                d_34_stepsUsed_: int
                                out23_: _dafny.Seq
                                out24_: _dafny.Seq
                                out25_: bool
                                out26_: int
                                out23_, out24_, out25_, out26_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_27_constrainedPrompt_, generated, currentConstrainedOut, d_30_symbolBudget_, eosToken)
                                d_31_symbolGenerated_ = out23_
                                d_32_symbolOut_ = out24_
                                d_33_hitEos_ = out25_
                                d_34_stepsUsed_ = out26_
                                generated = d_31_symbolGenerated_
                                currentConstrainedOut = d_32_symbolOut_
                                d_1_steps_ = (d_1_steps_) + (d_34_stepsUsed_)
                                if d_33_hitEos_:
                                    raise _dafny.Break("0")
                            elif True:
                                d_35_preferred_: _dafny.Seq
                                d_35_preferred_ = d_4_flatGroups_
                                if (len(d_3_recentAfterAnswer_)) > (0):
                                    out27_: _dafny.Seq
                                    out27_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_4_flatGroups_, d_3_recentAfterAnswer_)
                                    d_35_preferred_ = out27_
                                d_36_penalized_: _dafny.Seq
                                out28_: _dafny.Seq
                                out28_ = VerifiedDecoderAgent.CSDHelpers.SubtractTokenSets(generated, d_35_preferred_)
                                d_36_penalized_ = out28_
                                d_37_next_: _dafny.Seq
                                d_37_next_ = eosToken
                                if (d_28_validCount_) <= (3):
                                    out29_: _dafny.Seq
                                    out29_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_27_constrainedPrompt_, currentConstrainedOut, eosToken)
                                    d_37_next_ = out29_
                                elif (len(d_35_preferred_)) > (0):
                                    out30_: _dafny.Seq
                                    out30_ = (d_0_helpers_).SafeBoostedConstrainedStep(lm, parser, d_27_constrainedPrompt_, currentConstrainedOut, d_35_preferred_, _dafny.BigRational('4e0'), eosToken)
                                    d_37_next_ = out30_
                                elif (len(d_36_penalized_)) > (0):
                                    out31_: _dafny.Seq
                                    out31_ = (d_0_helpers_).SafePenalizedConstrainedStep(lm, parser, d_27_constrainedPrompt_, currentConstrainedOut, d_36_penalized_, _dafny.BigRational('2e0'), eosToken)
                                    d_37_next_ = out31_
                                elif (d_28_validCount_) <= (12):
                                    out32_: _dafny.Seq
                                    out32_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_27_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                                    d_37_next_ = out32_
                                elif True:
                                    d_38_gatedNext_: _dafny.Seq
                                    d_39_wasConstrained_: bool
                                    out33_: _dafny.Seq
                                    out34_: bool
                                    out33_, out34_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, d_27_constrainedPrompt_, currentConstrainedOut, eosToken)
                                    d_38_gatedNext_ = out33_
                                    d_39_wasConstrained_ = out34_
                                    d_37_next_ = d_38_gatedNext_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_37_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_40_appendedGenerated_: _dafny.Seq
                                    d_41_appendedInside_: bool
                                    d_42_appendedCurrent_: _dafny.Seq
                                    out35_: _dafny.Seq
                                    out36_: bool
                                    out37_: _dafny.Seq
                                    out35_, out36_, out37_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_37_next_)
                                    d_40_appendedGenerated_ = out35_
                                    d_41_appendedInside_ = out36_
                                    d_42_appendedCurrent_ = out37_
                                    generated = d_40_appendedGenerated_
                                    insideConstrainedOut = d_41_appendedInside_
                                    currentConstrainedOut = d_42_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

