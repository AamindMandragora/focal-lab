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
        d_3_spanDone_: bool
        d_3_spanDone_ = False
        d_4_flatGroups_: _dafny.Seq
        out0_: _dafny.Seq
        out0_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
        d_4_flatGroups_ = out0_
        d_5_recentAfterAnswer_: _dafny.Seq
        out1_: _dafny.Seq
        out1_ = VerifiedDecoderAgent.CSDHelpers.ExtractAfterKeyword(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "answer")))
        d_5_recentAfterAnswer_ = out1_
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if (d_2_cueArmed_) and (not(d_3_spanDone_)):
                            d_6_openedGenerated_: _dafny.Seq
                            d_7_openedInside_: bool
                            d_8_openedCurrent_: _dafny.Seq
                            out2_: _dafny.Seq
                            out3_: bool
                            out4_: _dafny.Seq
                            out2_, out3_, out4_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_6_openedGenerated_ = out2_
                            d_7_openedInside_ = out3_
                            d_8_openedCurrent_ = out4_
                            generated = d_6_openedGenerated_
                            insideConstrainedOut = d_7_openedInside_
                            currentConstrainedOut = d_8_openedCurrent_
                            d_2_cueArmed_ = False
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_9_next_: _dafny.Seq
                            out5_: _dafny.Seq
                            out5_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_9_next_ = out5_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_9_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_9_next_]))
                                out6_: _dafny.Seq
                                out6_ = VerifiedDecoderAgent.CSDHelpers.ExtractAfterKeyword(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "answer")))
                                d_5_recentAfterAnswer_ = out6_
                                if ((d_9_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))) and (not(d_3_spanDone_)):
                                    d_10_enteredGenerated_: _dafny.Seq
                                    d_11_enteredInside_: bool
                                    d_12_enteredCurrent_: _dafny.Seq
                                    out7_: _dafny.Seq
                                    out8_: bool
                                    out9_: _dafny.Seq
                                    out7_, out8_, out9_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                    d_10_enteredGenerated_ = out7_
                                    d_11_enteredInside_ = out8_
                                    d_12_enteredCurrent_ = out9_
                                    generated = d_10_enteredGenerated_
                                    insideConstrainedOut = d_11_enteredInside_
                                    currentConstrainedOut = d_12_enteredCurrent_
                                    d_2_cueArmed_ = False
                                elif True:
                                    d_2_cueArmed_ = False
                                    if not(d_3_spanDone_):
                                        if ((d_9_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")))) or ((d_9_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ":")))):
                                            d_2_cueArmed_ = True
                                        elif (d_9_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "is"))):
                                            d_2_cueArmed_ = True
                                        elif (len(d_5_recentAfterAnswer_)) > (0):
                                            d_2_cueArmed_ = True
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_13_closedGenerated_: _dafny.Seq
                        d_14_closedInside_: bool
                        d_15_closedCurrent_: _dafny.Seq
                        out10_: _dafny.Seq
                        out11_: bool
                        out12_: _dafny.Seq
                        out10_, out11_, out12_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_13_closedGenerated_ = out10_
                        d_14_closedInside_ = out11_
                        d_15_closedCurrent_ = out12_
                        generated = d_13_closedGenerated_
                        insideConstrainedOut = d_14_closedInside_
                        currentConstrainedOut = d_15_closedCurrent_
                        d_3_spanDone_ = True
                        d_2_cueArmed_ = False
                        out13_: _dafny.Seq
                        out13_ = VerifiedDecoderAgent.CSDHelpers.ExtractAfterKeyword(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "answer")))
                        d_5_recentAfterAnswer_ = out13_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_16_shouldRollback_: bool
                        d_16_shouldRollback_ = False
                        if (len(currentConstrainedOut)) >= (16):
                            d_17_deadEnd_: bool
                            out14_: bool
                            out14_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                            d_17_deadEnd_ = out14_
                            if d_17_deadEnd_:
                                d_16_shouldRollback_ = True
                        if d_16_shouldRollback_:
                            d_18_rolledGenerated_: _dafny.Seq
                            d_19_rolledCurrent_: _dafny.Seq
                            out15_: _dafny.Seq
                            out16_: _dafny.Seq
                            out15_, out16_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                            d_18_rolledGenerated_ = out15_
                            d_19_rolledCurrent_ = out16_
                            generated = d_18_rolledGenerated_
                            insideConstrainedOut = True
                            currentConstrainedOut = d_19_rolledCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_20_stablePrefix_: _dafny.Seq
                            d_20_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_21_constrainedPrompt_: _dafny.Seq
                            d_21_constrainedPrompt_ = (prompt) + (d_20_stablePrefix_)
                            d_22_validCount_: int
                            out17_: int
                            out17_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                            d_22_validCount_ = out17_
                            d_23_remaining_: int
                            d_23_remaining_ = (maxSteps) - (d_1_steps_)
                            if (((d_23_remaining_) >= (2)) and ((stepTokenBudget) > (1))) and ((d_22_validCount_) > (10)):
                                d_24_symbolBudget_: int
                                if (stepTokenBudget) > (d_23_remaining_):
                                    d_24_symbolBudget_ = d_23_remaining_
                                elif True:
                                    d_24_symbolBudget_ = stepTokenBudget
                                d_25_symbolGenerated_: _dafny.Seq
                                d_26_symbolOut_: _dafny.Seq
                                d_27_hitEos_: bool
                                d_28_stepsUsed_: int
                                out18_: _dafny.Seq
                                out19_: _dafny.Seq
                                out20_: bool
                                out21_: int
                                out18_, out19_, out20_, out21_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_21_constrainedPrompt_, generated, currentConstrainedOut, d_24_symbolBudget_, eosToken)
                                d_25_symbolGenerated_ = out18_
                                d_26_symbolOut_ = out19_
                                d_27_hitEos_ = out20_
                                d_28_stepsUsed_ = out21_
                                generated = d_25_symbolGenerated_
                                insideConstrainedOut = True
                                currentConstrainedOut = d_26_symbolOut_
                                d_1_steps_ = (d_1_steps_) + (d_28_stepsUsed_)
                                if d_27_hitEos_:
                                    if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                        d_29_closedGenerated2_: _dafny.Seq
                                        d_30_closedInside2_: bool
                                        d_31_closedCurrent2_: _dafny.Seq
                                        out22_: _dafny.Seq
                                        out23_: bool
                                        out24_: _dafny.Seq
                                        out22_, out23_, out24_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                        d_29_closedGenerated2_ = out22_
                                        d_30_closedInside2_ = out23_
                                        d_31_closedCurrent2_ = out24_
                                        generated = d_29_closedGenerated2_
                                        insideConstrainedOut = d_30_closedInside2_
                                        currentConstrainedOut = d_31_closedCurrent2_
                                        d_3_spanDone_ = True
                                        d_2_cueArmed_ = False
                                        out25_: _dafny.Seq
                                        out25_ = VerifiedDecoderAgent.CSDHelpers.ExtractAfterKeyword(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "answer")))
                                        d_5_recentAfterAnswer_ = out25_
                                        d_1_steps_ = (d_1_steps_) + (1)
                                    raise _dafny.Break("0")
                            elif True:
                                d_32_preferred_: _dafny.Seq
                                d_32_preferred_ = d_4_flatGroups_
                                if (len(d_5_recentAfterAnswer_)) > (0):
                                    d_33_narrowed_: _dafny.Seq
                                    out26_: _dafny.Seq
                                    out26_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_4_flatGroups_, d_5_recentAfterAnswer_)
                                    d_33_narrowed_ = out26_
                                    if (len(d_33_narrowed_)) > (0):
                                        d_32_preferred_ = d_33_narrowed_
                                d_34_next_: _dafny.Seq
                                d_34_next_ = eosToken
                                if (d_22_validCount_) <= (4):
                                    out27_: _dafny.Seq
                                    out27_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_21_constrainedPrompt_, currentConstrainedOut, eosToken)
                                    d_34_next_ = out27_
                                elif ((len(d_32_preferred_)) > (0)) and ((d_22_validCount_) <= (12)):
                                    out28_: _dafny.Seq
                                    out28_ = (d_0_helpers_).SafeBoostedConstrainedStep(lm, parser, d_21_constrainedPrompt_, currentConstrainedOut, d_32_preferred_, _dafny.BigRational('4e0'), eosToken)
                                    d_34_next_ = out28_
                                elif (d_22_validCount_) <= (12):
                                    out29_: _dafny.Seq
                                    out29_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_21_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                                    d_34_next_ = out29_
                                elif True:
                                    out30_: _dafny.Seq
                                    out30_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_21_constrainedPrompt_, currentConstrainedOut, eosToken)
                                    d_34_next_ = out30_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_34_next_) == (eosToken):
                                    if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                        d_35_closedGenerated3_: _dafny.Seq
                                        d_36_closedInside3_: bool
                                        d_37_closedCurrent3_: _dafny.Seq
                                        out31_: _dafny.Seq
                                        out32_: bool
                                        out33_: _dafny.Seq
                                        out31_, out32_, out33_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                        d_35_closedGenerated3_ = out31_
                                        d_36_closedInside3_ = out32_
                                        d_37_closedCurrent3_ = out33_
                                        generated = d_35_closedGenerated3_
                                        insideConstrainedOut = d_36_closedInside3_
                                        currentConstrainedOut = d_37_closedCurrent3_
                                        d_3_spanDone_ = True
                                        d_2_cueArmed_ = False
                                        out34_: _dafny.Seq
                                        out34_ = VerifiedDecoderAgent.CSDHelpers.ExtractAfterKeyword(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "answer")))
                                        d_5_recentAfterAnswer_ = out34_
                                        d_1_steps_ = (d_1_steps_) + (1)
                                    raise _dafny.Break("0")
                                elif True:
                                    d_38_appendedGenerated_: _dafny.Seq
                                    d_39_appendedInside_: bool
                                    d_40_appendedCurrent_: _dafny.Seq
                                    out35_: _dafny.Seq
                                    out36_: bool
                                    out37_: _dafny.Seq
                                    out35_, out36_, out37_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_34_next_)
                                    d_38_appendedGenerated_ = out35_
                                    d_39_appendedInside_ = out36_
                                    d_40_appendedCurrent_ = out37_
                                    generated = d_38_appendedGenerated_
                                    insideConstrainedOut = d_39_appendedInside_
                                    currentConstrainedOut = d_40_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

