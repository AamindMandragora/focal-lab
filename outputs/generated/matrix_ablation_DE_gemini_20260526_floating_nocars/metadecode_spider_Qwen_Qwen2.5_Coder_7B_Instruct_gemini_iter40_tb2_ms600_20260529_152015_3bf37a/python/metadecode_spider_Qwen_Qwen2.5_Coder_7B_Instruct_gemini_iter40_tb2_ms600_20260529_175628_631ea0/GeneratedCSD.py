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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate a single, syntactically correct and semantically appropriate SQL query that answers the user's question. Use the provided schema."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        if (not(insideConstrained)) and ((maxSteps) > (0)):
            d_3_chunkBudget_: int
            if (5) < (maxSteps):
                d_3_chunkBudget_ = 5
            elif True:
                d_3_chunkBudget_ = maxSteps
            d_4_generatedOut_: _dafny.Seq
            d_5_stoppedOnOpen_: bool
            d_6_stoppedOnEos_: bool
            d_7_stepsUsed_: int
            out0_: _dafny.Seq
            out1_: bool
            out2_: bool
            out3_: int
            out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_3_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
            d_4_generatedOut_ = out0_
            d_5_stoppedOnOpen_ = out1_
            d_6_stoppedOnEos_ = out2_
            d_7_stepsUsed_ = out3_
            d_2_steps_ = (d_2_steps_) + (d_7_stepsUsed_)
            generated = d_4_generatedOut_
            if d_6_stoppedOnEos_:
                cost = d_2_steps_
                return generated, insideConstrainedOut, currentConstrainedOut, cost
            if d_5_stoppedOnOpen_:
                d_8_gen_: _dafny.Seq
                d_9_inside_: bool
                d_10_current_: _dafny.Seq
                out4_: _dafny.Seq
                out5_: bool
                out6_: _dafny.Seq
                out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                d_8_gen_ = out4_
                d_9_inside_ = out5_
                d_10_current_ = out6_
                generated = d_8_gen_
                insideConstrainedOut = d_9_inside_
                currentConstrainedOut = d_10_current_
        with _dafny.label("0"):
            while (d_2_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_11_openedGenerated_: _dafny.Seq
                        d_12_openedInside_: bool
                        d_13_openedCurrent_: _dafny.Seq
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: _dafny.Seq
                        out7_, out8_, out9_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                        d_11_openedGenerated_ = out7_
                        d_12_openedInside_ = out8_
                        d_13_openedCurrent_ = out9_
                        generated = d_11_openedGenerated_
                        insideConstrainedOut = d_12_openedInside_
                        currentConstrainedOut = d_13_openedCurrent_
                        d_2_steps_ = (d_2_steps_) + (1)
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_14_closedGenerated_: _dafny.Seq
                        d_15_closedInside_: bool
                        d_16_closedCurrent_: _dafny.Seq
                        out10_: _dafny.Seq
                        out11_: bool
                        out12_: _dafny.Seq
                        out10_, out11_, out12_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_14_closedGenerated_ = out10_
                        d_15_closedInside_ = out11_
                        d_16_closedCurrent_ = out12_
                        generated = d_14_closedGenerated_
                        insideConstrainedOut = d_15_closedInside_
                        currentConstrainedOut = d_16_closedCurrent_
                        d_2_steps_ = (d_2_steps_) + (1)
                        raise _dafny.Break("0")
                    elif True:
                        d_17_constrainedPrompt_: _dafny.Seq
                        d_17_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_18_sqlKeywordGroups_: _dafny.Seq
                        d_18_sqlKeywordGroups_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SELECT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "DISTINCT"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "FROM"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "WHERE")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "AND")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "OR"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "JOIN")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "ON"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "GROUP")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "BY")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "HAVING")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "ORDER")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "LIMIT"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "INTERSECT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "UNION")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "EXCEPT"))])])
                        d_19_groupsToBoost_: _dafny.Seq
                        d_19_groupsToBoost_ = (d_18_sqlKeywordGroups_) + (validTokenGroups)
                        d_20_narrowThreshold_: int
                        d_20_narrowThreshold_ = 12
                        d_21_contains__where_: bool
                        d_21_contains__where_ = (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "WHERE"))) in (currentConstrainedOut)
                        d_22_contains__group_: bool
                        d_22_contains__group_ = (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "GROUP"))) in (currentConstrainedOut)
                        d_23_next_: _dafny.Seq
                        d_23_next_ = eosToken
                        if (d_21_contains__where_) or (d_22_contains__group_):
                            d_24_penaltyTokens_: _dafny.Seq
                            d_24_penaltyTokens_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "ORDER")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "LIMIT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ";"))])
                            out13_: _dafny.Seq
                            out13_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_17_constrainedPrompt_, currentConstrainedOut, d_19_groupsToBoost_, _dafny.BigRational('4e0'), d_24_penaltyTokens_, _dafny.BigRational('5e0'), d_20_narrowThreshold_, eosToken)
                            d_23_next_ = out13_
                        elif True:
                            out14_: _dafny.Seq
                            out14_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_17_constrainedPrompt_, currentConstrainedOut, d_19_groupsToBoost_, _dafny.BigRational('4e0'), d_20_narrowThreshold_, eosToken)
                            d_23_next_ = out14_
                        d_2_steps_ = (d_2_steps_) + (1)
                        if (d_23_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_25_appendedGenerated_: _dafny.Seq
                            d_26_appendedInside_: bool
                            d_27_appendedCurrent_: _dafny.Seq
                            out15_: _dafny.Seq
                            out16_: bool
                            out17_: _dafny.Seq
                            out15_, out16_, out17_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_23_next_)
                            d_25_appendedGenerated_ = out15_
                            d_26_appendedInside_ = out16_
                            d_27_appendedCurrent_ = out17_
                            generated = d_25_appendedGenerated_
                            insideConstrainedOut = d_26_appendedInside_
                            currentConstrainedOut = d_27_appendedCurrent_
                    pass
            pass
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

