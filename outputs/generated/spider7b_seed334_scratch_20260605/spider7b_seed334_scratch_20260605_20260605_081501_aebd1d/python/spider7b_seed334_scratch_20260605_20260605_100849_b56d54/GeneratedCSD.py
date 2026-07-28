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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Output exactly one SQL query in the format: SQL: <<YOUR QUERY>>. No explanation, no markdown. The query must be valid SQL using only the provided schema. Place the entire SQL query between << and >>.")))
        d_1_sqlKeywordGroups_: _dafny.Seq
        d_1_sqlKeywordGroups_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SELECT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SELECT DISTINCT"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "FROM"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "WHERE")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "AND")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "OR")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "NOT"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "JOIN")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "INNER JOIN")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "LEFT JOIN")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "ON"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "GROUP BY")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "HAVING")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "ORDER BY")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "LIMIT"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "COUNT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SUM")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "AVG")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "MAX")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "MIN"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "AS")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "IN")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "LIKE")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "BETWEEN")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "IS")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "NULL"))])])
        d_2_seenFrom_: bool
        d_2_seenFrom_ = False
        d_3_seenWhere_: bool
        d_3_seenWhere_ = False
        d_4_narrowThreshold_: int
        d_4_narrowThreshold_ = 12
        d_5_steps_: int
        d_5_steps_ = 0
        with _dafny.label("0"):
            while (d_5_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_6_remaining_: int
                        d_6_remaining_ = (maxSteps) - (d_5_steps_)
                        d_7_chunkBudget_: int
                        if (d_6_remaining_) > (40):
                            d_7_chunkBudget_ = 40
                        elif True:
                            d_7_chunkBudget_ = d_6_remaining_
                        d_8_generatedOut_: _dafny.Seq
                        d_9_stoppedOnOpenSpan_: bool
                        d_10_stoppedOnEos_: bool
                        d_11_stepsUsed_: int
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: bool
                        out3_: int
                        out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_7_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_8_generatedOut_ = out0_
                        d_9_stoppedOnOpenSpan_ = out1_
                        d_10_stoppedOnEos_ = out2_
                        d_11_stepsUsed_ = out3_
                        d_5_steps_ = (d_5_steps_) + (d_11_stepsUsed_)
                        generated = d_8_generatedOut_
                        if d_10_stoppedOnEos_:
                            raise _dafny.Break("0")
                        elif d_9_stoppedOnOpenSpan_:
                            d_12_g2_: _dafny.Seq
                            d_13_i2_: bool
                            d_14_c2_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                            d_12_g2_ = out4_
                            d_13_i2_ = out5_
                            d_14_c2_ = out6_
                            generated = d_12_g2_
                            insideConstrainedOut = d_13_i2_
                            currentConstrainedOut = d_14_c2_
                            d_2_seenFrom_ = False
                            d_3_seenWhere_ = False
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
                        d_5_steps_ = (d_5_steps_) + (1)
                    elif True:
                        d_18_constrainedPrompt_: _dafny.Seq
                        d_18_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_19_next_: _dafny.Seq
                        d_19_next_ = eosToken
                        d_20_validCount_: int
                        out10_: int
                        out10_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                        d_20_validCount_ = out10_
                        if (not(d_2_seenFrom_)) and ((d_20_validCount_) <= (d_4_narrowThreshold_)):
                            d_21_groups_: _dafny.Seq
                            d_21_groups_ = (d_1_sqlKeywordGroups_) + (validTokenGroups)
                            out11_: _dafny.Seq
                            out11_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, d_18_constrainedPrompt_, currentConstrainedOut, d_21_groups_, _dafny.BigRational('6e0'), eosToken)
                            d_19_next_ = out11_
                        elif (d_2_seenFrom_) and (not(d_3_seenWhere_)):
                            out12_: _dafny.Seq
                            out12_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_18_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ";")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "LIMIT"))]), _dafny.BigRational('3e0'), d_4_narrowThreshold_, eosToken)
                            d_19_next_ = out12_
                        elif True:
                            out13_: _dafny.Seq
                            out13_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_18_constrainedPrompt_, currentConstrainedOut, (d_1_sqlKeywordGroups_) + (validTokenGroups), _dafny.BigRational('4e0'), d_4_narrowThreshold_, eosToken)
                            d_19_next_ = out13_
                        d_5_steps_ = (d_5_steps_) + (1)
                        if (d_19_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_22_appendedGenerated_: _dafny.Seq
                            d_23_appendedInside_: bool
                            d_24_appendedCurrent_: _dafny.Seq
                            out14_: _dafny.Seq
                            out15_: bool
                            out16_: _dafny.Seq
                            out14_, out15_, out16_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_19_next_)
                            d_22_appendedGenerated_ = out14_
                            d_23_appendedInside_ = out15_
                            d_24_appendedCurrent_ = out16_
                            generated = d_22_appendedGenerated_
                            insideConstrainedOut = d_23_appendedInside_
                            currentConstrainedOut = d_24_appendedCurrent_
                            if (d_19_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "FROM"))):
                                d_2_seenFrom_ = True
                            elif (d_19_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "WHERE"))):
                                d_3_seenWhere_ = True
                    pass
            pass
        cost = d_5_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

