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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Output exactly: SQL: <<query>> where query is a valid SQL SELECT using only the schema. Spider SQL rules: lowercase keywords, spaces around all operators and parentheses like count ( * ), use explicit JOIN table ON condition for multi-table queries, qualify columns as table.column when joining, use INTERSECT for 'both X and Y' queries, never use WHERE for join conditions across tables, no markdown, nothing after >>.")))
        d_1_seenFrom_: bool
        d_1_seenFrom_ = False
        d_2_seenWhere_: bool
        d_2_seenWhere_ = False
        d_3_sqlKeywordGroups_: _dafny.Seq
        d_3_sqlKeywordGroups_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "select")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SELECT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "distinct")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "DISTINCT"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "from")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "FROM"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "where")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "WHERE")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "and")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "AND")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "or")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "OR")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "not")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "NOT"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "join")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "JOIN")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "inner")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "INNER")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "left")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "LEFT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "on")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "ON"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "group")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "GROUP")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "by")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "BY")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "having")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "HAVING")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "order")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "ORDER")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "limit")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "LIMIT"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "count")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "COUNT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "sum")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SUM")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "avg")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "AVG")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "max")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "MAX")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "min")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "MIN")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "as")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "AS")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "*")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "(")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ")")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ","))])])
        d_4_narrowThreshold_: int
        d_4_narrowThreshold_ = 20
        d_5_steps_: int
        d_5_steps_ = 0
        with _dafny.label("0"):
            while (d_5_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_6_remaining_: int
                        d_6_remaining_ = (maxSteps) - (d_5_steps_)
                        d_7_chunkBudget_: int
                        if (d_6_remaining_) > (20):
                            d_7_chunkBudget_ = 20
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
                            d_1_seenFrom_ = False
                            d_2_seenWhere_ = False
                        elif True:
                            if (d_5_steps_) < (maxSteps):
                                d_15_g2_: _dafny.Seq
                                d_16_i2_: bool
                                d_17_c2_: _dafny.Seq
                                out7_: _dafny.Seq
                                out8_: bool
                                out9_: _dafny.Seq
                                out7_, out8_, out9_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                d_15_g2_ = out7_
                                d_16_i2_ = out8_
                                d_17_c2_ = out9_
                                generated = d_15_g2_
                                insideConstrainedOut = d_16_i2_
                                currentConstrainedOut = d_17_c2_
                                d_5_steps_ = (d_5_steps_) + (1)
                                d_1_seenFrom_ = False
                                d_2_seenWhere_ = False
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_18_g2_: _dafny.Seq
                        d_19_i2_: bool
                        d_20_c2_: _dafny.Seq
                        out10_: _dafny.Seq
                        out11_: bool
                        out12_: _dafny.Seq
                        out10_, out11_, out12_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_18_g2_ = out10_
                        d_19_i2_ = out11_
                        d_20_c2_ = out12_
                        generated = d_18_g2_
                        insideConstrainedOut = d_19_i2_
                        currentConstrainedOut = d_20_c2_
                        d_5_steps_ = (d_5_steps_) + (1)
                        raise _dafny.Break("0")
                    elif True:
                        d_21_constrainedPrompt_: _dafny.Seq
                        d_21_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_22_validCount_: int
                        out13_: int
                        out13_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                        d_22_validCount_ = out13_
                        d_23_next_: _dafny.Seq
                        d_23_next_ = eosToken
                        if (d_22_validCount_) <= (d_4_narrowThreshold_):
                            d_24_groups_: _dafny.Seq
                            d_24_groups_ = (d_3_sqlKeywordGroups_) + (validTokenGroups)
                            out14_: _dafny.Seq
                            out14_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, d_21_constrainedPrompt_, currentConstrainedOut, d_24_groups_, _dafny.BigRational('7e0'), eosToken)
                            d_23_next_ = out14_
                        elif (d_1_seenFrom_) and (not(d_2_seenWhere_)):
                            out15_: _dafny.Seq
                            out15_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_21_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('5e0'), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ";")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "LIMIT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "limit"))]), _dafny.BigRational('4e0'), d_4_narrowThreshold_, eosToken)
                            d_23_next_ = out15_
                        elif d_2_seenWhere_:
                            d_25_gatedNext_: _dafny.Seq
                            d_26_wasConstrained_: bool
                            out16_: _dafny.Seq
                            out17_: bool
                            out16_, out17_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, d_21_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_25_gatedNext_ = out16_
                            d_26_wasConstrained_ = out17_
                            d_23_next_ = d_25_gatedNext_
                        elif True:
                            out18_: _dafny.Seq
                            out18_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_21_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('5e0'), d_4_narrowThreshold_, eosToken)
                            d_23_next_ = out18_
                        d_5_steps_ = (d_5_steps_) + (1)
                        if (d_23_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_27_g2_: _dafny.Seq
                            d_28_i2_: bool
                            d_29_c2_: _dafny.Seq
                            out19_: _dafny.Seq
                            out20_: bool
                            out21_: _dafny.Seq
                            out19_, out20_, out21_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_23_next_)
                            d_27_g2_ = out19_
                            d_28_i2_ = out20_
                            d_29_c2_ = out21_
                            generated = d_27_g2_
                            insideConstrainedOut = d_28_i2_
                            currentConstrainedOut = d_29_c2_
                            if ((d_23_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "FROM")))) or ((d_23_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "from")))):
                                d_1_seenFrom_ = True
                            elif ((d_23_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "WHERE")))) or ((d_23_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "where")))):
                                d_2_seenWhere_ = True
                    pass
            pass
        cost = d_5_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

