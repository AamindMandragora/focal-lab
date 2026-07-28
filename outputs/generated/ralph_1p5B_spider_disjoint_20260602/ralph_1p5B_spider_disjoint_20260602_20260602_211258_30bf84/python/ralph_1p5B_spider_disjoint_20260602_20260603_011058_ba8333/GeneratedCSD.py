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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Output exactly: SQL: <<YOUR SQL QUERY HERE>>. The SQL query must be a single valid SQL SELECT statement using only the tables and columns from the provided schema. Keep queries simple and direct. No explanation, no markdown, no semicolon at end.")))
        d_1_sqlKeywordGroups_: _dafny.Seq
        d_1_sqlKeywordGroups_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SELECT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "select"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "DISTINCT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "distinct"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "FROM")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "from"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "WHERE")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "where"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "AND")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "and")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "OR")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "or"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "JOIN")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "join")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "LEFT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "left")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "INNER")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "inner")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "RIGHT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "right")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "OUTER")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "outer"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "ON")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "on"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "GROUP")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "group"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "BY")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "by"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "HAVING")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "having"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "ORDER")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "order"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "LIMIT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "limit"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "COUNT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "count")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SUM")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "sum")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "AVG")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "avg")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "MAX")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "max")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "MIN")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "min"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "AS")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "as"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "IN")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "in")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "NOT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "not")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "LIKE")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "like")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "BETWEEN")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "between")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "EXISTS")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "exists"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "INTERSECT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "intersect")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "UNION")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "union")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "EXCEPT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "except"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "*"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "(")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ")"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<=")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">=")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "!=")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<>"))])])
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_narrowThreshold_: int
        d_3_narrowThreshold_ = 16
        d_4_seenGroupBy_: bool
        d_4_seenGroupBy_ = False
        d_5_seenOrderBy_: bool
        d_5_seenOrderBy_ = False
        d_6_lastToken_: _dafny.Seq
        d_6_lastToken_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
        d_7_secondLastToken_: _dafny.Seq
        d_7_secondLastToken_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
        d_8_phase1Done_: bool
        d_8_phase1Done_ = insideConstrained
        d_9_phase1Steps_: int
        d_9_phase1Steps_ = 0
        while (((d_2_steps_) < (maxSteps)) and (not(d_8_phase1Done_))) and ((d_9_phase1Steps_) < (5)):
            d_10_next_: _dafny.Seq
            out0_: _dafny.Seq
            out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
            d_10_next_ = out0_
            d_2_steps_ = (d_2_steps_) + (1)
            d_9_phase1Steps_ = (d_9_phase1Steps_) + (1)
            if (d_10_next_) == (eosToken):
                cost = d_2_steps_
                return generated, insideConstrainedOut, currentConstrainedOut, cost
            elif (d_10_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                insideConstrainedOut = True
                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_10_next_]))
                d_8_phase1Done_ = True
            elif True:
                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_10_next_]))
                if (d_10_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ":"))):
                    d_8_phase1Done_ = True
        if ((d_2_steps_) < (maxSteps)) and (not(insideConstrainedOut)):
            d_11_openedGenerated_: _dafny.Seq
            d_12_openedInside_: bool
            d_13_openedCurrent_: _dafny.Seq
            out1_: _dafny.Seq
            out2_: bool
            out3_: _dafny.Seq
            out1_, out2_, out3_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_11_openedGenerated_ = out1_
            d_12_openedInside_ = out2_
            d_13_openedCurrent_ = out3_
            generated = d_11_openedGenerated_
            insideConstrainedOut = d_12_openedInside_
            currentConstrainedOut = d_13_openedCurrent_
            d_2_steps_ = (d_2_steps_) + (1)
        with _dafny.label("0"):
            while (d_2_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_14_next_: _dafny.Seq
                        out4_: _dafny.Seq
                        out4_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_14_next_ = out4_
                        d_2_steps_ = (d_2_steps_) + (1)
                        if (d_14_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_14_next_]))
                        raise _dafny.Break("0")
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_15_closedGenerated_: _dafny.Seq
                        d_16_closedInside_: bool
                        d_17_closedCurrent_: _dafny.Seq
                        out5_: _dafny.Seq
                        out6_: bool
                        out7_: _dafny.Seq
                        out5_, out6_, out7_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_15_closedGenerated_ = out5_
                        d_16_closedInside_ = out6_
                        d_17_closedCurrent_ = out7_
                        generated = d_15_closedGenerated_
                        insideConstrainedOut = d_16_closedInside_
                        currentConstrainedOut = d_17_closedCurrent_
                        d_2_steps_ = (d_2_steps_) + (1)
                        raise _dafny.Break("0")
                    elif True:
                        d_18_constrainedPrompt_: _dafny.Seq
                        d_18_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_19_validCount_: int
                        out8_: int
                        out8_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                        d_19_validCount_ = out8_
                        d_20_next_: _dafny.Seq
                        d_20_next_ = eosToken
                        d_21_inRepetitionRisk_: bool
                        d_21_inRepetitionRisk_ = (d_4_seenGroupBy_) or (d_5_seenOrderBy_)
                        if (d_19_validCount_) <= (d_3_narrowThreshold_):
                            d_22_allGroups_: _dafny.Seq
                            d_22_allGroups_ = (d_1_sqlKeywordGroups_) + (validTokenGroups)
                            out9_: _dafny.Seq
                            out9_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, d_18_constrainedPrompt_, currentConstrainedOut, d_22_allGroups_, _dafny.BigRational('5e0'), eosToken)
                            d_20_next_ = out9_
                        elif (d_21_inRepetitionRisk_) and ((d_6_lastToken_) != (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "")))):
                            d_23_penaltyTokens_: _dafny.Seq = _dafny.Seq({})
                            if ((d_7_secondLastToken_) != (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "")))) and ((d_7_secondLastToken_) != (d_6_lastToken_)):
                                d_23_penaltyTokens_ = _dafny.SeqWithoutIsStrInference([d_6_lastToken_, d_7_secondLastToken_])
                            elif True:
                                d_23_penaltyTokens_ = _dafny.SeqWithoutIsStrInference([d_6_lastToken_])
                            out10_: _dafny.Seq
                            out10_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_18_constrainedPrompt_, currentConstrainedOut, (d_1_sqlKeywordGroups_) + (validTokenGroups), _dafny.BigRational('3e0'), d_23_penaltyTokens_, _dafny.BigRational('6e0'), d_3_narrowThreshold_, eosToken)
                            d_20_next_ = out10_
                        elif True:
                            d_24_gatedNext_: _dafny.Seq
                            d_25_wasConstrained_: bool
                            out11_: _dafny.Seq
                            out12_: bool
                            out11_, out12_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, d_18_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_24_gatedNext_ = out11_
                            d_25_wasConstrained_ = out12_
                            d_20_next_ = d_24_gatedNext_
                        d_2_steps_ = (d_2_steps_) + (1)
                        if (d_20_next_) == (eosToken):
                            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_2_steps_) < (maxSteps)):
                                d_26_closedGenerated_: _dafny.Seq
                                d_27_closedInside_: bool
                                d_28_closedCurrent_: _dafny.Seq
                                out13_: _dafny.Seq
                                out14_: bool
                                out15_: _dafny.Seq
                                out13_, out14_, out15_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_26_closedGenerated_ = out13_
                                d_27_closedInside_ = out14_
                                d_28_closedCurrent_ = out15_
                                generated = d_26_closedGenerated_
                                insideConstrainedOut = d_27_closedInside_
                                currentConstrainedOut = d_28_closedCurrent_
                                d_2_steps_ = (d_2_steps_) + (1)
                            raise _dafny.Break("0")
                        elif True:
                            d_7_secondLastToken_ = d_6_lastToken_
                            d_6_lastToken_ = d_20_next_
                            if ((d_20_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "GROUP")))) or ((d_20_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "group")))):
                                d_4_seenGroupBy_ = True
                            elif ((d_20_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "ORDER")))) or ((d_20_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "order")))):
                                d_5_seenOrderBy_ = True
                            elif ((((d_20_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "WHERE")))) or ((d_20_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "where"))))) or ((d_20_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "HAVING"))))) or ((d_20_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "having")))):
                                d_4_seenGroupBy_ = False
                                d_5_seenOrderBy_ = False
                            d_29_appendedGenerated_: _dafny.Seq
                            d_30_appendedInside_: bool
                            d_31_appendedCurrent_: _dafny.Seq
                            out16_: _dafny.Seq
                            out17_: bool
                            out18_: _dafny.Seq
                            out16_, out17_, out18_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_20_next_)
                            d_29_appendedGenerated_ = out16_
                            d_30_appendedInside_ = out17_
                            d_31_appendedCurrent_ = out18_
                            generated = d_29_appendedGenerated_
                            insideConstrainedOut = d_30_appendedInside_
                            currentConstrainedOut = d_31_appendedCurrent_
                    pass
            pass
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

