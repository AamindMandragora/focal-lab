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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate exactly one SQL query using only the provided schema context. The complete answer must be exactly SQL: <<YOUR QUERY>>. Inside the delimiters emit only the SQL body: no explanation, Markdown, comments, or extra prose. Prefer the schema table and column names exactly as given. Use joins when selected columns and filter columns come from different tables. For questions requiring both of two values, use INTERSECT or GROUP BY/HAVING rather than simple either-value logic. Do not add a trailing semicolon. Close the query only when it is complete."))
        if (len(validTokenGroups)) > (0):
            d_1_guidance_ = (d_1_guidance_) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " Use parser-valid schema tokens from the supplied contextual token groups when they match the question.")))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_openCount_: int
        out0_: int
        out0_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))
        d_2_openCount_ = out0_
        d_3_headerDone_: bool
        d_3_headerDone_ = (insideConstrainedOut) or ((len(generated)) > (0))
        d_4_spanStarted_: bool
        d_4_spanStarted_ = (insideConstrainedOut) or ((d_2_openCount_) > (0))
        d_5_seenFromCount_: int
        out1_: int
        out1_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "FROM")))
        d_5_seenFromCount_ = out1_
        d_6_seenLowerFromCount_: int
        out2_: int
        out2_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "from")))
        d_6_seenLowerFromCount_ = out2_
        d_7_seenWhereCount_: int
        out3_: int
        out3_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "WHERE")))
        d_7_seenWhereCount_ = out3_
        d_8_seenLowerWhereCount_: int
        out4_: int
        out4_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "where")))
        d_8_seenLowerWhereCount_ = out4_
        d_9_seenJoinCount_: int
        out5_: int
        out5_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "JOIN")))
        d_9_seenJoinCount_ = out5_
        d_10_seenLowerJoinCount_: int
        out6_: int
        out6_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "join")))
        d_10_seenLowerJoinCount_ = out6_
        d_11_seenFrom_: bool
        d_11_seenFrom_ = ((d_5_seenFromCount_) > (0)) or ((d_6_seenLowerFromCount_) > (0))
        d_12_seenWhere_: bool
        d_12_seenWhere_ = ((d_7_seenWhereCount_) > (0)) or ((d_8_seenLowerWhereCount_) > (0))
        d_13_seenJoin_: bool
        d_13_seenJoin_ = ((d_9_seenJoinCount_) > (0)) or ((d_10_seenLowerJoinCount_) > (0))
        d_14_openingBoostLeft_: int
        d_14_openingBoostLeft_ = 12
        d_15_afterFromBoostLeft_: int
        d_15_afterFromBoostLeft_ = 0
        d_16_afterJoinBoostLeft_: int
        d_16_afterJoinBoostLeft_ = 0
        d_17_sqlKeywordGroups_: _dafny.Seq
        d_17_sqlKeywordGroups_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SELECT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "select")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "DISTINCT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "distinct"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "FROM")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "from"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "WHERE")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "where")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "AND")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "and")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "OR")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "or")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "NOT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "not")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "IN")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "in")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "LIKE")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "like")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "BETWEEN")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "between"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "JOIN")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "join")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "ON")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "on"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "GROUP")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "group")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "BY")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "by")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "HAVING")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "having"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "ORDER")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "order")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "DESC")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "desc")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "ASC")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "asc")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "LIMIT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "limit"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "INTERSECT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "intersect")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "UNION")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "union")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "EXCEPT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "except"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "COUNT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "count")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SUM")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "sum")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "AVG")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "avg")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "MIN")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "min")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "MAX")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "max"))])])
        d_18_eosPenaltyTokens_: _dafny.Seq
        d_18_eosPenaltyTokens_ = _dafny.SeqWithoutIsStrInference([eosToken, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "```")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\n\n")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ";"))])
        d_19_narrowThreshold_: int
        d_19_narrowThreshold_ = 10
        d_20_localLimit_: int
        d_20_localLimit_ = maxSteps
        if (d_20_localLimit_) > (110):
            d_20_localLimit_ = 110
        d_21_steps_: int
        d_21_steps_ = 0
        with _dafny.label("0"):
            while (d_21_steps_) < (d_20_localLimit_):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if not(d_3_headerDone_):
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SQL: "))]))
                            d_3_headerDone_ = True
                            d_21_steps_ = (d_21_steps_) + (1)
                        elif not(d_4_spanStarted_):
                            d_22_openedGenerated_: _dafny.Seq
                            d_23_openedInside_: bool
                            d_24_openedCurrent_: _dafny.Seq
                            out7_: _dafny.Seq
                            out8_: bool
                            out9_: _dafny.Seq
                            out7_, out8_, out9_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_22_openedGenerated_ = out7_
                            d_23_openedInside_ = out8_
                            d_24_openedCurrent_ = out9_
                            generated = d_22_openedGenerated_
                            insideConstrainedOut = d_23_openedInside_
                            currentConstrainedOut = d_24_openedCurrent_
                            d_4_spanStarted_ = True
                            d_11_seenFrom_ = False
                            d_12_seenWhere_ = False
                            d_13_seenJoin_ = False
                            d_14_openingBoostLeft_ = 12
                            d_15_afterFromBoostLeft_ = 0
                            d_16_afterJoinBoostLeft_ = 0
                            d_21_steps_ = (d_21_steps_) + (1)
                        elif True:
                            d_25_sink_: _dafny.Seq
                            out10_: _dafny.Seq
                            out10_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_25_sink_ = out10_
                            d_21_steps_ = (d_21_steps_) + (1)
                            raise _dafny.Break("0")
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_26_closedGenerated_: _dafny.Seq
                        d_27_closedInside_: bool
                        d_28_closedCurrent_: _dafny.Seq
                        out11_: _dafny.Seq
                        out12_: bool
                        out13_: _dafny.Seq
                        out11_, out12_, out13_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_26_closedGenerated_ = out11_
                        d_27_closedInside_ = out12_
                        d_28_closedCurrent_ = out13_
                        generated = d_26_closedGenerated_
                        insideConstrainedOut = d_27_closedInside_
                        currentConstrainedOut = d_28_closedCurrent_
                        d_21_steps_ = (d_21_steps_) + (1)
                        raise _dafny.Break("0")
                    elif True:
                        d_29_constrainedPrompt_: _dafny.Seq
                        d_29_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_30_next_: _dafny.Seq
                        d_30_next_ = eosToken
                        if (not(d_11_seenFrom_)) and ((d_14_openingBoostLeft_) > (0)):
                            d_31_openingGroups_: _dafny.Seq
                            d_31_openingGroups_ = (d_17_sqlKeywordGroups_) + (validTokenGroups)
                            out14_: _dafny.Seq
                            out14_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, d_29_constrainedPrompt_, currentConstrainedOut, d_31_openingGroups_, _dafny.BigRational('45e-1'), eosToken)
                            d_30_next_ = out14_
                        elif (d_15_afterFromBoostLeft_) > (0):
                            d_32_fromGroups_: _dafny.Seq
                            d_32_fromGroups_ = (validTokenGroups) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "JOIN")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "join"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "WHERE")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "where"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "ON")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "on"))])]))
                            out15_: _dafny.Seq
                            out15_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, d_29_constrainedPrompt_, currentConstrainedOut, d_32_fromGroups_, _dafny.BigRational('3e0'), eosToken)
                            d_30_next_ = out15_
                        elif ((d_13_seenJoin_) and (not(d_12_seenWhere_))) and ((d_16_afterJoinBoostLeft_) > (0)):
                            d_33_joinGroups_: _dafny.Seq
                            d_33_joinGroups_ = (validTokenGroups) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "ON")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "on"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "WHERE")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "where"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "AND")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "and"))])]))
                            out16_: _dafny.Seq
                            out16_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, d_29_constrainedPrompt_, currentConstrainedOut, d_33_joinGroups_, _dafny.BigRational('3e0'), eosToken)
                            d_30_next_ = out16_
                        elif True:
                            d_34_gatedNext_: _dafny.Seq
                            d_35_wasConstrained_: bool
                            out17_: _dafny.Seq
                            out18_: bool
                            out17_, out18_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, d_29_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_34_gatedNext_ = out17_
                            d_35_wasConstrained_ = out18_
                            d_30_next_ = d_34_gatedNext_
                        d_21_steps_ = (d_21_steps_) + (1)
                        if ((d_30_next_) == (eosToken)) and ((d_21_steps_) < (d_20_localLimit_)):
                            out19_: _dafny.Seq
                            out19_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_29_constrainedPrompt_, currentConstrainedOut, _dafny.SeqWithoutIsStrInference([]), _dafny.BigRational('0e0'), d_18_eosPenaltyTokens_, _dafny.BigRational('8e0'), d_19_narrowThreshold_, eosToken)
                            d_30_next_ = out19_
                            d_21_steps_ = (d_21_steps_) + (1)
                        if (d_30_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_36_appendedGenerated_: _dafny.Seq
                            d_37_appendedInside_: bool
                            d_38_appendedCurrent_: _dafny.Seq
                            out20_: _dafny.Seq
                            out21_: bool
                            out22_: _dafny.Seq
                            out20_, out21_, out22_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_30_next_)
                            d_36_appendedGenerated_ = out20_
                            d_37_appendedInside_ = out21_
                            d_38_appendedCurrent_ = out22_
                            generated = d_36_appendedGenerated_
                            insideConstrainedOut = d_37_appendedInside_
                            currentConstrainedOut = d_38_appendedCurrent_
                            if (d_14_openingBoostLeft_) > (0):
                                d_14_openingBoostLeft_ = (d_14_openingBoostLeft_) - (1)
                            if (d_15_afterFromBoostLeft_) > (0):
                                d_15_afterFromBoostLeft_ = (d_15_afterFromBoostLeft_) - (1)
                            if (d_16_afterJoinBoostLeft_) > (0):
                                d_16_afterJoinBoostLeft_ = (d_16_afterJoinBoostLeft_) - (1)
                            if ((d_30_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "FROM")))) or ((d_30_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "from")))):
                                d_11_seenFrom_ = True
                                d_15_afterFromBoostLeft_ = 4
                            elif ((d_30_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "WHERE")))) or ((d_30_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "where")))):
                                d_12_seenWhere_ = True
                            elif ((d_30_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "JOIN")))) or ((d_30_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "join")))):
                                d_13_seenJoin_ = True
                                d_16_afterJoinBoostLeft_ = 4
                            if ((d_21_steps_) < (d_20_localLimit_)) and ((parser).IsCompletePrefix(currentConstrainedOut)):
                                d_39_closedGenerated2_: _dafny.Seq
                                d_40_closedInside2_: bool
                                d_41_closedCurrent2_: _dafny.Seq
                                out23_: _dafny.Seq
                                out24_: bool
                                out25_: _dafny.Seq
                                out23_, out24_, out25_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_39_closedGenerated2_ = out23_
                                d_40_closedInside2_ = out24_
                                d_41_closedCurrent2_ = out25_
                                generated = d_39_closedGenerated2_
                                insideConstrainedOut = d_40_closedInside2_
                                currentConstrainedOut = d_41_closedCurrent2_
                                d_21_steps_ = (d_21_steps_) + (1)
                                raise _dafny.Break("0")
                    pass
            pass
        cost = d_21_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

