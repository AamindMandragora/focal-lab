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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate exactly one SQL query using only the provided schema context. The complete answer must be exactly SQL: <<YOUR QUERY>>. Inside the delimiters emit only the SQL body: no explanation, Markdown, comments, or extra prose. Prefer schema table and column names exactly as given. Use joins when selected columns and filter columns come from different tables. For questions requiring the same entity to satisfy two separate conditions, prefer INTERSECT or GROUP BY/HAVING over simple either-value logic. Do not add a trailing semicolon. Close the query only when it is complete."))
        if (len(validTokenGroups)) > (0):
            d_1_guidance_ = (d_1_guidance_) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " Use parser-valid schema tokens from the supplied contextual token groups when they match the question, but keep the SQL plan faithful to the question.")))
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
        d_14_sqlKeywordGroups_: _dafny.Seq
        d_14_sqlKeywordGroups_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SELECT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "select")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "DISTINCT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "distinct"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "FROM")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "from"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "WHERE")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "where")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "AND")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "and")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "OR")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "or")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "NOT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "not")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "IN")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "in")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "LIKE")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "like")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "BETWEEN")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "between"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "JOIN")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "join")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "ON")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "on"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "GROUP")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "group")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "BY")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "by")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "HAVING")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "having"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "ORDER")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "order")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "DESC")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "desc")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "ASC")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "asc")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "LIMIT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "limit"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "INTERSECT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "intersect")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "UNION")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "union")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "EXCEPT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "except"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "COUNT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "count")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SUM")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "sum")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "AVG")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "avg")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "MIN")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "min")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "MAX")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "max"))])])
        d_15_eosPenaltyTokens_: _dafny.Seq
        d_15_eosPenaltyTokens_ = _dafny.SeqWithoutIsStrInference([eosToken, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "```")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\n\n")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ";"))])
        d_16_narrowThreshold_: int
        d_16_narrowThreshold_ = 20
        d_17_schemaThreshold_: int
        d_17_schemaThreshold_ = 32
        d_18_joinThreshold_: int
        d_18_joinThreshold_ = 48
        d_19_steps_: int
        d_19_steps_ = 0
        with _dafny.label("0"):
            while (d_19_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if not(d_3_headerDone_):
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SQL: "))]))
                            d_3_headerDone_ = True
                            d_19_steps_ = (d_19_steps_) + (1)
                        elif not(d_4_spanStarted_):
                            d_20_openedGenerated_: _dafny.Seq
                            d_21_openedInside_: bool
                            d_22_openedCurrent_: _dafny.Seq
                            out7_: _dafny.Seq
                            out8_: bool
                            out9_: _dafny.Seq
                            out7_, out8_, out9_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_20_openedGenerated_ = out7_
                            d_21_openedInside_ = out8_
                            d_22_openedCurrent_ = out9_
                            generated = d_20_openedGenerated_
                            insideConstrainedOut = d_21_openedInside_
                            currentConstrainedOut = d_22_openedCurrent_
                            d_4_spanStarted_ = True
                            d_11_seenFrom_ = False
                            d_12_seenWhere_ = False
                            d_13_seenJoin_ = False
                            d_19_steps_ = (d_19_steps_) + (1)
                        elif True:
                            d_23_sink_: _dafny.Seq
                            out10_: _dafny.Seq
                            out10_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_23_sink_ = out10_
                            d_19_steps_ = (d_19_steps_) + (1)
                            raise _dafny.Break("0")
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_24_closedGenerated_: _dafny.Seq
                        d_25_closedInside_: bool
                        d_26_closedCurrent_: _dafny.Seq
                        out11_: _dafny.Seq
                        out12_: bool
                        out13_: _dafny.Seq
                        out11_, out12_, out13_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_24_closedGenerated_ = out11_
                        d_25_closedInside_ = out12_
                        d_26_closedCurrent_ = out13_
                        generated = d_24_closedGenerated_
                        insideConstrainedOut = d_25_closedInside_
                        currentConstrainedOut = d_26_closedCurrent_
                        d_19_steps_ = (d_19_steps_) + (1)
                        raise _dafny.Break("0")
                    elif True:
                        d_27_constrainedPrompt_: _dafny.Seq
                        d_27_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_28_validCount_: int
                        out14_: int
                        out14_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                        d_28_validCount_ = out14_
                        d_29_next_: _dafny.Seq
                        d_29_next_ = eosToken
                        if not(d_11_seenFrom_):
                            d_30_openingGroups_: _dafny.Seq
                            d_30_openingGroups_ = (d_14_sqlKeywordGroups_) + (validTokenGroups)
                            out15_: _dafny.Seq
                            out15_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, d_27_constrainedPrompt_, currentConstrainedOut, d_30_openingGroups_, _dafny.BigRational('6e0'), eosToken)
                            d_29_next_ = out15_
                        elif (d_28_validCount_) <= (d_16_narrowThreshold_):
                            out16_: _dafny.Seq
                            out16_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_27_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('5e0'), d_16_narrowThreshold_, eosToken)
                            d_29_next_ = out16_
                        elif ((d_13_seenJoin_) and (not(d_12_seenWhere_))) and ((d_28_validCount_) <= (d_18_joinThreshold_)):
                            d_31_joinGroups_: _dafny.Seq
                            d_31_joinGroups_ = (validTokenGroups) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "ON")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "on"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "WHERE")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "where"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "AND")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "and"))])]))
                            out17_: _dafny.Seq
                            out17_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, d_27_constrainedPrompt_, currentConstrainedOut, d_31_joinGroups_, _dafny.BigRational('4e0'), eosToken)
                            d_29_next_ = out17_
                        elif (not(d_12_seenWhere_)) and ((d_28_validCount_) <= (d_17_schemaThreshold_)):
                            out18_: _dafny.Seq
                            out18_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_27_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('3e0'), d_17_schemaThreshold_, eosToken)
                            d_29_next_ = out18_
                        elif True:
                            d_32_gatedNext_: _dafny.Seq
                            d_33_wasConstrained_: bool
                            out19_: _dafny.Seq
                            out20_: bool
                            out19_, out20_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, d_27_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_32_gatedNext_ = out19_
                            d_33_wasConstrained_ = out20_
                            d_29_next_ = d_32_gatedNext_
                        d_19_steps_ = (d_19_steps_) + (1)
                        if ((d_29_next_) == (eosToken)) and ((d_19_steps_) < (maxSteps)):
                            out21_: _dafny.Seq
                            out21_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_27_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_15_eosPenaltyTokens_, _dafny.BigRational('8e0'), d_16_narrowThreshold_, eosToken)
                            d_29_next_ = out21_
                            d_19_steps_ = (d_19_steps_) + (1)
                        if ((d_29_next_) == (eosToken)) and ((d_19_steps_) < (maxSteps)):
                            d_34_retryGroups_: _dafny.Seq
                            d_34_retryGroups_ = (validTokenGroups) + (d_14_sqlKeywordGroups_)
                            out22_: _dafny.Seq
                            out22_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, d_27_constrainedPrompt_, currentConstrainedOut, d_34_retryGroups_, _dafny.BigRational('3e0'), eosToken)
                            d_29_next_ = out22_
                            d_19_steps_ = (d_19_steps_) + (1)
                        if (d_29_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_35_appendedGenerated_: _dafny.Seq
                            d_36_appendedInside_: bool
                            d_37_appendedCurrent_: _dafny.Seq
                            out23_: _dafny.Seq
                            out24_: bool
                            out25_: _dafny.Seq
                            out23_, out24_, out25_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_29_next_)
                            d_35_appendedGenerated_ = out23_
                            d_36_appendedInside_ = out24_
                            d_37_appendedCurrent_ = out25_
                            generated = d_35_appendedGenerated_
                            insideConstrainedOut = d_36_appendedInside_
                            currentConstrainedOut = d_37_appendedCurrent_
                            if ((d_29_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "FROM")))) or ((d_29_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "from")))):
                                d_11_seenFrom_ = True
                            elif ((d_29_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "WHERE")))) or ((d_29_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "where")))):
                                d_12_seenWhere_ = True
                            elif ((d_29_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "JOIN")))) or ((d_29_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "join")))):
                                d_13_seenJoin_ = True
                            if ((d_19_steps_) < (maxSteps)) and ((parser).IsCompletePrefix(currentConstrainedOut)):
                                d_38_closedGenerated2_: _dafny.Seq
                                d_39_closedInside2_: bool
                                d_40_closedCurrent2_: _dafny.Seq
                                out26_: _dafny.Seq
                                out27_: bool
                                out28_: _dafny.Seq
                                out26_, out27_, out28_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_38_closedGenerated2_ = out26_
                                d_39_closedInside2_ = out27_
                                d_40_closedCurrent2_ = out28_
                                generated = d_38_closedGenerated2_
                                insideConstrainedOut = d_39_closedInside2_
                                currentConstrainedOut = d_40_closedCurrent2_
                                d_19_steps_ = (d_19_steps_) + (1)
                                raise _dafny.Break("0")
                    pass
            pass
        cost = d_19_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

