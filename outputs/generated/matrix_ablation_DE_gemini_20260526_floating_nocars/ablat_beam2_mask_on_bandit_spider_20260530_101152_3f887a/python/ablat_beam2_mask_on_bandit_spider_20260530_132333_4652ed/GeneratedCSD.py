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
        if (maxSteps) == (0):
            cost = 0
            return generated, insideConstrainedOut, currentConstrainedOut, cost
        d_1_guidance_: _dafny.Seq
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate exactly one SQL query using only the provided schema context. The complete answer must be exactly SQL: <<YOUR QUERY>>. Inside the delimiters emit only the SQL body: no explanation, Markdown, comments, or extra prose. Prefer schema table and column names exactly as given. Use joins when selected columns and filter columns come from different tables. For questions requiring both of two values, prefer INTERSECT or GROUP BY/HAVING. Do not add a trailing semicolon. Close the query only when it is complete."))
        if (len(validTokenGroups)) > (0):
            d_1_guidance_ = (d_1_guidance_) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " Use supplied schema tokens when they clearly match the question, while preserving a parser-valid SQL plan.")))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_headerDone_: bool
        d_2_headerDone_ = (insideConstrainedOut) or ((len(generated)) > (0))
        d_3_spanStarted_: bool
        d_3_spanStarted_ = insideConstrainedOut
        d_4_seenSelectCount_: int
        out0_: int
        out0_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SELECT")))
        d_4_seenSelectCount_ = out0_
        d_5_seenLowerSelectCount_: int
        out1_: int
        out1_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "select")))
        d_5_seenLowerSelectCount_ = out1_
        d_6_seenFromCount_: int
        out2_: int
        out2_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "FROM")))
        d_6_seenFromCount_ = out2_
        d_7_seenLowerFromCount_: int
        out3_: int
        out3_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "from")))
        d_7_seenLowerFromCount_ = out3_
        d_8_seenWhereCount_: int
        out4_: int
        out4_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "WHERE")))
        d_8_seenWhereCount_ = out4_
        d_9_seenLowerWhereCount_: int
        out5_: int
        out5_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "where")))
        d_9_seenLowerWhereCount_ = out5_
        d_10_seenJoinCount_: int
        out6_: int
        out6_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "JOIN")))
        d_10_seenJoinCount_ = out6_
        d_11_seenLowerJoinCount_: int
        out7_: int
        out7_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "join")))
        d_11_seenLowerJoinCount_ = out7_
        d_12_seenSelect_: bool
        d_12_seenSelect_ = ((d_4_seenSelectCount_) > (0)) or ((d_5_seenLowerSelectCount_) > (0))
        d_13_seenFrom_: bool
        d_13_seenFrom_ = ((d_6_seenFromCount_) > (0)) or ((d_7_seenLowerFromCount_) > (0))
        d_14_seenWhere_: bool
        d_14_seenWhere_ = ((d_8_seenWhereCount_) > (0)) or ((d_9_seenLowerWhereCount_) > (0))
        d_15_seenJoin_: bool
        d_15_seenJoin_ = ((d_10_seenJoinCount_) > (0)) or ((d_11_seenLowerJoinCount_) > (0))
        d_16_sqlKeywordGroups_: _dafny.Seq
        d_16_sqlKeywordGroups_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SELECT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "select")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "DISTINCT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "distinct"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "FROM")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "from"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "WHERE")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "where")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "AND")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "and")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "OR")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "or")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "NOT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "not")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "LIKE")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "like")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "BETWEEN")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "between"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "JOIN")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "join")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "ON")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "on"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "GROUP")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "group")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "BY")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "by")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "HAVING")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "having"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "ORDER")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "order")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "DESC")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "desc")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "ASC")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "asc")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "LIMIT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "limit"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "INTERSECT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "intersect")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "UNION")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "union")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "EXCEPT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "except"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "COUNT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "count")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SUM")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "sum")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "AVG")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "avg")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "MIN")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "min")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "MAX")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "max"))])])
        d_17_eosPenaltyTokens_: _dafny.Seq
        d_17_eosPenaltyTokens_ = _dafny.SeqWithoutIsStrInference([eosToken, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "```")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\n\n")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ";"))])
        d_18_narrowThreshold_: int
        d_18_narrowThreshold_ = 12
        d_19_steps_: int
        d_19_steps_ = 0
        with _dafny.label("0"):
            while (d_19_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if not(d_2_headerDone_):
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SQL: "))]))
                            d_2_headerDone_ = True
                            d_19_steps_ = (d_19_steps_) + (1)
                        elif not(d_3_spanStarted_):
                            d_20_openedGenerated_: _dafny.Seq
                            d_21_openedInside_: bool
                            d_22_openedCurrent_: _dafny.Seq
                            out8_: _dafny.Seq
                            out9_: bool
                            out10_: _dafny.Seq
                            out8_, out9_, out10_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_20_openedGenerated_ = out8_
                            d_21_openedInside_ = out9_
                            d_22_openedCurrent_ = out10_
                            generated = d_20_openedGenerated_
                            insideConstrainedOut = d_21_openedInside_
                            currentConstrainedOut = d_22_openedCurrent_
                            d_3_spanStarted_ = True
                            d_12_seenSelect_ = False
                            d_13_seenFrom_ = False
                            d_14_seenWhere_ = False
                            d_15_seenJoin_ = False
                            d_19_steps_ = (d_19_steps_) + (1)
                        elif True:
                            d_23_sink_: _dafny.Seq
                            out11_: _dafny.Seq
                            out11_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_23_sink_ = out11_
                            d_19_steps_ = (d_19_steps_) + (1)
                            raise _dafny.Break("0")
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_24_closedGenerated_: _dafny.Seq
                        d_25_closedInside_: bool
                        d_26_closedCurrent_: _dafny.Seq
                        out12_: _dafny.Seq
                        out13_: bool
                        out14_: _dafny.Seq
                        out12_, out13_, out14_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_24_closedGenerated_ = out12_
                        d_25_closedInside_ = out13_
                        d_26_closedCurrent_ = out14_
                        generated = d_24_closedGenerated_
                        insideConstrainedOut = d_25_closedInside_
                        currentConstrainedOut = d_26_closedCurrent_
                        d_19_steps_ = (d_19_steps_) + (1)
                        raise _dafny.Break("0")
                    elif True:
                        d_27_constrainedPrompt_: _dafny.Seq
                        d_27_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_28_validCount_: int
                        out15_: int
                        out15_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                        d_28_validCount_ = out15_
                        d_29_next_: _dafny.Seq
                        d_29_next_ = eosToken
                        if (not(d_12_seenSelect_)) or ((not(d_13_seenFrom_)) and ((len(currentConstrainedOut)) <= (6))):
                            out16_: _dafny.Seq
                            out16_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, d_27_constrainedPrompt_, currentConstrainedOut, d_16_sqlKeywordGroups_, _dafny.BigRational('5e0'), eosToken)
                            d_29_next_ = out16_
                        elif (d_28_validCount_) <= (d_18_narrowThreshold_):
                            out17_: _dafny.Seq
                            out17_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_27_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_18_narrowThreshold_, eosToken)
                            d_29_next_ = out17_
                        elif ((d_15_seenJoin_) and (not(d_14_seenWhere_))) and ((d_28_validCount_) <= (24)):
                            d_30_joinGroups_: _dafny.Seq
                            d_30_joinGroups_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "ON")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "on"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "WHERE")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "where"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "AND")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "and"))])])
                            out18_: _dafny.Seq
                            out18_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, d_27_constrainedPrompt_, currentConstrainedOut, d_30_joinGroups_, _dafny.BigRational('3e0'), eosToken)
                            d_29_next_ = out18_
                        elif True:
                            d_31_gatedNext_: _dafny.Seq
                            d_32_wasConstrained_: bool
                            out19_: _dafny.Seq
                            out20_: bool
                            out19_, out20_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, d_27_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_31_gatedNext_ = out19_
                            d_32_wasConstrained_ = out20_
                            d_29_next_ = d_31_gatedNext_
                        d_19_steps_ = (d_19_steps_) + (1)
                        if ((d_29_next_) == (eosToken)) and ((d_19_steps_) < (maxSteps)):
                            out21_: _dafny.Seq
                            out21_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_27_constrainedPrompt_, currentConstrainedOut, d_16_sqlKeywordGroups_, _dafny.BigRational('3e0'), d_17_eosPenaltyTokens_, _dafny.BigRational('8e0'), d_18_narrowThreshold_, eosToken)
                            d_29_next_ = out21_
                            d_19_steps_ = (d_19_steps_) + (1)
                        if (d_29_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_33_appendedGenerated_: _dafny.Seq
                            d_34_appendedInside_: bool
                            d_35_appendedCurrent_: _dafny.Seq
                            out22_: _dafny.Seq
                            out23_: bool
                            out24_: _dafny.Seq
                            out22_, out23_, out24_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_29_next_)
                            d_33_appendedGenerated_ = out22_
                            d_34_appendedInside_ = out23_
                            d_35_appendedCurrent_ = out24_
                            generated = d_33_appendedGenerated_
                            insideConstrainedOut = d_34_appendedInside_
                            currentConstrainedOut = d_35_appendedCurrent_
                            if ((d_29_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SELECT")))) or ((d_29_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "select")))):
                                d_12_seenSelect_ = True
                            elif ((d_29_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "FROM")))) or ((d_29_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "from")))):
                                d_13_seenFrom_ = True
                            elif ((d_29_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "WHERE")))) or ((d_29_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "where")))):
                                d_14_seenWhere_ = True
                            elif ((d_29_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "JOIN")))) or ((d_29_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "join")))):
                                d_15_seenJoin_ = True
                            if (d_19_steps_) < (maxSteps):
                                if (parser).IsCompletePrefix(currentConstrainedOut):
                                    d_36_closedGenerated2_: _dafny.Seq
                                    d_37_closedInside2_: bool
                                    d_38_closedCurrent2_: _dafny.Seq
                                    out25_: _dafny.Seq
                                    out26_: bool
                                    out27_: _dafny.Seq
                                    out25_, out26_, out27_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_36_closedGenerated2_ = out25_
                                    d_37_closedInside2_ = out26_
                                    d_38_closedCurrent2_ = out27_
                                    generated = d_36_closedGenerated2_
                                    insideConstrainedOut = d_37_closedInside2_
                                    currentConstrainedOut = d_38_closedCurrent2_
                                    d_19_steps_ = (d_19_steps_) + (1)
                                    raise _dafny.Break("0")
                    pass
            pass
        cost = d_19_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

