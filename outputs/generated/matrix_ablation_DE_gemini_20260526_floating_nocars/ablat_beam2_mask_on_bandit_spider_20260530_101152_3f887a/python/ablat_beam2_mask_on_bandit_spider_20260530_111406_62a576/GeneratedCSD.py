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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate exactly one SQL query using only the provided schema context. The complete answer must be exactly SQL: <<YOUR QUERY>>. Inside the delimiters emit only the SQL body: no explanation, Markdown, comments, or extra prose. Prefer explicit table.column names from the schema over invented aliases such as T1 or T2. Use correct join paths from the schema. For questions requiring both of two values, use INTERSECT or GROUP BY/HAVING when appropriate. Put spaces around SQL keywords, joins, operators, and parentheses, and close the query as soon as it is complete."))
        if (len(validTokenGroups)) > (0):
            d_1_guidance_ = (d_1_guidance_) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " Prefer parser-valid schema tokens from the supplied contextual token groups when they match the question.")))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_openCount_: int
        out0_: int
        out0_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))
        d_2_openCount_ = out0_
        d_3_headerDone_: bool
        d_3_headerDone_ = (insideConstrainedOut) or ((len(generated)) > (0))
        d_4_spanStarted_: bool
        d_4_spanStarted_ = (insideConstrainedOut) or ((d_2_openCount_) > (0))
        d_5_sqlKeywordGroups_: _dafny.Seq
        d_5_sqlKeywordGroups_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SELECT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "select")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "DISTINCT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "distinct"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "FROM")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "from"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "WHERE")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "where")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "AND")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "and")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "OR")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "or"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "JOIN")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "join")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "ON")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "on"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "GROUP")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "group")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "HAVING")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "having")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "ORDER")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "order")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "BY")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "by"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "LIMIT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "limit")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "INTERSECT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "intersect")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "UNION")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "union"))])])
        d_6_penaltyTokens_: _dafny.Seq
        d_6_penaltyTokens_ = _dafny.SeqWithoutIsStrInference([eosToken, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "```")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\n\n")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "T1")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "T2")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "T3")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "t1")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "t2")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "AS")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "as"))])
        d_7_earlyGroupLimit_: int
        d_7_earlyGroupLimit_ = 10
        d_8_narrowThreshold_: int
        d_8_narrowThreshold_ = 6
        d_9_steps_: int
        d_9_steps_ = 0
        with _dafny.label("0"):
            while (d_9_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if not(d_3_headerDone_):
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SQL: "))]))
                            d_3_headerDone_ = True
                            d_9_steps_ = (d_9_steps_) + (1)
                        elif not(d_4_spanStarted_):
                            d_10_openedGenerated_: _dafny.Seq
                            d_11_openedInside_: bool
                            d_12_openedCurrent_: _dafny.Seq
                            out1_: _dafny.Seq
                            out2_: bool
                            out3_: _dafny.Seq
                            out1_, out2_, out3_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_10_openedGenerated_ = out1_
                            d_11_openedInside_ = out2_
                            d_12_openedCurrent_ = out3_
                            generated = d_10_openedGenerated_
                            insideConstrainedOut = d_11_openedInside_
                            currentConstrainedOut = d_12_openedCurrent_
                            d_4_spanStarted_ = True
                            d_9_steps_ = (d_9_steps_) + (1)
                        elif True:
                            d_13_sink_: _dafny.Seq
                            out4_: _dafny.Seq
                            out4_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_13_sink_ = out4_
                            d_9_steps_ = (d_9_steps_) + (1)
                            raise _dafny.Break("0")
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_14_closedGenerated_: _dafny.Seq
                        d_15_closedInside_: bool
                        d_16_closedCurrent_: _dafny.Seq
                        out5_: _dafny.Seq
                        out6_: bool
                        out7_: _dafny.Seq
                        out5_, out6_, out7_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_14_closedGenerated_ = out5_
                        d_15_closedInside_ = out6_
                        d_16_closedCurrent_ = out7_
                        generated = d_14_closedGenerated_
                        insideConstrainedOut = d_15_closedInside_
                        currentConstrainedOut = d_16_closedCurrent_
                        d_9_steps_ = (d_9_steps_) + (1)
                        raise _dafny.Break("0")
                    elif True:
                        d_17_constrainedPrompt_: _dafny.Seq
                        d_17_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_18_next_: _dafny.Seq
                        d_18_next_ = eosToken
                        if (len(currentConstrainedOut)) < (d_7_earlyGroupLimit_):
                            d_19_groups_: _dafny.Seq
                            d_19_groups_ = (d_5_sqlKeywordGroups_) + (validTokenGroups)
                            out8_: _dafny.Seq
                            out8_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, d_17_constrainedPrompt_, currentConstrainedOut, d_19_groups_, _dafny.BigRational('4e0'), eosToken)
                            d_18_next_ = out8_
                        elif True:
                            out9_: _dafny.Seq
                            out9_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_17_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('3e0'), d_6_penaltyTokens_, _dafny.BigRational('7e0'), d_8_narrowThreshold_, eosToken)
                            d_18_next_ = out9_
                        d_9_steps_ = (d_9_steps_) + (1)
                        if ((d_18_next_) == (eosToken)) and ((d_9_steps_) < (maxSteps)):
                            out10_: _dafny.Seq
                            out10_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_17_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('2e0'), d_6_penaltyTokens_, _dafny.BigRational('1e1'), d_8_narrowThreshold_, eosToken)
                            d_18_next_ = out10_
                            d_9_steps_ = (d_9_steps_) + (1)
                        if (d_18_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_20_appendedGenerated_: _dafny.Seq
                            d_21_appendedInside_: bool
                            d_22_appendedCurrent_: _dafny.Seq
                            out11_: _dafny.Seq
                            out12_: bool
                            out13_: _dafny.Seq
                            out11_, out12_, out13_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_18_next_)
                            d_20_appendedGenerated_ = out11_
                            d_21_appendedInside_ = out12_
                            d_22_appendedCurrent_ = out13_
                            generated = d_20_appendedGenerated_
                            insideConstrainedOut = d_21_appendedInside_
                            currentConstrainedOut = d_22_appendedCurrent_
                            if ((d_9_steps_) < (maxSteps)) and ((parser).IsCompletePrefix(currentConstrainedOut)):
                                d_23_closedGenerated2_: _dafny.Seq
                                d_24_closedInside2_: bool
                                d_25_closedCurrent2_: _dafny.Seq
                                out14_: _dafny.Seq
                                out15_: bool
                                out16_: _dafny.Seq
                                out14_, out15_, out16_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_23_closedGenerated2_ = out14_
                                d_24_closedInside2_ = out15_
                                d_25_closedCurrent2_ = out16_
                                generated = d_23_closedGenerated2_
                                insideConstrainedOut = d_24_closedInside2_
                                currentConstrainedOut = d_25_closedCurrent2_
                                d_9_steps_ = (d_9_steps_) + (1)
                                raise _dafny.Break("0")
                    pass
            pass
        cost = d_9_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

