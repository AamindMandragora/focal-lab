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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate exactly one SQL query using only the provided schema context. The complete answer must be exactly SQL: <<YOUR QUERY>>. Inside the delimiters emit only the SQL body: no explanation, Markdown, comments, or trailing semicolon. Prefer Spider reference style: lowercase SQL keywords and functions, use the table and column names exactly as given, and avoid table aliases unless the same table must be joined more than once. Do not invent columns. Do not add unnecessary joins; join only tables needed for selected columns or filter conditions. For questions requiring both of two values, prefer INTERSECT or GROUP BY/HAVING when that matches the schema semantics. Finish only after the SQL query is complete."))
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
        d_9_seenFrom_: bool
        d_9_seenFrom_ = ((d_5_seenFromCount_) > (0)) or ((d_6_seenLowerFromCount_) > (0))
        d_10_seenWhere_: bool
        d_10_seenWhere_ = ((d_7_seenWhereCount_) > (0)) or ((d_8_seenLowerWhereCount_) > (0))
        d_11_sqlStyleGroups_: _dafny.Seq
        d_11_sqlStyleGroups_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "select")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "distinct"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "from"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "where")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "and")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "or")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "not")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "in")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "like")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "between"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "join")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "on"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "group")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "by")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "having"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "order")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "limit"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "intersect")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "union")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "except"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "count")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "sum")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "avg")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "min")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "max"))])])
        d_12_penaltyTokens_: _dafny.Seq
        d_12_penaltyTokens_ = _dafny.SeqWithoutIsStrInference([eosToken, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "```")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\n\n")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ";")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "AS")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "as")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "T1")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "T2")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "T3")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "T4")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "T5")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "T6")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "t1")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "t2")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "t3")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "t4")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "t5")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "t6"))])
        d_13_narrowThreshold_: int
        d_13_narrowThreshold_ = 200
        d_14_steps_: int
        d_14_steps_ = 0
        with _dafny.label("0"):
            while (d_14_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if not(d_3_headerDone_):
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SQL: "))]))
                            d_3_headerDone_ = True
                            d_14_steps_ = (d_14_steps_) + (1)
                        elif not(d_4_spanStarted_):
                            d_15_openedGenerated_: _dafny.Seq
                            d_16_openedInside_: bool
                            d_17_openedCurrent_: _dafny.Seq
                            out5_: _dafny.Seq
                            out6_: bool
                            out7_: _dafny.Seq
                            out5_, out6_, out7_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_15_openedGenerated_ = out5_
                            d_16_openedInside_ = out6_
                            d_17_openedCurrent_ = out7_
                            generated = d_15_openedGenerated_
                            insideConstrainedOut = d_16_openedInside_
                            currentConstrainedOut = d_17_openedCurrent_
                            d_4_spanStarted_ = True
                            d_9_seenFrom_ = False
                            d_10_seenWhere_ = False
                            d_14_steps_ = (d_14_steps_) + (1)
                        elif True:
                            d_18_sink_: _dafny.Seq
                            out8_: _dafny.Seq
                            out8_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_18_sink_ = out8_
                            d_14_steps_ = (d_14_steps_) + (1)
                            raise _dafny.Break("0")
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_19_closedGenerated_: _dafny.Seq
                        d_20_closedInside_: bool
                        d_21_closedCurrent_: _dafny.Seq
                        out9_: _dafny.Seq
                        out10_: bool
                        out11_: _dafny.Seq
                        out9_, out10_, out11_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_19_closedGenerated_ = out9_
                        d_20_closedInside_ = out10_
                        d_21_closedCurrent_ = out11_
                        generated = d_19_closedGenerated_
                        insideConstrainedOut = d_20_closedInside_
                        currentConstrainedOut = d_21_closedCurrent_
                        d_14_steps_ = (d_14_steps_) + (1)
                        raise _dafny.Break("0")
                    elif True:
                        d_22_constrainedPrompt_: _dafny.Seq
                        d_22_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_23_groups_: _dafny.Seq
                        d_23_groups_ = (d_11_sqlStyleGroups_) + (validTokenGroups)
                        d_24_next_: _dafny.Seq
                        out12_: _dafny.Seq
                        out12_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_22_constrainedPrompt_, currentConstrainedOut, d_23_groups_, _dafny.BigRational('5e0'), d_12_penaltyTokens_, _dafny.BigRational('6e0'), d_13_narrowThreshold_, eosToken)
                        d_24_next_ = out12_
                        d_14_steps_ = (d_14_steps_) + (1)
                        if ((d_24_next_) == (eosToken)) and ((d_14_steps_) < (maxSteps)):
                            out13_: _dafny.Seq
                            out13_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_22_constrainedPrompt_, currentConstrainedOut, d_23_groups_, _dafny.BigRational('7e0'), d_12_penaltyTokens_, _dafny.BigRational('1e1'), d_13_narrowThreshold_, eosToken)
                            d_24_next_ = out13_
                            d_14_steps_ = (d_14_steps_) + (1)
                        if (d_24_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_25_appendedGenerated_: _dafny.Seq
                            d_26_appendedInside_: bool
                            d_27_appendedCurrent_: _dafny.Seq
                            out14_: _dafny.Seq
                            out15_: bool
                            out16_: _dafny.Seq
                            out14_, out15_, out16_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_24_next_)
                            d_25_appendedGenerated_ = out14_
                            d_26_appendedInside_ = out15_
                            d_27_appendedCurrent_ = out16_
                            generated = d_25_appendedGenerated_
                            insideConstrainedOut = d_26_appendedInside_
                            currentConstrainedOut = d_27_appendedCurrent_
                            if ((d_24_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "FROM")))) or ((d_24_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "from")))):
                                d_9_seenFrom_ = True
                            elif ((d_24_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "WHERE")))) or ((d_24_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "where")))):
                                d_10_seenWhere_ = True
                            if ((d_14_steps_) < (maxSteps)) and ((parser).IsCompletePrefix(currentConstrainedOut)):
                                d_28_closedGenerated2_: _dafny.Seq
                                d_29_closedInside2_: bool
                                d_30_closedCurrent2_: _dafny.Seq
                                out17_: _dafny.Seq
                                out18_: bool
                                out19_: _dafny.Seq
                                out17_, out18_, out19_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_28_closedGenerated2_ = out17_
                                d_29_closedInside2_ = out18_
                                d_30_closedCurrent2_ = out19_
                                generated = d_28_closedGenerated2_
                                insideConstrainedOut = d_29_closedInside2_
                                currentConstrainedOut = d_30_closedCurrent2_
                                d_14_steps_ = (d_14_steps_) + (1)
                                raise _dafny.Break("0")
                    pass
            pass
        cost = d_14_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

