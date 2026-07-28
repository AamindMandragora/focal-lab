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
        out0_: _dafny.Seq
        out1_: bool
        out2_: _dafny.Seq
        out3_: int
        out0_, out1_, out2_, out3_ = default__.AuthorBody(lm, parser, prompt, generatedPrefix, insideConstrained, currentConstrained, maxSteps, stepTokenBudget, validTokenGroups, eosToken)
        generated = out0_
        insideConstrainedOut = out1_
        currentConstrainedOut = out2_
        cost = out3_
        if ((maxSteps) > (0)) and ((cost) <= (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

    @staticmethod
    def AuthorBody(lm, parser, prompt, generatedPrefix, insideConstrained, currentConstrained, maxSteps, stepTokenBudget, validTokenGroups, eosToken):
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
        generated = generatedPrefix
        insideConstrainedOut = insideConstrained
        currentConstrainedOut = currentConstrained
        cost = 0
        d_1_guidance_: _dafny.Seq
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Output SQL: <<query>> where the query is a valid SQL SELECT statement using only the given schema tables and columns. Put your answer between << and >>. No explanation."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_sqlKeyGroup_: _dafny.Seq
        d_3_sqlKeyGroup_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SELECT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "FROM")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "WHERE")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "JOIN")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "ON")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "GROUP")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "ORDER")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "HAVING")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "LIMIT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "UNION")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "INTERSECT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "EXCEPT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "AS")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "BY")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "DISTINCT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "COUNT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SUM")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "MAX")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "MIN")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "AVG")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "INNER")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "LEFT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "RIGHT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "OUTER")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "CROSS")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "AND")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "OR")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "NOT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "IN")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "EXISTS")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "BETWEEN")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "LIKE")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "IS")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "NULL")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "ALL")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "ANY")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "CASE")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "WHEN")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "THEN")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "ELSE")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "END")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "INTO")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "VALUES")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SET")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "WITH")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " SELECT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " FROM")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " WHERE")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " JOIN")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " ON")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " GROUP")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " ORDER")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " HAVING")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " LIMIT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " UNION")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " INTERSECT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " EXCEPT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " AS")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " BY")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " DISTINCT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " COUNT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " SUM")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " MAX")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " MIN")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " AVG")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " INNER")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " LEFT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " RIGHT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " AND")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " OR")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " NOT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " IN")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " EXISTS")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " BETWEEN")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " LIKE")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " IS")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " NULL")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " CASE")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " WHEN")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " THEN")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " ELSE")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " END")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "select")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "from")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "where")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "join")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "on")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "group")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "order")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "having")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "limit")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "union")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "intersect")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "except")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "as")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "by")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "distinct")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "count")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "sum")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "max")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "min")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "avg")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "inner")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "left")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "right")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "and")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "or")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "not")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "in")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "exists")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "between")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "like")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "is")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "null"))])
        d_4_enhancedGroups_: _dafny.Seq
        d_4_enhancedGroups_ = (validTokenGroups) + (_dafny.SeqWithoutIsStrInference([d_3_sqlKeyGroup_]))
        d_5_preambleBudget_: int
        d_5_preambleBudget_ = 5
        d_6_preambleSteps_: int
        d_6_preambleSteps_ = 0
        with _dafny.label("0"):
            while (((d_2_steps_) < (maxSteps)) and ((d_6_preambleSteps_) < (d_5_preambleBudget_))) and (not(insideConstrainedOut)):
                with _dafny.c_label("0"):
                    d_7_next_: _dafny.Seq
                    out0_: _dafny.Seq
                    out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                    d_7_next_ = out0_
                    d_2_steps_ = (d_2_steps_) + (1)
                    d_6_preambleSteps_ = (d_6_preambleSteps_) + (1)
                    if (d_7_next_) == (eosToken):
                        raise _dafny.Break("0")
                    elif True:
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_7_next_]))
                        if (d_7_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                            insideConstrainedOut = True
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    pass
            pass
        if (not(insideConstrainedOut)) and ((d_2_steps_) < (maxSteps)):
            d_8_og_: _dafny.Seq
            d_9_oi_: bool
            d_10_oc_: _dafny.Seq
            out1_: _dafny.Seq
            out2_: bool
            out3_: _dafny.Seq
            out1_, out2_, out3_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_8_og_ = out1_
            d_9_oi_ = out2_
            d_10_oc_ = out3_
            generated = d_8_og_
            insideConstrainedOut = d_9_oi_
            currentConstrainedOut = d_10_oc_
            d_2_steps_ = (d_2_steps_) + (1)
        with _dafny.label("1"):
            while ((d_2_steps_) < (maxSteps)) and (insideConstrainedOut):
                with _dafny.c_label("1"):
                    if (len(currentConstrainedOut)) >= (80):
                        d_11_closeBudget2_: int
                        d_11_closeBudget2_ = (maxSteps) - (d_2_steps_)
                        d_12_cg2_: _dafny.Seq
                        d_13_ci2_: bool
                        d_14_cc2_: _dafny.Seq
                        out4_: _dafny.Seq
                        out5_: bool
                        out6_: _dafny.Seq
                        out4_, out5_, out6_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_11_closeBudget2_)
                        d_12_cg2_ = out4_
                        d_13_ci2_ = out5_
                        d_14_cc2_ = out6_
                        generated = d_12_cg2_
                        insideConstrainedOut = d_13_ci2_
                        currentConstrainedOut = d_14_cc2_
                        d_2_steps_ = maxSteps
                        raise _dafny.Break("1")
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
                        d_2_steps_ = (d_2_steps_) + (1)
                        raise _dafny.Break("1")
                    elif True:
                        d_18_constrainedPrompt_: _dafny.Seq
                        d_18_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_19_next_: _dafny.Seq
                        out10_: _dafny.Seq
                        out10_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, d_18_constrainedPrompt_, currentConstrainedOut, d_4_enhancedGroups_, _dafny.BigRational('6e0'), eosToken)
                        d_19_next_ = out10_
                        d_2_steps_ = (d_2_steps_) + (1)
                        if (d_19_next_) == (eosToken):
                            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_2_steps_) < (maxSteps)):
                                d_20_closedGenerated_: _dafny.Seq
                                d_21_closedInside_: bool
                                d_22_closedCurrent_: _dafny.Seq
                                out11_: _dafny.Seq
                                out12_: bool
                                out13_: _dafny.Seq
                                out11_, out12_, out13_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_20_closedGenerated_ = out11_
                                d_21_closedInside_ = out12_
                                d_22_closedCurrent_ = out13_
                                generated = d_20_closedGenerated_
                                insideConstrainedOut = d_21_closedInside_
                                currentConstrainedOut = d_22_closedCurrent_
                                d_2_steps_ = (d_2_steps_) + (1)
                            raise _dafny.Break("1")
                        elif True:
                            d_23_appendedGenerated_: _dafny.Seq
                            d_24_appendedInside_: bool
                            d_25_appendedCurrent_: _dafny.Seq
                            out14_: _dafny.Seq
                            out15_: bool
                            out16_: _dafny.Seq
                            out14_, out15_, out16_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_19_next_)
                            d_23_appendedGenerated_ = out14_
                            d_24_appendedInside_ = out15_
                            d_25_appendedCurrent_ = out16_
                            generated = d_23_appendedGenerated_
                            insideConstrainedOut = d_24_appendedInside_
                            currentConstrainedOut = d_25_appendedCurrent_
                    pass
            pass
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_26_closeBudget_: int
            d_26_closeBudget_ = (maxSteps) - (d_2_steps_)
            d_27_cg_: _dafny.Seq
            d_28_ci_: bool
            d_29_cc_: _dafny.Seq
            out17_: _dafny.Seq
            out18_: bool
            out19_: _dafny.Seq
            out17_, out18_, out19_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_26_closeBudget_)
            d_27_cg_ = out17_
            d_28_ci_ = out18_
            d_29_cc_ = out19_
            generated = d_27_cg_
            insideConstrainedOut = d_28_ci_
            currentConstrainedOut = d_29_cc_
            d_2_steps_ = maxSteps
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

