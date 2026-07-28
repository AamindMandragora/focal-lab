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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate exactly one valid SQL query answering the question. Output format must be exactly: SQL: <<SELECT ...>> with the complete SQL query inside the << >> delimiters. Use only the tables and columns from the provided schema. For questions requiring multiple tables, use JOIN. For aggregation, use GROUP BY, HAVING. For ordering, use ORDER BY. No explanation, no markdown, no extra text.")))
        d_1_sqlKeywordGroups_: _dafny.Seq
        d_1_sqlKeywordGroups_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SELECT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "select")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "DISTINCT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "distinct"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "FROM")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "from"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "WHERE")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "where")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "AND")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "and")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "OR")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "or"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "JOIN")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "join")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "LEFT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "left")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "INNER")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "inner")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "RIGHT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "right")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "OUTER")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "outer"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "ON")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "on"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "GROUP")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "group")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "BY")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "by")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "ORDER")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "order"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "HAVING")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "having"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "LIMIT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "limit"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "COUNT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "count")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SUM")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "sum")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "AVG")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "avg")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "MAX")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "max")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "MIN")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "min"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "AS")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "as"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "IN")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "in")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "NOT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "not")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "LIKE")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "like")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "BETWEEN")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "between")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "EXISTS")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "exists"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "INTERSECT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "intersect")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "UNION")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "union")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "EXCEPT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "except"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<=")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">=")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "!=")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<>"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "(")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ")")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ",")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "*")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "."))])])
        d_2_steps_: int
        d_2_steps_ = 0
        if (not(insideConstrainedOut)) and ((d_2_steps_) < (maxSteps)):
            d_3_chunkBudget_: int
            d_3_chunkBudget_ = 5
            if (d_3_chunkBudget_) > ((maxSteps) - (d_2_steps_)):
                d_3_chunkBudget_ = (maxSteps) - (d_2_steps_)
            if (d_3_chunkBudget_) > (0):
                d_4_chunkGenerated_: _dafny.Seq
                d_5_stoppedOnOpen_: bool
                d_6_stoppedOnEos_: bool
                d_7_chunkSteps_: int
                out0_: _dafny.Seq
                out1_: bool
                out2_: bool
                out3_: int
                out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_3_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                d_4_chunkGenerated_ = out0_
                d_5_stoppedOnOpen_ = out1_
                d_6_stoppedOnEos_ = out2_
                d_7_chunkSteps_ = out3_
                generated = d_4_chunkGenerated_
                d_2_steps_ = (d_2_steps_) + (d_7_chunkSteps_)
                if d_6_stoppedOnEos_:
                    cost = d_2_steps_
                    return generated, insideConstrainedOut, currentConstrainedOut, cost
                if d_5_stoppedOnOpen_:
                    d_8_enteredGenerated_: _dafny.Seq
                    d_9_enteredInside_: bool
                    d_10_enteredCurrent_: _dafny.Seq
                    out4_: _dafny.Seq
                    out5_: bool
                    out6_: _dafny.Seq
                    out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                    d_8_enteredGenerated_ = out4_
                    d_9_enteredInside_ = out5_
                    d_10_enteredCurrent_ = out6_
                    generated = d_8_enteredGenerated_
                    insideConstrainedOut = d_9_enteredInside_
                    currentConstrainedOut = d_10_enteredCurrent_
        if (not(insideConstrainedOut)) and ((d_2_steps_) < (maxSteps)):
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
        d_14_consecutiveEos_: int
        d_14_consecutiveEos_ = 0
        with _dafny.label("0"):
            while (d_2_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        raise _dafny.Break("0")
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_15_closedGenerated_: _dafny.Seq
                        d_16_closedInside_: bool
                        d_17_closedCurrent_: _dafny.Seq
                        out10_: _dafny.Seq
                        out11_: bool
                        out12_: _dafny.Seq
                        out10_, out11_, out12_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_15_closedGenerated_ = out10_
                        d_16_closedInside_ = out11_
                        d_17_closedCurrent_ = out12_
                        generated = d_15_closedGenerated_
                        insideConstrainedOut = d_16_closedInside_
                        currentConstrainedOut = d_17_closedCurrent_
                        d_2_steps_ = (d_2_steps_) + (1)
                        raise _dafny.Break("0")
                    elif True:
                        d_18_constrainedPrompt_: _dafny.Seq
                        d_18_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_19_validCount_: int
                        out13_: int
                        out13_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                        d_19_validCount_ = out13_
                        d_20_next_: _dafny.Seq
                        d_20_next_ = eosToken
                        if (d_14_consecutiveEos_) >= (2):
                            d_21_allGroups_: _dafny.Seq
                            d_21_allGroups_ = (d_1_sqlKeywordGroups_) + (validTokenGroups)
                            out14_: _dafny.Seq
                            out14_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, d_18_constrainedPrompt_, currentConstrainedOut, d_21_allGroups_, _dafny.BigRational('1e1'), eosToken)
                            d_20_next_ = out14_
                            d_2_steps_ = (d_2_steps_) + (1)
                            if (d_20_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_14_consecutiveEos_ = 0
                                d_22_appendedGenerated_: _dafny.Seq
                                d_23_appendedInside_: bool
                                d_24_appendedCurrent_: _dafny.Seq
                                out15_: _dafny.Seq
                                out16_: bool
                                out17_: _dafny.Seq
                                out15_, out16_, out17_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_20_next_)
                                d_22_appendedGenerated_ = out15_
                                d_23_appendedInside_ = out16_
                                d_24_appendedCurrent_ = out17_
                                generated = d_22_appendedGenerated_
                                insideConstrainedOut = d_23_appendedInside_
                                currentConstrainedOut = d_24_appendedCurrent_
                        elif (len(currentConstrainedOut)) >= (80):
                            out18_: _dafny.Seq
                            out18_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_18_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('2e0'), eosToken)
                            d_20_next_ = out18_
                            d_2_steps_ = (d_2_steps_) + (1)
                            if (d_20_next_) == (eosToken):
                                d_14_consecutiveEos_ = (d_14_consecutiveEos_) + (1)
                            elif True:
                                d_14_consecutiveEos_ = 0
                                d_25_appendedGenerated_: _dafny.Seq
                                d_26_appendedInside_: bool
                                d_27_appendedCurrent_: _dafny.Seq
                                out19_: _dafny.Seq
                                out20_: bool
                                out21_: _dafny.Seq
                                out19_, out20_, out21_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_20_next_)
                                d_25_appendedGenerated_ = out19_
                                d_26_appendedInside_ = out20_
                                d_27_appendedCurrent_ = out21_
                                generated = d_25_appendedGenerated_
                                insideConstrainedOut = d_26_appendedInside_
                                currentConstrainedOut = d_27_appendedCurrent_
                        elif (d_19_validCount_) <= (5):
                            d_28_allGroups_: _dafny.Seq
                            d_28_allGroups_ = (d_1_sqlKeywordGroups_) + (validTokenGroups)
                            out22_: _dafny.Seq
                            out22_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, d_18_constrainedPrompt_, currentConstrainedOut, d_28_allGroups_, _dafny.BigRational('6e0'), eosToken)
                            d_20_next_ = out22_
                            d_2_steps_ = (d_2_steps_) + (1)
                            if (d_20_next_) == (eosToken):
                                d_14_consecutiveEos_ = (d_14_consecutiveEos_) + (1)
                            elif True:
                                d_14_consecutiveEos_ = 0
                                d_29_appendedGenerated_: _dafny.Seq
                                d_30_appendedInside_: bool
                                d_31_appendedCurrent_: _dafny.Seq
                                out23_: _dafny.Seq
                                out24_: bool
                                out25_: _dafny.Seq
                                out23_, out24_, out25_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_20_next_)
                                d_29_appendedGenerated_ = out23_
                                d_30_appendedInside_ = out24_
                                d_31_appendedCurrent_ = out25_
                                generated = d_29_appendedGenerated_
                                insideConstrainedOut = d_30_appendedInside_
                                currentConstrainedOut = d_31_appendedCurrent_
                        elif True:
                            d_32_gatedNext_: _dafny.Seq
                            d_33_wasConstrained_: bool
                            out26_: _dafny.Seq
                            out27_: bool
                            out26_, out27_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, d_18_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_32_gatedNext_ = out26_
                            d_33_wasConstrained_ = out27_
                            d_20_next_ = d_32_gatedNext_
                            d_2_steps_ = (d_2_steps_) + (1)
                            if (d_20_next_) == (eosToken):
                                d_14_consecutiveEos_ = (d_14_consecutiveEos_) + (1)
                            elif True:
                                d_14_consecutiveEos_ = 0
                                d_34_appendedGenerated_: _dafny.Seq
                                d_35_appendedInside_: bool
                                d_36_appendedCurrent_: _dafny.Seq
                                out28_: _dafny.Seq
                                out29_: bool
                                out30_: _dafny.Seq
                                out28_, out29_, out30_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_20_next_)
                                d_34_appendedGenerated_ = out28_
                                d_35_appendedInside_ = out29_
                                d_36_appendedCurrent_ = out30_
                                generated = d_34_appendedGenerated_
                                insideConstrainedOut = d_35_appendedInside_
                                currentConstrainedOut = d_36_appendedCurrent_
                    pass
            pass
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

