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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Return exactly SQL: <<query>>. Inside the delimiters output only SQL using the given schema. No prose, Markdown, comments, trailing semicolon, or unnecessary DISTINCT. Avoid aliases/AS/T1/T2 unless a self-join requires them. Prefer explicit joins through linking tables. For conditions requiring both values, use INTERSECT or GROUP BY/HAVING."))
        if (len(validTokenGroups)) > (0):
            d_1_guidance_ = (d_1_guidance_) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " Prefer schema names from the question context.")))
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
        d_7_seenJoinCount_: int
        out3_: int
        out3_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "JOIN")))
        d_7_seenJoinCount_ = out3_
        d_8_seenLowerJoinCount_: int
        out4_: int
        out4_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "join")))
        d_8_seenLowerJoinCount_ = out4_
        d_9_seenFrom_: bool
        d_9_seenFrom_ = ((d_5_seenFromCount_) > (0)) or ((d_6_seenLowerFromCount_) > (0))
        d_10_seenJoin_: bool
        d_10_seenJoin_ = ((d_7_seenJoinCount_) > (0)) or ((d_8_seenLowerJoinCount_) > (0))
        d_11_relationBoostBudget_: int
        d_11_relationBoostBudget_ = 0
        if d_9_seenFrom_:
            d_11_relationBoostBudget_ = 3
        d_12_sqlKeywordGroups_: _dafny.Seq
        d_12_sqlKeywordGroups_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SELECT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "select")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "DISTINCT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "distinct"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "FROM")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "from"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "WHERE")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "where")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "AND")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "and")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "OR")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "or"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "JOIN")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "join")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "ON")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "on"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "GROUP")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "group")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "BY")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "by")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "HAVING")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "having"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "ORDER")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "order")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "DESC")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "desc")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "ASC")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "asc")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "LIMIT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "limit"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "INTERSECT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "intersect")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "UNION")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "union")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "EXCEPT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "except"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "COUNT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "count")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SUM")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "sum")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "AVG")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "avg")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "MIN")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "min")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "MAX")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "max"))])])
        d_13_aliasPenaltyTokens_: _dafny.Seq
        d_13_aliasPenaltyTokens_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "AS")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "as")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "T1")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "T2")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "T3")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "t1")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "t2")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "t3")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ";")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))])
        d_14_narrowThreshold_: int
        d_14_narrowThreshold_ = 12
        d_15_localLimit_: int
        d_15_localLimit_ = maxSteps
        if (d_15_localLimit_) > (96):
            d_15_localLimit_ = 96
        d_16_steps_: int
        d_16_steps_ = 0
        with _dafny.label("0"):
            while (d_16_steps_) < (d_15_localLimit_):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if not(d_3_headerDone_):
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SQL: "))]))
                            d_3_headerDone_ = True
                            d_16_steps_ = (d_16_steps_) + (1)
                        elif not(d_4_spanStarted_):
                            d_17_openedGenerated_: _dafny.Seq
                            d_18_openedInside_: bool
                            d_19_openedCurrent_: _dafny.Seq
                            out5_: _dafny.Seq
                            out6_: bool
                            out7_: _dafny.Seq
                            out5_, out6_, out7_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_17_openedGenerated_ = out5_
                            d_18_openedInside_ = out6_
                            d_19_openedCurrent_ = out7_
                            generated = d_17_openedGenerated_
                            insideConstrainedOut = d_18_openedInside_
                            currentConstrainedOut = d_19_openedCurrent_
                            d_4_spanStarted_ = True
                            d_9_seenFrom_ = False
                            d_10_seenJoin_ = False
                            d_11_relationBoostBudget_ = 0
                            d_16_steps_ = (d_16_steps_) + (1)
                        elif True:
                            d_20_sink_: _dafny.Seq
                            out8_: _dafny.Seq
                            out8_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_20_sink_ = out8_
                            d_16_steps_ = (d_16_steps_) + (1)
                            raise _dafny.Break("0")
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_21_closedGenerated_: _dafny.Seq
                        d_22_closedInside_: bool
                        d_23_closedCurrent_: _dafny.Seq
                        out9_: _dafny.Seq
                        out10_: bool
                        out11_: _dafny.Seq
                        out9_, out10_, out11_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_21_closedGenerated_ = out9_
                        d_22_closedInside_ = out10_
                        d_23_closedCurrent_ = out11_
                        generated = d_21_closedGenerated_
                        insideConstrainedOut = d_22_closedInside_
                        currentConstrainedOut = d_23_closedCurrent_
                        d_16_steps_ = (d_16_steps_) + (1)
                        raise _dafny.Break("0")
                    elif True:
                        d_24_constrainedPrompt_: _dafny.Seq
                        d_24_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_25_next_: _dafny.Seq
                        d_25_next_ = eosToken
                        if not(d_9_seenFrom_):
                            out12_: _dafny.Seq
                            out12_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, d_24_constrainedPrompt_, currentConstrainedOut, d_12_sqlKeywordGroups_, _dafny.BigRational('5e0'), eosToken)
                            d_25_next_ = out12_
                        elif (d_11_relationBoostBudget_) > (0):
                            out13_: _dafny.Seq
                            out13_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_24_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('3e0'), d_13_aliasPenaltyTokens_, _dafny.BigRational('6e0'), d_14_narrowThreshold_, eosToken)
                            d_25_next_ = out13_
                        elif True:
                            d_26_gatedNext_: _dafny.Seq
                            d_27_wasConstrained_: bool
                            out14_: _dafny.Seq
                            out15_: bool
                            out14_, out15_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, d_24_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_26_gatedNext_ = out14_
                            d_27_wasConstrained_ = out15_
                            d_25_next_ = d_26_gatedNext_
                        d_16_steps_ = (d_16_steps_) + (1)
                        if (d_25_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_28_appendedGenerated_: _dafny.Seq
                            d_29_appendedInside_: bool
                            d_30_appendedCurrent_: _dafny.Seq
                            out16_: _dafny.Seq
                            out17_: bool
                            out18_: _dafny.Seq
                            out16_, out17_, out18_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_25_next_)
                            d_28_appendedGenerated_ = out16_
                            d_29_appendedInside_ = out17_
                            d_30_appendedCurrent_ = out18_
                            generated = d_28_appendedGenerated_
                            insideConstrainedOut = d_29_appendedInside_
                            currentConstrainedOut = d_30_appendedCurrent_
                            if (d_11_relationBoostBudget_) > (0):
                                d_11_relationBoostBudget_ = (d_11_relationBoostBudget_) - (1)
                            if ((d_25_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "FROM")))) or ((d_25_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "from")))):
                                d_9_seenFrom_ = True
                                d_11_relationBoostBudget_ = 4
                            elif ((d_25_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "JOIN")))) or ((d_25_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "join")))):
                                d_10_seenJoin_ = True
                                d_11_relationBoostBudget_ = 4
                            elif ((((((d_25_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "INTERSECT")))) or ((d_25_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "intersect"))))) or ((d_25_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "UNION"))))) or ((d_25_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "union"))))) or ((d_25_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "EXCEPT"))))) or ((d_25_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "except")))):
                                d_9_seenFrom_ = False
                                d_10_seenJoin_ = False
                                d_11_relationBoostBudget_ = 0
                            if ((d_16_steps_) < (d_15_localLimit_)) and ((parser).IsCompletePrefix(currentConstrainedOut)):
                                d_31_closedGenerated2_: _dafny.Seq
                                d_32_closedInside2_: bool
                                d_33_closedCurrent2_: _dafny.Seq
                                out19_: _dafny.Seq
                                out20_: bool
                                out21_: _dafny.Seq
                                out19_, out20_, out21_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_31_closedGenerated2_ = out19_
                                d_32_closedInside2_ = out20_
                                d_33_closedCurrent2_ = out21_
                                generated = d_31_closedGenerated2_
                                insideConstrainedOut = d_32_closedInside2_
                                currentConstrainedOut = d_33_closedCurrent2_
                                d_16_steps_ = (d_16_steps_) + (1)
                                raise _dafny.Break("0")
                    pass
            pass
        if (((d_16_steps_) < (maxSteps)) and (insideConstrainedOut)) and ((parser).IsCompletePrefix(currentConstrainedOut)):
            d_34_finalClosedGenerated_: _dafny.Seq
            d_35_finalClosedInside_: bool
            d_36_finalClosedCurrent_: _dafny.Seq
            out22_: _dafny.Seq
            out23_: bool
            out24_: _dafny.Seq
            out22_, out23_, out24_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
            d_34_finalClosedGenerated_ = out22_
            d_35_finalClosedInside_ = out23_
            d_36_finalClosedCurrent_ = out24_
            generated = d_34_finalClosedGenerated_
            insideConstrainedOut = d_35_finalClosedInside_
            currentConstrainedOut = d_36_finalClosedCurrent_
            d_16_steps_ = (d_16_steps_) + (1)
        cost = d_16_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

