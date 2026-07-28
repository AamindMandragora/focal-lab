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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Output exactly one SQL query in the format: SQL: <<SELECT ...>>. No explanation. No markdown. Use only table and column names from the provided schema. Begin the answer with SQL: << and end with >>.")))
        d_1_sqlKeywordGroups_: _dafny.Seq
        d_1_sqlKeywordGroups_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SELECT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "select"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "FROM")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "from"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "WHERE")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "where"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "JOIN")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "join")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "INNER")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "LEFT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "RIGHT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "OUTER")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "CROSS"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "ON")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "on"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "GROUP")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "group")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "BY")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "by"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "ORDER")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "order")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "HAVING")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "having"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "LIMIT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "limit"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "DISTINCT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "distinct"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "AND")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "and")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "OR")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "or")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "NOT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "not"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "COUNT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "count")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SUM")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "sum")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "AVG")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "avg")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "MAX")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "max")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "MIN")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "min"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "AS")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "as"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "IN")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "in")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "LIKE")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "like")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "BETWEEN")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "between"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "*")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "!=")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<=")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">=")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "(")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ")"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ",")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "."))])])
        d_2_allGroups_: _dafny.Seq
        d_2_allGroups_ = (d_1_sqlKeywordGroups_) + (validTokenGroups)
        d_3_narrowThreshold_: int
        d_3_narrowThreshold_ = 16
        d_4_steps_: int
        d_4_steps_ = 0
        d_5_prefixBudget_: int
        d_5_prefixBudget_ = 4
        if (not(insideConstrainedOut)) and (((d_4_steps_) + (d_5_prefixBudget_)) <= (maxSteps)):
            d_6_chunkOut_: _dafny.Seq
            d_7_stoppedOnOpen_: bool
            d_8_stoppedOnEos_: bool
            d_9_chunkUsed_: int
            out0_: _dafny.Seq
            out1_: bool
            out2_: bool
            out3_: int
            out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_5_prefixBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
            d_6_chunkOut_ = out0_
            d_7_stoppedOnOpen_ = out1_
            d_8_stoppedOnEos_ = out2_
            d_9_chunkUsed_ = out3_
            generated = d_6_chunkOut_
            d_4_steps_ = (d_4_steps_) + (d_9_chunkUsed_)
            if d_7_stoppedOnOpen_:
                d_10_eg_: _dafny.Seq
                d_11_ei_: bool
                d_12_ec_: _dafny.Seq
                out4_: _dafny.Seq
                out5_: bool
                out6_: _dafny.Seq
                out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                d_10_eg_ = out4_
                d_11_ei_ = out5_
                d_12_ec_ = out6_
                generated = d_10_eg_
                insideConstrainedOut = d_11_ei_
                currentConstrainedOut = d_12_ec_
        if (not(insideConstrainedOut)) and ((d_4_steps_) < (maxSteps)):
            d_13_og_: _dafny.Seq
            d_14_oi_: bool
            d_15_oc_: _dafny.Seq
            out7_: _dafny.Seq
            out8_: bool
            out9_: _dafny.Seq
            out7_, out8_, out9_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_13_og_ = out7_
            d_14_oi_ = out8_
            d_15_oc_ = out9_
            generated = d_13_og_
            insideConstrainedOut = d_14_oi_
            currentConstrainedOut = d_15_oc_
            d_4_steps_ = (d_4_steps_) + (1)
        with _dafny.label("0"):
            while (d_4_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        raise _dafny.Break("0")
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_16_cg_: _dafny.Seq
                        d_17_ci_: bool
                        d_18_cc_: _dafny.Seq
                        out10_: _dafny.Seq
                        out11_: bool
                        out12_: _dafny.Seq
                        out10_, out11_, out12_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_16_cg_ = out10_
                        d_17_ci_ = out11_
                        d_18_cc_ = out12_
                        generated = d_16_cg_
                        insideConstrainedOut = d_17_ci_
                        currentConstrainedOut = d_18_cc_
                        d_4_steps_ = (d_4_steps_) + (1)
                        raise _dafny.Break("0")
                    elif True:
                        d_19_constrainedPrompt_: _dafny.Seq
                        d_19_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_20_next_: _dafny.Seq
                        out13_: _dafny.Seq
                        out13_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_19_constrainedPrompt_, currentConstrainedOut, d_2_allGroups_, _dafny.BigRational('4e0'), d_3_narrowThreshold_, eosToken)
                        d_20_next_ = out13_
                        d_4_steps_ = (d_4_steps_) + (1)
                        if (d_20_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_21_ag_: _dafny.Seq
                            d_22_ai_: bool
                            d_23_ac_: _dafny.Seq
                            out14_: _dafny.Seq
                            out15_: bool
                            out16_: _dafny.Seq
                            out14_, out15_, out16_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_20_next_)
                            d_21_ag_ = out14_
                            d_22_ai_ = out15_
                            d_23_ac_ = out16_
                            generated = d_21_ag_
                            insideConstrainedOut = d_22_ai_
                            currentConstrainedOut = d_23_ac_
                    pass
            pass
        cost = d_4_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

