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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate exactly one SQL query. Output format must be: SQL: <<SELECT ...>> with no explanation, no markdown, no extra text. Use only the schema tables and columns provided."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_sqlKeywordGroups_: _dafny.Seq
        d_2_sqlKeywordGroups_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SELECT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "DISTINCT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "ALL"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "FROM"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "WHERE")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "AND")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "OR")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "NOT"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "JOIN")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "INNER")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "LEFT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "RIGHT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "OUTER")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "ON"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "GROUP")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "BY")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "HAVING"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "ORDER")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "ASC")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "DESC"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "LIMIT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "OFFSET"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "COUNT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SUM")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "AVG")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "MIN")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "MAX"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "AS")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "IN")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "BETWEEN")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "LIKE")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "IS")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "NULL")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "INTERSECT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "UNION")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "EXCEPT"))])])
        d_3_seenFrom_: bool
        d_3_seenFrom_ = False
        d_4_seenWhere_: bool
        d_4_seenWhere_ = False
        d_5_narrowThreshold_: int
        d_5_narrowThreshold_ = 15
        d_6_steps_: int
        d_6_steps_ = 0
        if ((d_6_steps_) < (maxSteps)) and (not(insideConstrainedOut)):
            d_7_chunkMax_: int
            d_7_chunkMax_ = 6
            if (d_7_chunkMax_) > ((maxSteps) - (d_6_steps_)):
                d_7_chunkMax_ = (maxSteps) - (d_6_steps_)
            if (d_7_chunkMax_) > (0):
                d_8_chunkGenerated_: _dafny.Seq
                d_9_stoppedOnOpen_: bool
                d_10_stoppedOnEos_: bool
                d_11_chunkSteps_: int
                out0_: _dafny.Seq
                out1_: bool
                out2_: bool
                out3_: int
                out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_7_chunkMax_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                d_8_chunkGenerated_ = out0_
                d_9_stoppedOnOpen_ = out1_
                d_10_stoppedOnEos_ = out2_
                d_11_chunkSteps_ = out3_
                generated = d_8_chunkGenerated_
                d_6_steps_ = (d_6_steps_) + (d_11_chunkSteps_)
                if d_10_stoppedOnEos_:
                    pass
                elif d_9_stoppedOnOpen_:
                    insideConstrainedOut = True
                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    d_3_seenFrom_ = False
                    d_4_seenWhere_ = False
        if ((d_6_steps_) < (maxSteps)) and (not(insideConstrainedOut)):
            d_12_openGenerated_: _dafny.Seq
            d_13_openInside_: bool
            d_14_openCurrent_: _dafny.Seq
            out4_: _dafny.Seq
            out5_: bool
            out6_: _dafny.Seq
            out4_, out5_, out6_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_12_openGenerated_ = out4_
            d_13_openInside_ = out5_
            d_14_openCurrent_ = out6_
            generated = d_12_openGenerated_
            insideConstrainedOut = d_13_openInside_
            currentConstrainedOut = d_14_openCurrent_
            d_6_steps_ = (d_6_steps_) + (1)
            d_3_seenFrom_ = False
            d_4_seenWhere_ = False
        with _dafny.label("0"):
            while (d_6_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        raise _dafny.Break("0")
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
                        d_6_steps_ = (d_6_steps_) + (1)
                        raise _dafny.Break("0")
                    elif True:
                        d_18_constrainedPrompt_: _dafny.Seq
                        d_18_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_19_next_: _dafny.Seq
                        d_19_next_ = eosToken
                        d_20_groups_: _dafny.Seq
                        d_20_groups_ = (d_2_sqlKeywordGroups_) + (validTokenGroups)
                        if not(d_3_seenFrom_):
                            out10_: _dafny.Seq
                            out10_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, d_18_constrainedPrompt_, currentConstrainedOut, d_20_groups_, _dafny.BigRational('5e0'), eosToken)
                            d_19_next_ = out10_
                        elif True:
                            out11_: _dafny.Seq
                            out11_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_18_constrainedPrompt_, currentConstrainedOut, d_20_groups_, _dafny.BigRational('4e0'), d_5_narrowThreshold_, eosToken)
                            d_19_next_ = out11_
                        d_6_steps_ = (d_6_steps_) + (1)
                        if (d_19_next_) == (eosToken):
                            if (parser).IsCompletePrefix(currentConstrainedOut):
                                if (d_6_steps_) < (maxSteps):
                                    d_21_closedGenerated_: _dafny.Seq
                                    d_22_closedInside_: bool
                                    d_23_closedCurrent_: _dafny.Seq
                                    out12_: _dafny.Seq
                                    out13_: bool
                                    out14_: _dafny.Seq
                                    out12_, out13_, out14_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_21_closedGenerated_ = out12_
                                    d_22_closedInside_ = out13_
                                    d_23_closedCurrent_ = out14_
                                    generated = d_21_closedGenerated_
                                    insideConstrainedOut = d_22_closedInside_
                                    currentConstrainedOut = d_23_closedCurrent_
                                    d_6_steps_ = (d_6_steps_) + (1)
                            raise _dafny.Break("0")
                        elif True:
                            d_24_appendedGenerated_: _dafny.Seq
                            d_25_appendedInside_: bool
                            d_26_appendedCurrent_: _dafny.Seq
                            out15_: _dafny.Seq
                            out16_: bool
                            out17_: _dafny.Seq
                            out15_, out16_, out17_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_19_next_)
                            d_24_appendedGenerated_ = out15_
                            d_25_appendedInside_ = out16_
                            d_26_appendedCurrent_ = out17_
                            generated = d_24_appendedGenerated_
                            insideConstrainedOut = d_25_appendedInside_
                            currentConstrainedOut = d_26_appendedCurrent_
                            if (d_19_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "FROM"))):
                                d_3_seenFrom_ = True
                            elif (d_19_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "WHERE"))):
                                d_4_seenWhere_ = True
                    pass
            pass
        cost = d_6_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

