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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Return exactly one answer in the form SQL: <<query>> and nothing else. Generate one valid SQLite query for the given Spider schema and question. Use only provided schema table and column names. Prefer canonical Spider SQL: select only the requested columns, preserve every condition, aggregation, grouping, ordering, distinct requirement, set operation, and limit. Use the shortest valid join path through real foreign-key relationships; do not add extra bridge tables when a direct table/column relationship answers the question. Prefer unaliased table.column references; avoid T1/T2 aliases unless a self-join is truly necessary. For row-level phrases such as not more than, no more than, at most, or do not have more than, use WHERE <= or the corresponding predicate. For true absence questions such as not used by any, have no, without any related row, or entities not in another set, prefer EXCEPT or NOT IN only when it represents set difference. Use INTERSECT when the same selected entities must satisfy two separate category/year/attribute conditions. For entity superlatives, use ORDER BY with ASC or DESC and LIMIT 1; use MIN or MAX mainly when the requested output is the value itself. For counts use count(*), for totals use sum, and for averages use avg. Do not output markdown, explanations, comments, or a trailing semicolon before the closing delimiter.")))
        if (maxSteps) == (0):
            cost = 0
        elif (maxSteps) == (1):
            insideConstrainedOut = False
            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
            generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SQL:"))]))
            cost = 1
        elif (maxSteps) == (2):
            insideConstrainedOut = False
            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
            generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SQL:")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " "))]))
            cost = 2
        elif True:
            insideConstrainedOut = False
            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
            generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SQL:")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " "))]))
            d_1_openedGenerated_: _dafny.Seq
            d_2_openedInside_: bool
            d_3_openedCurrent_: _dafny.Seq
            out0_: _dafny.Seq
            out1_: bool
            out2_: _dafny.Seq
            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_1_openedGenerated_ = out0_
            d_2_openedInside_ = out1_
            d_3_openedCurrent_ = out2_
            generated = d_1_openedGenerated_
            insideConstrainedOut = d_2_openedInside_
            currentConstrainedOut = d_3_openedCurrent_
            d_4_steps_: int
            d_4_steps_ = 3
            d_5_hitEos_: bool
            d_5_hitEos_ = False
            if ((d_4_steps_) + (1)) < (maxSteps):
                d_6_firstCap_: int
                d_6_firstCap_ = ((maxSteps) - (d_4_steps_)) - (1)
                if (d_6_firstCap_) > (320):
                    d_6_firstCap_ = 320
                d_7_firstPrompt_: _dafny.Seq
                d_7_firstPrompt_ = (prompt) + (generated)
                d_8_symbolGenerated_: _dafny.Seq
                d_9_symbolOut_: _dafny.Seq
                d_10_symbolHitEos_: bool
                d_11_used_: int
                out3_: _dafny.Seq
                out4_: _dafny.Seq
                out5_: bool
                out6_: int
                out3_, out4_, out5_, out6_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_7_firstPrompt_, generated, currentConstrainedOut, d_6_firstCap_, eosToken)
                d_8_symbolGenerated_ = out3_
                d_9_symbolOut_ = out4_
                d_10_symbolHitEos_ = out5_
                d_11_used_ = out6_
                generated = d_8_symbolGenerated_
                currentConstrainedOut = d_9_symbolOut_
                insideConstrainedOut = True
                d_5_hitEos_ = d_10_symbolHitEos_
                d_4_steps_ = (d_4_steps_) + (d_11_used_)
                while ((((((d_4_steps_) + (1)) < (maxSteps)) and (insideConstrainedOut)) and ((len(currentConstrainedOut)) < (420))) and (not(d_5_hitEos_))) and (not((parser).IsCompletePrefix(currentConstrainedOut))):
                    d_12_stablePrefix_: _dafny.Seq
                    d_12_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                    d_13_constrainedPrompt_: _dafny.Seq
                    d_13_constrainedPrompt_ = (prompt) + (d_12_stablePrefix_)
                    d_14_next_: _dafny.Seq
                    out7_: _dafny.Seq
                    out7_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_13_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('5e0'), _dafny.SeqWithoutIsStrInference([eosToken, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ";")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "T1")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "T2")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "T3")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "T4")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "T5")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "AS"))]), _dafny.BigRational('1e1'), 18, eosToken)
                    d_14_next_ = out7_
                    d_4_steps_ = (d_4_steps_) + (1)
                    if (d_14_next_) == (eosToken):
                        d_5_hitEos_ = True
                    elif True:
                        d_15_appendedGenerated_: _dafny.Seq
                        d_16_appendedInside_: bool
                        d_17_appendedCurrent_: _dafny.Seq
                        out8_: _dafny.Seq
                        out9_: bool
                        out10_: _dafny.Seq
                        out8_, out9_, out10_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_14_next_)
                        d_15_appendedGenerated_ = out8_
                        d_16_appendedInside_ = out9_
                        d_17_appendedCurrent_ = out10_
                        generated = d_15_appendedGenerated_
                        insideConstrainedOut = d_16_appendedInside_
                        currentConstrainedOut = d_17_appendedCurrent_
            if ((insideConstrainedOut) and ((parser).IsCompletePrefix(currentConstrainedOut))) and ((d_4_steps_) < (maxSteps)):
                d_18_closedGenerated_: _dafny.Seq
                d_19_closedInside_: bool
                d_20_closedCurrent_: _dafny.Seq
                out11_: _dafny.Seq
                out12_: bool
                out13_: _dafny.Seq
                out11_, out12_, out13_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_18_closedGenerated_ = out11_
                d_19_closedInside_ = out12_
                d_20_closedCurrent_ = out13_
                generated = d_18_closedGenerated_
                insideConstrainedOut = d_19_closedInside_
                currentConstrainedOut = d_20_closedCurrent_
                d_4_steps_ = (d_4_steps_) + (1)
            elif (insideConstrainedOut) and ((d_4_steps_) < (maxSteps)):
                generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]))
                insideConstrainedOut = False
                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                d_4_steps_ = (d_4_steps_) + (1)
            cost = d_4_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

