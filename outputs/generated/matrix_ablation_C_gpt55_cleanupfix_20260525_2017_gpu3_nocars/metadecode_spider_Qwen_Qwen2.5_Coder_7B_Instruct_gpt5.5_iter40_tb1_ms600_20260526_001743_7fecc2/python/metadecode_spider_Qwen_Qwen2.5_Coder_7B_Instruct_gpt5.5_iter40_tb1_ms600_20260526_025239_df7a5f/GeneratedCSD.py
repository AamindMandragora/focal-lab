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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Return exactly one answer in the form SQL: <<query>> and nothing else. Before emitting, silently identify the requested output columns, all needed tables, foreign-key joins, filters, grouping, aggregation, ordering, and limits. Generate one valid SQLite query using only the provided Spider schema. Prefer canonical Spider SQL: select only requested columns, preserve every condition, no markdown, no trailing semicolon. When selected columns or filter columns come from related tables, use explicit JOIN ... ON foreign keys and qualify columns as table.column; do not omit the table that owns a filtered or ordered column. Avoid aliases unless a self-join or ambiguity truly requires them; otherwise use original table names. Do not add DISTINCT unless the question explicitly asks for distinct or different values. Use EXCEPT only for true absent-set questions; use INTERSECT when the same selected entities must satisfy two separate year/category/row conditions. For row-level wording such as at most, no more than, or do not have more than, use a direct predicate such as <=. For more than lowest, less than highest, above average, or below average, use the needed join plus a scalar min, max, or avg subquery.")))
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
            if ((d_4_steps_) + (1)) < (maxSteps):
                d_5_firstCap_: int
                d_5_firstCap_ = ((maxSteps) - (d_4_steps_)) - (1)
                if (d_5_firstCap_) > (260):
                    d_5_firstCap_ = 260
                d_6_firstPrompt_: _dafny.Seq
                d_6_firstPrompt_ = (prompt) + (generated)
                d_7_symbolGenerated_: _dafny.Seq
                d_8_symbolOut_: _dafny.Seq
                d_9_symbolHitEos_: bool
                d_10_used_: int
                out3_: _dafny.Seq
                out4_: _dafny.Seq
                out5_: bool
                out6_: int
                out3_, out4_, out5_, out6_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_6_firstPrompt_, generated, currentConstrainedOut, d_5_firstCap_, eosToken)
                d_7_symbolGenerated_ = out3_
                d_8_symbolOut_ = out4_
                d_9_symbolHitEos_ = out5_
                d_10_used_ = out6_
                generated = d_7_symbolGenerated_
                currentConstrainedOut = d_8_symbolOut_
                insideConstrainedOut = True
                d_4_steps_ = (d_4_steps_) + (d_10_used_)
                while (((((d_4_steps_) + (1)) < (maxSteps)) and (insideConstrainedOut)) and ((len(currentConstrainedOut)) < (420))) and (not((parser).IsCompletePrefix(currentConstrainedOut))):
                    d_11_stablePrefix_: _dafny.Seq
                    d_11_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                    d_12_constrainedPrompt_: _dafny.Seq
                    d_12_constrainedPrompt_ = (prompt) + (d_11_stablePrefix_)
                    d_13_next_: _dafny.Seq
                    out7_: _dafny.Seq
                    out7_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_12_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('6e0'), _dafny.SeqWithoutIsStrInference([eosToken, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ";")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " ;")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SQL:")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "T1")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "T2")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "T3"))]), _dafny.BigRational('1e1'), 18, eosToken)
                    d_13_next_ = out7_
                    d_4_steps_ = (d_4_steps_) + (1)
                    if (d_13_next_) == (eosToken):
                        pass
                    elif True:
                        d_14_appendedGenerated_: _dafny.Seq
                        d_15_appendedInside_: bool
                        d_16_appendedCurrent_: _dafny.Seq
                        out8_: _dafny.Seq
                        out9_: bool
                        out10_: _dafny.Seq
                        out8_, out9_, out10_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_13_next_)
                        d_14_appendedGenerated_ = out8_
                        d_15_appendedInside_ = out9_
                        d_16_appendedCurrent_ = out10_
                        generated = d_14_appendedGenerated_
                        insideConstrainedOut = d_15_appendedInside_
                        currentConstrainedOut = d_16_appendedCurrent_
            if ((insideConstrainedOut) and ((parser).IsCompletePrefix(currentConstrainedOut))) and ((d_4_steps_) < (maxSteps)):
                d_17_closedGenerated_: _dafny.Seq
                d_18_closedInside_: bool
                d_19_closedCurrent_: _dafny.Seq
                out11_: _dafny.Seq
                out12_: bool
                out13_: _dafny.Seq
                out11_, out12_, out13_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_17_closedGenerated_ = out11_
                d_18_closedInside_ = out12_
                d_19_closedCurrent_ = out13_
                generated = d_17_closedGenerated_
                insideConstrainedOut = d_18_closedInside_
                currentConstrainedOut = d_19_closedCurrent_
                d_4_steps_ = (d_4_steps_) + (1)
            cost = d_4_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

