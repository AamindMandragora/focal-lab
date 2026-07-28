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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Return exactly one answer in the form SQL: <<query>> and nothing else. Generate a single valid SQLite query for the given Spider schema and question. Use only schema table and column names, and put every predicate on the table where that column is actually defined. Prefer exact Spider style: requested columns only, no extra DISTINCT, no trailing semicolon, no markdown, lowercase/uppercase is irrelevant but table and column choices must be exact. Avoid aliases such as T1/T2 and avoid AS unless a true self-join requires aliases; for ordinary joins use original table names in SELECT, FROM, JOIN, ON, WHERE, GROUP BY, ORDER BY. Join through foreign keys when output columns and filter columns are in different tables. Preserve every condition, comparison, aggregation, grouping, ordering, superlative, and LIMIT from the question. For at most, no more than, not more than, or do not have more than, use a row-level predicate such as <= rather than EXCEPT. Use EXCEPT only for true set difference questions, and use INTERSECT only when the same selected entities must satisfy two separate row/year/category conditions.")))
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
            d_5_warm_: int
            d_5_warm_ = 0
            d_6_hitEos_: bool
            d_6_hitEos_ = False
            while (((((d_4_steps_) + (1)) < (maxSteps)) and (insideConstrainedOut)) and ((d_5_warm_) < (12))) and (not((parser).IsCompletePrefix(currentConstrainedOut))):
                d_7_stableWarm_: _dafny.Seq
                d_7_stableWarm_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                d_8_warmPrompt_: _dafny.Seq
                d_8_warmPrompt_ = (prompt) + (d_7_stableWarm_)
                d_9_nextWarm_: _dafny.Seq
                out3_: _dafny.Seq
                out3_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_8_warmPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('6e0'), _dafny.SeqWithoutIsStrInference([eosToken, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ";")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " ;")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "AS")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " as")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " T1")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " T2")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " T3")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "T1")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "T2")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "T3")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "t1")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "t2")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "alias"))]), _dafny.BigRational('14e0'), 64, eosToken)
                d_9_nextWarm_ = out3_
                d_4_steps_ = (d_4_steps_) + (1)
                d_5_warm_ = (d_5_warm_) + (1)
                if (d_9_nextWarm_) == (eosToken):
                    pass
                elif True:
                    d_10_appendedWarmGenerated_: _dafny.Seq
                    d_11_appendedWarmInside_: bool
                    d_12_appendedWarmCurrent_: _dafny.Seq
                    out4_: _dafny.Seq
                    out5_: bool
                    out6_: _dafny.Seq
                    out4_, out5_, out6_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_9_nextWarm_)
                    d_10_appendedWarmGenerated_ = out4_
                    d_11_appendedWarmInside_ = out5_
                    d_12_appendedWarmCurrent_ = out6_
                    generated = d_10_appendedWarmGenerated_
                    insideConstrainedOut = d_11_appendedWarmInside_
                    currentConstrainedOut = d_12_appendedWarmCurrent_
            if ((((d_4_steps_) + (1)) < (maxSteps)) and (insideConstrainedOut)) and (not((parser).IsCompletePrefix(currentConstrainedOut))):
                d_13_firstCap_: int
                d_13_firstCap_ = ((maxSteps) - (d_4_steps_)) - (1)
                if (d_13_firstCap_) > (280):
                    d_13_firstCap_ = 280
                d_14_stablePrefix_: _dafny.Seq
                d_14_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                d_15_firstPrompt_: _dafny.Seq
                d_15_firstPrompt_ = (prompt) + (d_14_stablePrefix_)
                d_16_symbolGenerated_: _dafny.Seq
                d_17_symbolOut_: _dafny.Seq
                d_18_symbolHitEos_: bool
                d_19_used_: int
                out7_: _dafny.Seq
                out8_: _dafny.Seq
                out9_: bool
                out10_: int
                out7_, out8_, out9_, out10_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_15_firstPrompt_, generated, currentConstrainedOut, d_13_firstCap_, eosToken)
                d_16_symbolGenerated_ = out7_
                d_17_symbolOut_ = out8_
                d_18_symbolHitEos_ = out9_
                d_19_used_ = out10_
                generated = d_16_symbolGenerated_
                currentConstrainedOut = d_17_symbolOut_
                insideConstrainedOut = True
                d_6_hitEos_ = d_18_symbolHitEos_
                d_4_steps_ = (d_4_steps_) + (d_19_used_)
            while ((((((d_4_steps_) + (1)) < (maxSteps)) and (insideConstrainedOut)) and ((len(currentConstrainedOut)) < (380))) and (not(d_6_hitEos_))) and (not((parser).IsCompletePrefix(currentConstrainedOut))):
                d_20_stableTail_: _dafny.Seq
                d_20_stableTail_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                d_21_tailPrompt_: _dafny.Seq
                d_21_tailPrompt_ = (prompt) + (d_20_stableTail_)
                d_22_nextTail_: _dafny.Seq
                out11_: _dafny.Seq
                out11_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_21_tailPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('5e0'), _dafny.SeqWithoutIsStrInference([eosToken, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ";")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " ;")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "AS")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " as")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " T1")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " T2")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " T3")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "T1")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "T2")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "T3")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "t1")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "t2"))]), _dafny.BigRational('12e0'), 32, eosToken)
                d_22_nextTail_ = out11_
                d_4_steps_ = (d_4_steps_) + (1)
                if (d_22_nextTail_) == (eosToken):
                    d_6_hitEos_ = True
                elif True:
                    d_23_appendedTailGenerated_: _dafny.Seq
                    d_24_appendedTailInside_: bool
                    d_25_appendedTailCurrent_: _dafny.Seq
                    out12_: _dafny.Seq
                    out13_: bool
                    out14_: _dafny.Seq
                    out12_, out13_, out14_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_22_nextTail_)
                    d_23_appendedTailGenerated_ = out12_
                    d_24_appendedTailInside_ = out13_
                    d_25_appendedTailCurrent_ = out14_
                    generated = d_23_appendedTailGenerated_
                    insideConstrainedOut = d_24_appendedTailInside_
                    currentConstrainedOut = d_25_appendedTailCurrent_
            if ((insideConstrainedOut) and ((parser).IsCompletePrefix(currentConstrainedOut))) and ((d_4_steps_) < (maxSteps)):
                d_26_closedGenerated_: _dafny.Seq
                d_27_closedInside_: bool
                d_28_closedCurrent_: _dafny.Seq
                out15_: _dafny.Seq
                out16_: bool
                out17_: _dafny.Seq
                out15_, out16_, out17_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_26_closedGenerated_ = out15_
                d_27_closedInside_ = out16_
                d_28_closedCurrent_ = out17_
                generated = d_26_closedGenerated_
                insideConstrainedOut = d_27_closedInside_
                currentConstrainedOut = d_28_closedCurrent_
                d_4_steps_ = (d_4_steps_) + (1)
            cost = d_4_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

