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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Return exactly one answer as SQL: <<query>> and nothing else. Generate valid SQLite for the given Spider schema and question. Privately plan the select list, tables, foreign-key joins, filters, grouping, ordering, and limits. Use only schema names and values from the question. Prefer table.column names in joins, association tables when needed, count(*) for how many, and avg/min/max/sum exactly when asked. Preserve AND/OR conditions. Avoid extra DISTINCT, aliases like T1/T2 unless necessary, markdown, comments, and trailing semicolon.")))
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
                if (d_6_firstCap_) > (220):
                    d_6_firstCap_ = 220
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
                while ((((((d_4_steps_) + (1)) < (maxSteps)) and (insideConstrainedOut)) and ((len(currentConstrainedOut)) < (280))) and (not(d_5_hitEos_))) and (not((parser).IsCompletePrefix(currentConstrainedOut))):
                    d_12_stablePrefix_: _dafny.Seq
                    d_12_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                    d_13_constrainedPrompt_: _dafny.Seq
                    d_13_constrainedPrompt_ = (prompt) + (d_12_stablePrefix_)
                    d_14_next_: _dafny.Seq
                    out7_: _dafny.Seq
                    out7_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_13_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('5e0'), _dafny.SeqWithoutIsStrInference([eosToken, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ";")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " ;")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "T1")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "T2")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "T3")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "t1")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "t2"))]), _dafny.BigRational('12e0'), 16, eosToken)
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
                d_18_recover_: int
                d_18_recover_ = 0
                while ((((((d_4_steps_) + (1)) < (maxSteps)) and (insideConstrainedOut)) and ((d_18_recover_) < (36))) and ((len(currentConstrainedOut)) < (320))) and (not((parser).IsCompletePrefix(currentConstrainedOut))):
                    d_19_stablePrefix2_: _dafny.Seq
                    d_19_stablePrefix2_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                    d_20_constrainedPrompt2_: _dafny.Seq
                    d_20_constrainedPrompt2_ = (prompt) + (d_19_stablePrefix2_)
                    d_21_next2_: _dafny.Seq
                    out11_: _dafny.Seq
                    out11_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_20_constrainedPrompt2_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('6e0'), _dafny.SeqWithoutIsStrInference([eosToken, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ";")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " ;")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "T1")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "T2")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "T3")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "t1")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "t2"))]), _dafny.BigRational('14e0'), 20, eosToken)
                    d_21_next2_ = out11_
                    d_4_steps_ = (d_4_steps_) + (1)
                    d_18_recover_ = (d_18_recover_) + (1)
                    if (d_21_next2_) == (eosToken):
                        pass
                    elif True:
                        d_22_appendedGenerated2_: _dafny.Seq
                        d_23_appendedInside2_: bool
                        d_24_appendedCurrent2_: _dafny.Seq
                        out12_: _dafny.Seq
                        out13_: bool
                        out14_: _dafny.Seq
                        out12_, out13_, out14_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_21_next2_)
                        d_22_appendedGenerated2_ = out12_
                        d_23_appendedInside2_ = out13_
                        d_24_appendedCurrent2_ = out14_
                        generated = d_22_appendedGenerated2_
                        insideConstrainedOut = d_23_appendedInside2_
                        currentConstrainedOut = d_24_appendedCurrent2_
            if ((insideConstrainedOut) and ((parser).IsCompletePrefix(currentConstrainedOut))) and ((d_4_steps_) < (maxSteps)):
                d_25_closedGenerated_: _dafny.Seq
                d_26_closedInside_: bool
                d_27_closedCurrent_: _dafny.Seq
                out15_: _dafny.Seq
                out16_: bool
                out17_: _dafny.Seq
                out15_, out16_, out17_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_25_closedGenerated_ = out15_
                d_26_closedInside_ = out16_
                d_27_closedCurrent_ = out17_
                generated = d_25_closedGenerated_
                insideConstrainedOut = d_26_closedInside_
                currentConstrainedOut = d_27_closedCurrent_
                d_4_steps_ = (d_4_steps_) + (1)
            cost = d_4_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

