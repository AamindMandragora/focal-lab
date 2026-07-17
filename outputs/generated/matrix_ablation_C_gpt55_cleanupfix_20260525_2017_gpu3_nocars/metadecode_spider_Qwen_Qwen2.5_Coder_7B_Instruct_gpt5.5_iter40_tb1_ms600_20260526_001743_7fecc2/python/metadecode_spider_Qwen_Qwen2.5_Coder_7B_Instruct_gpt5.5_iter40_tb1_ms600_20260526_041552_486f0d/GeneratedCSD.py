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
        if (maxSteps) == (0):
            cost = 0
        elif (maxSteps) == (1):
            insideConstrainedOut = False
            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
            generated = (generatedPrefix) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SQL:"))]))
            cost = 1
        elif (maxSteps) == (2):
            insideConstrainedOut = False
            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
            generated = (generatedPrefix) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SQL:")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " "))]))
            cost = 2
        elif True:
            insideConstrainedOut = False
            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
            generated = (generatedPrefix) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SQL:")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " "))]))
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
            d_6_lengthLimit_: int
            d_6_lengthLimit_ = 460
            d_7_penaltyTokens_: _dafny.Seq
            d_7_penaltyTokens_ = _dafny.SeqWithoutIsStrInference([eosToken, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ";")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SELECT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "FROM")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "WHERE")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "JOIN")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "INNER")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "LEFT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "RIGHT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "FULL")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "ON")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "GROUP")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "ORDER")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "BY")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "HAVING")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "LIMIT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "AND")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "OR")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "IN")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "NOT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "EXISTS")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "BETWEEN")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "LIKE")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "COUNT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SUM")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "AVG")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "MIN")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "MAX")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "DISTINCT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "AS")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "DESC")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "ASC"))])
            while ((((((d_4_steps_) + (1)) < (maxSteps)) and (insideConstrainedOut)) and ((len(currentConstrainedOut)) < (d_6_lengthLimit_))) and (not(d_5_hitEos_))) and (not((parser).IsCompletePrefix(currentConstrainedOut))):
                d_8_stablePrefix_: _dafny.Seq
                d_8_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                d_9_constrainedPrompt_: _dafny.Seq
                d_9_constrainedPrompt_ = (prompt) + (d_8_stablePrefix_)
                d_10_next_: _dafny.Seq
                out3_: _dafny.Seq
                out3_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_9_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('5e0'), d_7_penaltyTokens_, _dafny.BigRational('25e-1'), 24, eosToken)
                d_10_next_ = out3_
                d_4_steps_ = (d_4_steps_) + (1)
                if (d_10_next_) == (eosToken):
                    if ((d_4_steps_) + (1)) < (maxSteps):
                        d_11_candidates_: _dafny.Seq
                        out4_: _dafny.Seq
                        out4_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_9_constrainedPrompt_, currentConstrainedOut, 5, eosToken)
                        d_11_candidates_ = out4_
                        d_4_steps_ = (d_4_steps_) + (1)
                        if (len(d_11_candidates_)) == (0):
                            d_5_hitEos_ = True
                        elif True:
                            d_12_alt_: _dafny.Seq
                            d_12_alt_ = (d_11_candidates_)[0]
                            if ((d_12_alt_) == (eosToken)) and ((len(d_11_candidates_)) > (1)):
                                d_12_alt_ = (d_11_candidates_)[1]
                            if (d_12_alt_) == (eosToken):
                                d_5_hitEos_ = True
                            elif True:
                                d_13_altValid_: bool
                                out5_: bool
                                out5_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_12_alt_)
                                d_13_altValid_ = out5_
                                if d_13_altValid_:
                                    d_14_fallbackGenerated_: _dafny.Seq
                                    d_15_fallbackInside_: bool
                                    d_16_fallbackCurrent_: _dafny.Seq
                                    out6_: _dafny.Seq
                                    out7_: bool
                                    out8_: _dafny.Seq
                                    out6_, out7_, out8_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_12_alt_)
                                    d_14_fallbackGenerated_ = out6_
                                    d_15_fallbackInside_ = out7_
                                    d_16_fallbackCurrent_ = out8_
                                    generated = d_14_fallbackGenerated_
                                    insideConstrainedOut = d_15_fallbackInside_
                                    currentConstrainedOut = d_16_fallbackCurrent_
                                elif True:
                                    d_5_hitEos_ = True
                    elif True:
                        d_5_hitEos_ = True
                elif True:
                    d_17_appendedGenerated_: _dafny.Seq
                    d_18_appendedInside_: bool
                    d_19_appendedCurrent_: _dafny.Seq
                    out9_: _dafny.Seq
                    out10_: bool
                    out11_: _dafny.Seq
                    out9_, out10_, out11_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_10_next_)
                    d_17_appendedGenerated_ = out9_
                    d_18_appendedInside_ = out10_
                    d_19_appendedCurrent_ = out11_
                    generated = d_17_appendedGenerated_
                    insideConstrainedOut = d_18_appendedInside_
                    currentConstrainedOut = d_19_appendedCurrent_
            if (((((d_4_steps_) + (1)) < (maxSteps)) and (insideConstrainedOut)) and (not(d_5_hitEos_))) and (not((parser).IsCompletePrefix(currentConstrainedOut))):
                d_20_remainingForSymbol_: int
                d_20_remainingForSymbol_ = ((maxSteps) - (d_4_steps_)) - (1)
                if (d_20_remainingForSymbol_) > (80):
                    d_20_remainingForSymbol_ = 80
                d_21_stablePrefix2_: _dafny.Seq
                d_21_stablePrefix2_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                d_22_constrainedPrompt2_: _dafny.Seq
                d_22_constrainedPrompt2_ = (prompt) + (d_21_stablePrefix2_)
                d_23_symbolGenerated_: _dafny.Seq
                d_24_symbolOut_: _dafny.Seq
                d_25_symbolHitEos_: bool
                d_26_used_: int
                out12_: _dafny.Seq
                out13_: _dafny.Seq
                out14_: bool
                out15_: int
                out12_, out13_, out14_, out15_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_22_constrainedPrompt2_, generated, currentConstrainedOut, d_20_remainingForSymbol_, eosToken)
                d_23_symbolGenerated_ = out12_
                d_24_symbolOut_ = out13_
                d_25_symbolHitEos_ = out14_
                d_26_used_ = out15_
                generated = d_23_symbolGenerated_
                currentConstrainedOut = d_24_symbolOut_
                insideConstrainedOut = True
                d_5_hitEos_ = d_25_symbolHitEos_
                d_4_steps_ = (d_4_steps_) + (d_26_used_)
            while (((((d_4_steps_) + (1)) < (maxSteps)) and (insideConstrainedOut)) and (not(d_5_hitEos_))) and (not((parser).IsCompletePrefix(currentConstrainedOut))):
                d_27_stablePrefix3_: _dafny.Seq
                d_27_stablePrefix3_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                d_28_constrainedPrompt3_: _dafny.Seq
                d_28_constrainedPrompt3_ = (prompt) + (d_27_stablePrefix3_)
                d_29_next2_: _dafny.Seq
                out16_: _dafny.Seq
                out16_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_28_constrainedPrompt3_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), _dafny.SeqWithoutIsStrInference([eosToken, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ";"))]), _dafny.BigRational('5e0'), 16, eosToken)
                d_29_next2_ = out16_
                d_4_steps_ = (d_4_steps_) + (1)
                if (d_29_next2_) == (eosToken):
                    d_5_hitEos_ = True
                elif True:
                    d_30_appendedGenerated2_: _dafny.Seq
                    d_31_appendedInside2_: bool
                    d_32_appendedCurrent2_: _dafny.Seq
                    out17_: _dafny.Seq
                    out18_: bool
                    out19_: _dafny.Seq
                    out17_, out18_, out19_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_29_next2_)
                    d_30_appendedGenerated2_ = out17_
                    d_31_appendedInside2_ = out18_
                    d_32_appendedCurrent2_ = out19_
                    generated = d_30_appendedGenerated2_
                    insideConstrainedOut = d_31_appendedInside2_
                    currentConstrainedOut = d_32_appendedCurrent2_
            if ((insideConstrainedOut) and ((parser).IsCompletePrefix(currentConstrainedOut))) and ((d_4_steps_) < (maxSteps)):
                d_33_closedGenerated_: _dafny.Seq
                d_34_closedInside_: bool
                d_35_closedCurrent_: _dafny.Seq
                out20_: _dafny.Seq
                out21_: bool
                out22_: _dafny.Seq
                out20_, out21_, out22_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_33_closedGenerated_ = out20_
                d_34_closedInside_ = out21_
                d_35_closedCurrent_ = out22_
                generated = d_33_closedGenerated_
                insideConstrainedOut = d_34_closedInside_
                currentConstrainedOut = d_35_closedCurrent_
                d_4_steps_ = (d_4_steps_) + (1)
            cost = d_4_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

