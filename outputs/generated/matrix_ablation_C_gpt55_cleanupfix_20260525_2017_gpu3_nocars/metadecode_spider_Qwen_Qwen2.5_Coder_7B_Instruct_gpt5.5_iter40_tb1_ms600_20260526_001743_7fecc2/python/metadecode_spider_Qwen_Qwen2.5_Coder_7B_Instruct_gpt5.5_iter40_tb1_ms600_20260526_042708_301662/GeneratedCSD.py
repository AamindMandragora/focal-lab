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
        if True:
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
                d_6_prefixLimit_: int
                d_6_prefixLimit_ = 56
                d_7_earlyPenaltyTokens_: _dafny.Seq
                d_7_earlyPenaltyTokens_ = _dafny.SeqWithoutIsStrInference([eosToken, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ";")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "T1")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "T2")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "T3")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "T4")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "T5"))])
                while ((((((d_4_steps_) + (1)) < (maxSteps)) and (insideConstrainedOut)) and ((len(currentConstrainedOut)) < (d_6_prefixLimit_))) and (not(d_5_hitEos_))) and (not((parser).IsCompletePrefix(currentConstrainedOut))):
                    d_8_stablePrefix_: _dafny.Seq
                    d_8_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                    d_9_constrainedPrompt_: _dafny.Seq
                    d_9_constrainedPrompt_ = (prompt) + (d_8_stablePrefix_)
                    d_10_next_: _dafny.Seq
                    out3_: _dafny.Seq
                    out3_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_9_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('45e-1'), d_7_earlyPenaltyTokens_, _dafny.BigRational('6e0'), 18, eosToken)
                    d_10_next_ = out3_
                    d_4_steps_ = (d_4_steps_) + (1)
                    if (d_10_next_) == (eosToken):
                        if ((d_4_steps_) + (1)) < (maxSteps):
                            d_11_candidates_: _dafny.Seq
                            out4_: _dafny.Seq
                            out4_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_9_constrainedPrompt_, currentConstrainedOut, 4, eosToken)
                            d_11_candidates_ = out4_
                            d_4_steps_ = (d_4_steps_) + (1)
                            d_12_alt_: _dafny.Seq
                            d_12_alt_ = eosToken
                            if (len(d_11_candidates_)) > (0):
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
                    d_20_firstCap_: int
                    d_20_firstCap_ = ((maxSteps) - (d_4_steps_)) - (1)
                    if (d_20_firstCap_) > (220):
                        d_20_firstCap_ = 220
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
                    out12_, out13_, out14_, out15_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_22_constrainedPrompt2_, generated, currentConstrainedOut, d_20_firstCap_, eosToken)
                    d_23_symbolGenerated_ = out12_
                    d_24_symbolOut_ = out13_
                    d_25_symbolHitEos_ = out14_
                    d_26_used_ = out15_
                    generated = d_23_symbolGenerated_
                    currentConstrainedOut = d_24_symbolOut_
                    insideConstrainedOut = True
                    d_5_hitEos_ = d_25_symbolHitEos_
                    d_4_steps_ = (d_4_steps_) + (d_26_used_)
                d_27_tailPenaltyTokens_: _dafny.Seq
                d_27_tailPenaltyTokens_ = _dafny.SeqWithoutIsStrInference([eosToken, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ";"))])
                while ((((((d_4_steps_) + (1)) < (maxSteps)) and (insideConstrainedOut)) and ((len(currentConstrainedOut)) < (420))) and (not(d_5_hitEos_))) and (not((parser).IsCompletePrefix(currentConstrainedOut))):
                    d_28_stablePrefix3_: _dafny.Seq
                    d_28_stablePrefix3_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                    d_29_constrainedPrompt3_: _dafny.Seq
                    d_29_constrainedPrompt3_ = (prompt) + (d_28_stablePrefix3_)
                    d_30_next2_: _dafny.Seq
                    out16_: _dafny.Seq
                    out16_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_29_constrainedPrompt3_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_27_tailPenaltyTokens_, _dafny.BigRational('5e0'), 16, eosToken)
                    d_30_next2_ = out16_
                    d_4_steps_ = (d_4_steps_) + (1)
                    if (d_30_next2_) == (eosToken):
                        if ((d_4_steps_) + (1)) < (maxSteps):
                            d_31_candidates2_: _dafny.Seq
                            out17_: _dafny.Seq
                            out17_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_29_constrainedPrompt3_, currentConstrainedOut, 4, eosToken)
                            d_31_candidates2_ = out17_
                            d_4_steps_ = (d_4_steps_) + (1)
                            d_32_alt2_: _dafny.Seq
                            d_32_alt2_ = eosToken
                            if (len(d_31_candidates2_)) > (0):
                                d_32_alt2_ = (d_31_candidates2_)[0]
                                if ((d_32_alt2_) == (eosToken)) and ((len(d_31_candidates2_)) > (1)):
                                    d_32_alt2_ = (d_31_candidates2_)[1]
                            if (d_32_alt2_) == (eosToken):
                                d_5_hitEos_ = True
                            elif True:
                                d_33_altValid2_: bool
                                out18_: bool
                                out18_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_32_alt2_)
                                d_33_altValid2_ = out18_
                                if d_33_altValid2_:
                                    d_34_fallbackGenerated2_: _dafny.Seq
                                    d_35_fallbackInside2_: bool
                                    d_36_fallbackCurrent2_: _dafny.Seq
                                    out19_: _dafny.Seq
                                    out20_: bool
                                    out21_: _dafny.Seq
                                    out19_, out20_, out21_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_32_alt2_)
                                    d_34_fallbackGenerated2_ = out19_
                                    d_35_fallbackInside2_ = out20_
                                    d_36_fallbackCurrent2_ = out21_
                                    generated = d_34_fallbackGenerated2_
                                    insideConstrainedOut = d_35_fallbackInside2_
                                    currentConstrainedOut = d_36_fallbackCurrent2_
                                elif True:
                                    d_5_hitEos_ = True
                        elif True:
                            d_5_hitEos_ = True
                    elif True:
                        d_37_appendedGenerated2_: _dafny.Seq
                        d_38_appendedInside2_: bool
                        d_39_appendedCurrent2_: _dafny.Seq
                        out22_: _dafny.Seq
                        out23_: bool
                        out24_: _dafny.Seq
                        out22_, out23_, out24_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_30_next2_)
                        d_37_appendedGenerated2_ = out22_
                        d_38_appendedInside2_ = out23_
                        d_39_appendedCurrent2_ = out24_
                        generated = d_37_appendedGenerated2_
                        insideConstrainedOut = d_38_appendedInside2_
                        currentConstrainedOut = d_39_appendedCurrent2_
                if ((insideConstrainedOut) and ((parser).IsCompletePrefix(currentConstrainedOut))) and ((d_4_steps_) < (maxSteps)):
                    d_40_closedGenerated_: _dafny.Seq
                    d_41_closedInside_: bool
                    d_42_closedCurrent_: _dafny.Seq
                    out25_: _dafny.Seq
                    out26_: bool
                    out27_: _dafny.Seq
                    out25_, out26_, out27_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                    d_40_closedGenerated_ = out25_
                    d_41_closedInside_ = out26_
                    d_42_closedCurrent_ = out27_
                    generated = d_40_closedGenerated_
                    insideConstrainedOut = d_41_closedInside_
                    currentConstrainedOut = d_42_closedCurrent_
                    d_4_steps_ = (d_4_steps_) + (1)
                cost = d_4_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

