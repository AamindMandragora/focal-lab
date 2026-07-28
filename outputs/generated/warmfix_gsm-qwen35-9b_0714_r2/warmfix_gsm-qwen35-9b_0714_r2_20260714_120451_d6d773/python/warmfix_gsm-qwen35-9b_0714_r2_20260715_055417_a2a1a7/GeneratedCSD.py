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
        out0_: _dafny.Seq
        out1_: bool
        out2_: _dafny.Seq
        out3_: int
        out0_, out1_, out2_, out3_ = default__.AuthorBody(lm, parser, prompt, generatedPrefix, insideConstrained, currentConstrained, maxSteps, stepTokenBudget, validTokenGroups, eosToken)
        generated = out0_
        insideConstrainedOut = out1_
        currentConstrainedOut = out2_
        cost = out3_
        if ((maxSteps) > (0)) and ((cost) <= (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

    @staticmethod
    def AuthorBody(lm, parser, prompt, generatedPrefix, insideConstrained, currentConstrained, maxSteps, stepTokenBudget, validTokenGroups, eosToken):
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
        generated = generatedPrefix
        insideConstrainedOut = insideConstrained
        currentConstrainedOut = currentConstrained
        cost = 0
        d_1_guidance_: _dafny.Seq
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. Write the final answer as <<int(formula)>> where formula uses the actual variable names from the problem (no curly braces). Use only +, -, *, /, //, %. Do NOT use ** or ^. Write exactly one <<int(...)>>."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_prefixBudget_: int
        if (maxSteps) > (100):
            d_3_prefixBudget_ = (maxSteps) - (80)
        elif True:
            d_3_prefixBudget_ = _dafny.euclidian_division(maxSteps, 2)
        d_4_observedSpanThreshold_: int
        if (maxSteps) > (60):
            d_4_observedSpanThreshold_ = (maxSteps) - (50)
        elif True:
            d_4_observedSpanThreshold_ = maxSteps
        d_5_nearBudgetThreshold_: int
        if (maxSteps) > (50):
            d_5_nearBudgetThreshold_ = 40
        elif True:
            d_5_nearBudgetThreshold_ = _dafny.euclidian_division(maxSteps, 4)
        d_6_maxSpanTokens_: int
        d_6_maxSpanTokens_ = 20
        d_7_spanTokens_: int
        d_7_spanTokens_ = 0
        d_8_penaltyTokens_: _dafny.Seq
        d_8_penaltyTokens_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "**")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "^"))])
        with _dafny.label("0"):
            while (d_2_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_9_remainingBudget_: int
                        d_9_remainingBudget_ = (maxSteps) - (d_2_steps_)
                        if (d_2_steps_) >= (d_3_prefixBudget_):
                            d_10_genStr_: _dafny.Seq
                            d_10_genStr_ = VerifiedDecoderAgent.CSDHelpers.PrefixToString(generated)
                            d_11_openCount_: int
                            d_11_openCount_ = VerifiedDecoderAgent.CSDHelpers.CountSubstring(d_10_genStr_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))
                            d_12_closeCount_: int
                            d_12_closeCount_ = VerifiedDecoderAgent.CSDHelpers.CountSubstring(d_10_genStr_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")))
                            if ((d_12_closeCount_) == (0)) and ((d_9_remainingBudget_) > (5)):
                                d_13_og_: _dafny.Seq
                                d_14_oi_: bool
                                d_15_oc_: _dafny.Seq
                                out0_: _dafny.Seq
                                out1_: bool
                                out2_: _dafny.Seq
                                out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                d_13_og_ = out0_
                                d_14_oi_ = out1_
                                d_15_oc_ = out2_
                                generated = d_13_og_
                                insideConstrainedOut = d_14_oi_
                                currentConstrainedOut = d_15_oc_
                                d_2_steps_ = (d_2_steps_) + (1)
                                d_7_spanTokens_ = 0
                            elif True:
                                raise _dafny.Break("0")
                        elif True:
                            (d_0_helpers_).SafePenalizeTokenLogits(lm, d_8_penaltyTokens_, _dafny.BigRational('4e0'))
                            d_16_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_16_next_ = out3_
                            d_2_steps_ = (d_2_steps_) + (1)
                            if (d_16_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif ((d_16_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))) and ((d_2_steps_) >= (d_4_observedSpanThreshold_)):
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_16_next_]))
                                d_17_eg_: _dafny.Seq
                                d_18_ei_: bool
                                d_19_ec_: _dafny.Seq
                                out4_: _dafny.Seq
                                out5_: bool
                                out6_: _dafny.Seq
                                out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_17_eg_ = out4_
                                d_18_ei_ = out5_
                                d_19_ec_ = out6_
                                generated = d_17_eg_
                                insideConstrainedOut = d_18_ei_
                                currentConstrainedOut = d_19_ec_
                                d_7_spanTokens_ = 0
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_16_next_]))
                    elif True:
                        d_20_remainingBudget_: int
                        d_20_remainingBudget_ = (maxSteps) - (d_2_steps_)
                        if ((d_20_remainingBudget_) <= (d_5_nearBudgetThreshold_)) or ((d_7_spanTokens_) >= (d_6_maxSpanTokens_)):
                            if (d_20_remainingBudget_) > (0):
                                d_21_sg_: _dafny.Seq
                                d_22_si_: bool
                                d_23_sc_: _dafny.Seq
                                out7_: _dafny.Seq
                                out8_: bool
                                out9_: _dafny.Seq
                                out7_, out8_, out9_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_20_remainingBudget_)
                                d_21_sg_ = out7_
                                d_22_si_ = out8_
                                d_23_sc_ = out9_
                                generated = d_21_sg_
                                insideConstrainedOut = d_22_si_
                                currentConstrainedOut = d_23_sc_
                                d_2_steps_ = (d_2_steps_) + (d_20_remainingBudget_)
                            raise _dafny.Break("0")
                        d_24_cg_: _dafny.Seq
                        d_25_ci_: bool
                        d_26_cc_: _dafny.Seq
                        d_27_closed_: bool
                        out10_: _dafny.Seq
                        out11_: bool
                        out12_: _dafny.Seq
                        out13_: bool
                        out10_, out11_, out12_, out13_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                        d_24_cg_ = out10_
                        d_25_ci_ = out11_
                        d_26_cc_ = out12_
                        d_27_closed_ = out13_
                        d_2_steps_ = (d_2_steps_) + (1)
                        if d_27_closed_:
                            generated = d_24_cg_
                            insideConstrainedOut = d_25_ci_
                            currentConstrainedOut = d_26_cc_
                            raise _dafny.Break("0")
                        elif True:
                            d_28_constrainedPrompt_: _dafny.Seq
                            d_28_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_29_next_: _dafny.Seq
                            out14_: _dafny.Seq
                            out14_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_28_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('2e0'), d_8_penaltyTokens_, _dafny.BigRational('6e0'), 6, eosToken)
                            d_29_next_ = out14_
                            if (d_29_next_) == (eosToken):
                                d_30_rb__g_: _dafny.Seq
                                d_31_rb__c_: _dafny.Seq
                                out15_: _dafny.Seq
                                out16_: _dafny.Seq
                                out15_, out16_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                                d_30_rb__g_ = out15_
                                d_31_rb__c_ = out16_
                                generated = d_30_rb__g_
                                currentConstrainedOut = d_31_rb__c_
                                d_32_rem_: int
                                d_32_rem_ = (maxSteps) - (d_2_steps_)
                                if (d_32_rem_) > (0):
                                    if (parser).IsCompletePrefix(currentConstrainedOut):
                                        d_33_close__g_: _dafny.Seq
                                        d_34_close__i_: bool
                                        d_35_close__c_: _dafny.Seq
                                        out17_: _dafny.Seq
                                        out18_: bool
                                        out19_: _dafny.Seq
                                        out17_, out18_, out19_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                        d_33_close__g_ = out17_
                                        d_34_close__i_ = out18_
                                        d_35_close__c_ = out19_
                                        generated = d_33_close__g_
                                        insideConstrainedOut = d_34_close__i_
                                        currentConstrainedOut = d_35_close__c_
                                        d_2_steps_ = (d_2_steps_) + (1)
                                    elif True:
                                        d_36_sg2_: _dafny.Seq
                                        d_37_si2_: bool
                                        d_38_sc2_: _dafny.Seq
                                        out20_: _dafny.Seq
                                        out21_: bool
                                        out22_: _dafny.Seq
                                        out20_, out21_, out22_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_32_rem_)
                                        d_36_sg2_ = out20_
                                        d_37_si2_ = out21_
                                        d_38_sc2_ = out22_
                                        generated = d_36_sg2_
                                        insideConstrainedOut = d_37_si2_
                                        currentConstrainedOut = d_38_sc2_
                                        d_2_steps_ = (d_2_steps_) + (d_32_rem_)
                                raise _dafny.Break("0")
                            elif True:
                                d_39_ag_: _dafny.Seq
                                d_40_ai_: bool
                                d_41_ac_: _dafny.Seq
                                out23_: _dafny.Seq
                                out24_: bool
                                out25_: _dafny.Seq
                                out23_, out24_, out25_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_29_next_)
                                d_39_ag_ = out23_
                                d_40_ai_ = out24_
                                d_41_ac_ = out25_
                                generated = d_39_ag_
                                insideConstrainedOut = d_40_ai_
                                currentConstrainedOut = d_41_ac_
                                d_7_spanTokens_ = (d_7_spanTokens_) + (1)
                    pass
            pass
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

