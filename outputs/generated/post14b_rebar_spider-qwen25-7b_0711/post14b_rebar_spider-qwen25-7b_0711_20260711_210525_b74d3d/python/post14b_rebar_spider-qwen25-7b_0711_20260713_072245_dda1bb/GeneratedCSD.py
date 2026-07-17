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
        if (maxSteps) == (0):
            return generated, insideConstrainedOut, currentConstrainedOut, cost
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Write a complete, concise SQL SELECT query that answers the question. Use only tables and columns from the schema. Include proper WHERE conditions, JOINs, subqueries, and aggregations as needed. Write minimal but complete SQL.")))
        d_1_steps_: int
        d_1_steps_ = 0
        if (not(insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
            (d_0_helpers_).SafeBoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]), _dafny.BigRational('25e0'))
            d_2_next_: _dafny.Seq
            out0_: _dafny.Seq
            out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
            d_2_next_ = out0_
            d_1_steps_ = (d_1_steps_) + (1)
            if (d_2_next_) == (eosToken):
                cost = d_1_steps_
                return generated, insideConstrainedOut, currentConstrainedOut, cost
            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_2_next_]))
            if (d_2_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                d_3_og_: _dafny.Seq
                d_4_oi_: bool
                d_5_oc_: _dafny.Seq
                out1_: _dafny.Seq
                out2_: bool
                out3_: _dafny.Seq
                out1_, out2_, out3_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                d_3_og_ = out1_
                d_4_oi_ = out2_
                d_5_oc_ = out3_
                generated = d_3_og_
                insideConstrainedOut = d_4_oi_
                currentConstrainedOut = d_5_oc_
            elif True:
                d_6_attempt_: int
                d_6_attempt_ = 0
                while (((d_6_attempt_) < (4)) and ((d_1_steps_) < (maxSteps))) and (not(insideConstrainedOut)):
                    (d_0_helpers_).SafeBoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]), _dafny.BigRational('25e0'))
                    d_7_next2_: _dafny.Seq
                    out4_: _dafny.Seq
                    out4_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                    d_7_next2_ = out4_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if (d_7_next2_) == (eosToken):
                        cost = d_1_steps_
                        return generated, insideConstrainedOut, currentConstrainedOut, cost
                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_7_next2_]))
                    if (d_7_next2_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                        d_8_og2_: _dafny.Seq
                        d_9_oi2_: bool
                        d_10_oc2_: _dafny.Seq
                        out5_: _dafny.Seq
                        out6_: bool
                        out7_: _dafny.Seq
                        out5_, out6_, out7_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                        d_8_og2_ = out5_
                        d_9_oi2_ = out6_
                        d_10_oc2_ = out7_
                        generated = d_8_og2_
                        insideConstrainedOut = d_9_oi2_
                        currentConstrainedOut = d_10_oc2_
                    d_6_attempt_ = (d_6_attempt_) + (1)
                if (not(insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                    d_11_og3_: _dafny.Seq
                    d_12_oi3_: bool
                    d_13_oc3_: _dafny.Seq
                    out8_: _dafny.Seq
                    out9_: bool
                    out10_: _dafny.Seq
                    out8_, out9_, out10_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                    d_11_og3_ = out8_
                    d_12_oi3_ = out9_
                    d_13_oc3_ = out10_
                    generated = d_11_og3_
                    insideConstrainedOut = d_12_oi3_
                    currentConstrainedOut = d_13_oc3_
                    d_1_steps_ = (d_1_steps_) + (1)
        d_14_minTokensBeforeClose_: int
        d_14_minTokensBeforeClose_ = 8
        d_15_maxSpanTokens_: int
        d_15_maxSpanTokens_ = 120
        d_16_constrainedTokensGenerated_: int
        d_16_constrainedTokensGenerated_ = len(currentConstrainedOut)
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        raise _dafny.Break("0")
                    if (d_16_constrainedTokensGenerated_) >= (d_15_maxSpanTokens_):
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            d_17_cg_: _dafny.Seq
                            d_18_ci_: bool
                            d_19_cc_: _dafny.Seq
                            out11_: _dafny.Seq
                            out12_: bool
                            out13_: _dafny.Seq
                            out11_, out12_, out13_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_17_cg_ = out11_
                            d_18_ci_ = out12_
                            d_19_cc_ = out13_
                            generated = d_17_cg_
                            insideConstrainedOut = d_18_ci_
                            currentConstrainedOut = d_19_cc_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif (d_1_steps_) < (maxSteps):
                            d_20_closeBudget_: int
                            d_20_closeBudget_ = (maxSteps) - (d_1_steps_)
                            d_21_cg2_: _dafny.Seq
                            d_22_ci2_: bool
                            d_23_cc2_: _dafny.Seq
                            out14_: _dafny.Seq
                            out15_: bool
                            out16_: _dafny.Seq
                            out14_, out15_, out16_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_20_closeBudget_)
                            d_21_cg2_ = out14_
                            d_22_ci2_ = out15_
                            d_23_cc2_ = out16_
                            generated = d_21_cg2_
                            insideConstrainedOut = d_22_ci2_
                            currentConstrainedOut = d_23_cc2_
                            d_1_steps_ = maxSteps
                        raise _dafny.Break("0")
                    d_24_isDeadEnd_: bool
                    out17_: bool
                    out17_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                    d_24_isDeadEnd_ = out17_
                    if d_24_isDeadEnd_:
                        d_25_rg_: _dafny.Seq
                        d_26_rc_: _dafny.Seq
                        out18_: _dafny.Seq
                        out19_: _dafny.Seq
                        out18_, out19_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                        d_25_rg_ = out18_
                        d_26_rc_ = out19_
                        generated = d_25_rg_
                        currentConstrainedOut = d_26_rc_
                        if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                            d_27_cg_: _dafny.Seq
                            d_28_ci_: bool
                            d_29_cc_: _dafny.Seq
                            out20_: _dafny.Seq
                            out21_: bool
                            out22_: _dafny.Seq
                            out20_, out21_, out22_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_27_cg_ = out20_
                            d_28_ci_ = out21_
                            d_29_cc_ = out22_
                            generated = d_27_cg_
                            insideConstrainedOut = d_28_ci_
                            currentConstrainedOut = d_29_cc_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif (d_1_steps_) < (maxSteps):
                            d_30_closeBudget_: int
                            d_30_closeBudget_ = (maxSteps) - (d_1_steps_)
                            d_31_cg2_: _dafny.Seq
                            d_32_ci2_: bool
                            d_33_cc2_: _dafny.Seq
                            out23_: _dafny.Seq
                            out24_: bool
                            out25_: _dafny.Seq
                            out23_, out24_, out25_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_30_closeBudget_)
                            d_31_cg2_ = out23_
                            d_32_ci2_ = out24_
                            d_33_cc2_ = out25_
                            generated = d_31_cg2_
                            insideConstrainedOut = d_32_ci2_
                            currentConstrainedOut = d_33_cc2_
                            d_1_steps_ = maxSteps
                        raise _dafny.Break("0")
                    if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_16_constrainedTokensGenerated_) >= (d_14_minTokensBeforeClose_)):
                        d_34_cg_: _dafny.Seq
                        d_35_ci_: bool
                        d_36_cc_: _dafny.Seq
                        d_37_closed_: bool
                        out26_: _dafny.Seq
                        out27_: bool
                        out28_: _dafny.Seq
                        out29_: bool
                        out26_, out27_, out28_, out29_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                        d_34_cg_ = out26_
                        d_35_ci_ = out27_
                        d_36_cc_ = out28_
                        d_37_closed_ = out29_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if d_37_closed_:
                            generated = d_34_cg_
                            insideConstrainedOut = d_35_ci_
                            currentConstrainedOut = d_36_cc_
                            raise _dafny.Break("0")
                        if (d_1_steps_) >= (maxSteps):
                            raise _dafny.Break("0")
                    d_38_constrainedPrompt_: _dafny.Seq
                    d_38_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                    d_39_penaltyToks_: _dafny.Seq
                    d_39_penaltyToks_ = _dafny.SeqWithoutIsStrInference([])
                    if (len(currentConstrainedOut)) >= (3):
                        d_39_penaltyToks_ = _dafny.SeqWithoutIsStrInference([(currentConstrainedOut)[(len(currentConstrainedOut)) - (1)], (currentConstrainedOut)[(len(currentConstrainedOut)) - (2)], (currentConstrainedOut)[(len(currentConstrainedOut)) - (3)]])
                    elif (len(currentConstrainedOut)) >= (2):
                        d_39_penaltyToks_ = _dafny.SeqWithoutIsStrInference([(currentConstrainedOut)[(len(currentConstrainedOut)) - (1)], (currentConstrainedOut)[(len(currentConstrainedOut)) - (2)]])
                    elif (len(currentConstrainedOut)) >= (1):
                        d_39_penaltyToks_ = _dafny.SeqWithoutIsStrInference([(currentConstrainedOut)[(len(currentConstrainedOut)) - (1)]])
                    d_40_next_: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                    if (len(d_39_penaltyToks_)) > (0):
                        out30_: _dafny.Seq
                        out30_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_38_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_39_penaltyToks_, _dafny.BigRational('3e0'), 20, eosToken)
                        d_40_next_ = out30_
                    elif True:
                        out31_: _dafny.Seq
                        out31_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_38_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 20, eosToken)
                        d_40_next_ = out31_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if (d_40_next_) == (eosToken):
                        if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                            d_41_cg_: _dafny.Seq
                            d_42_ci_: bool
                            d_43_cc_: _dafny.Seq
                            out32_: _dafny.Seq
                            out33_: bool
                            out34_: _dafny.Seq
                            out32_, out33_, out34_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_41_cg_ = out32_
                            d_42_ci_ = out33_
                            d_43_cc_ = out34_
                            generated = d_41_cg_
                            insideConstrainedOut = d_42_ci_
                            currentConstrainedOut = d_43_cc_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif (d_1_steps_) < (maxSteps):
                            d_44_closeBudget_: int
                            d_44_closeBudget_ = (maxSteps) - (d_1_steps_)
                            d_45_cg2_: _dafny.Seq
                            d_46_ci2_: bool
                            d_47_cc2_: _dafny.Seq
                            out35_: _dafny.Seq
                            out36_: bool
                            out37_: _dafny.Seq
                            out35_, out36_, out37_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_44_closeBudget_)
                            d_45_cg2_ = out35_
                            d_46_ci2_ = out36_
                            d_47_cc2_ = out37_
                            generated = d_45_cg2_
                            insideConstrainedOut = d_46_ci2_
                            currentConstrainedOut = d_47_cc2_
                            d_1_steps_ = maxSteps
                        raise _dafny.Break("0")
                    elif True:
                        d_48_ag_: _dafny.Seq
                        d_49_ai_: bool
                        d_50_ac_: _dafny.Seq
                        out38_: _dafny.Seq
                        out39_: bool
                        out40_: _dafny.Seq
                        out38_, out39_, out40_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_40_next_)
                        d_48_ag_ = out38_
                        d_49_ai_ = out39_
                        d_50_ac_ = out40_
                        generated = d_48_ag_
                        insideConstrainedOut = d_49_ai_
                        currentConstrainedOut = d_50_ac_
                        d_16_constrainedTokensGenerated_ = (d_16_constrainedTokensGenerated_) + (1)
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_51_closeBudget_: int
            d_51_closeBudget_ = (maxSteps) - (d_1_steps_)
            d_52_cg_: _dafny.Seq
            d_53_ci_: bool
            d_54_cc_: _dafny.Seq
            out41_: _dafny.Seq
            out42_: bool
            out43_: _dafny.Seq
            out41_, out42_, out43_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_51_closeBudget_)
            d_52_cg_ = out41_
            d_53_ci_ = out42_
            d_54_cc_ = out43_
            generated = d_52_cg_
            insideConstrainedOut = d_53_ci_
            currentConstrainedOut = d_54_cc_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

