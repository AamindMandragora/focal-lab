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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Output a valid and complete SQL query. Use WHERE clauses, JOINs, and conditions as needed. Write the full SQL query without abbreviation.")))
        d_1_steps_: int
        d_1_steps_ = 0
        if (not(insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
            d_2_og_: _dafny.Seq
            d_3_oi_: bool
            d_4_oc_: _dafny.Seq
            out0_: _dafny.Seq
            out1_: bool
            out2_: _dafny.Seq
            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_2_og_ = out0_
            d_3_oi_ = out1_
            d_4_oc_ = out2_
            generated = d_2_og_
            insideConstrainedOut = d_3_oi_
            currentConstrainedOut = d_4_oc_
            d_1_steps_ = (d_1_steps_) + (1)
        d_5_minTokensBeforeClose_: int
        d_5_minTokensBeforeClose_ = 50
        d_6_constrainedTokensGenerated_: int
        d_6_constrainedTokensGenerated_ = len(currentConstrainedOut)
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        raise _dafny.Break("0")
                    d_7_isDeadEnd_: bool
                    out3_: bool
                    out3_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                    d_7_isDeadEnd_ = out3_
                    if d_7_isDeadEnd_:
                        d_8_rg_: _dafny.Seq
                        d_9_rc_: _dafny.Seq
                        out4_: _dafny.Seq
                        out5_: _dafny.Seq
                        out4_, out5_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                        d_8_rg_ = out4_
                        d_9_rc_ = out5_
                        generated = d_8_rg_
                        currentConstrainedOut = d_9_rc_
                        if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                            d_10_cg_: _dafny.Seq
                            d_11_ci_: bool
                            d_12_cc_: _dafny.Seq
                            out6_: _dafny.Seq
                            out7_: bool
                            out8_: _dafny.Seq
                            out6_, out7_, out8_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_10_cg_ = out6_
                            d_11_ci_ = out7_
                            d_12_cc_ = out8_
                            generated = d_10_cg_
                            insideConstrainedOut = d_11_ci_
                            currentConstrainedOut = d_12_cc_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif (d_1_steps_) < (maxSteps):
                            d_13_closeBudget_: int
                            d_13_closeBudget_ = (maxSteps) - (d_1_steps_)
                            d_14_cg2_: _dafny.Seq
                            d_15_ci2_: bool
                            d_16_cc2_: _dafny.Seq
                            out9_: _dafny.Seq
                            out10_: bool
                            out11_: _dafny.Seq
                            out9_, out10_, out11_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_13_closeBudget_)
                            d_14_cg2_ = out9_
                            d_15_ci2_ = out10_
                            d_16_cc2_ = out11_
                            generated = d_14_cg2_
                            insideConstrainedOut = d_15_ci2_
                            currentConstrainedOut = d_16_cc2_
                            d_1_steps_ = maxSteps
                        raise _dafny.Break("0")
                    d_17_budgetRemaining_: int
                    d_17_budgetRemaining_ = (maxSteps) - (d_1_steps_)
                    if (((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_6_constrainedTokensGenerated_) >= (d_5_minTokensBeforeClose_))) and ((d_17_budgetRemaining_) <= (20)):
                        d_18_cg_: _dafny.Seq
                        d_19_ci_: bool
                        d_20_cc_: _dafny.Seq
                        out12_: _dafny.Seq
                        out13_: bool
                        out14_: _dafny.Seq
                        out12_, out13_, out14_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_18_cg_ = out12_
                        d_19_ci_ = out13_
                        d_20_cc_ = out14_
                        generated = d_18_cg_
                        insideConstrainedOut = d_19_ci_
                        currentConstrainedOut = d_20_cc_
                        d_1_steps_ = (d_1_steps_) + (1)
                        raise _dafny.Break("0")
                    if ((d_6_constrainedTokensGenerated_) >= (5)) and ((len(currentConstrainedOut)) >= (3)):
                        d_21_lastTok_: _dafny.Seq
                        d_21_lastTok_ = (currentConstrainedOut)[(len(currentConstrainedOut)) - (1)]
                        d_22_lastTok2_: _dafny.Seq
                        d_22_lastTok2_ = (currentConstrainedOut)[(len(currentConstrainedOut)) - (2)]
                        d_23_lastTok3_: _dafny.Seq
                        d_23_lastTok3_ = (currentConstrainedOut)[(len(currentConstrainedOut)) - (3)]
                        (d_0_helpers_).SafePenalizeTokenLogits(lm, _dafny.SeqWithoutIsStrInference([d_21_lastTok_, d_22_lastTok2_, d_23_lastTok3_]), _dafny.BigRational('3e0'))
                    d_24_constrainedPrompt_: _dafny.Seq
                    d_24_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                    d_25_next_: _dafny.Seq
                    out15_: _dafny.Seq
                    out15_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_24_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 30, eosToken)
                    d_25_next_ = out15_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if (d_25_next_) == (eosToken):
                        if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) <= (maxSteps)):
                            if (d_1_steps_) < (maxSteps):
                                d_26_cg_: _dafny.Seq
                                d_27_ci_: bool
                                d_28_cc_: _dafny.Seq
                                out16_: _dafny.Seq
                                out17_: bool
                                out18_: _dafny.Seq
                                out16_, out17_, out18_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_26_cg_ = out16_
                                d_27_ci_ = out17_
                                d_28_cc_ = out18_
                                generated = d_26_cg_
                                insideConstrainedOut = d_27_ci_
                                currentConstrainedOut = d_28_cc_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                d_29_closeBudget_: int
                                d_29_closeBudget_ = (maxSteps) - (d_1_steps_)
                                d_30_cg2_: _dafny.Seq
                                d_31_ci2_: bool
                                d_32_cc2_: _dafny.Seq
                                out19_: _dafny.Seq
                                out20_: bool
                                out21_: _dafny.Seq
                                out19_, out20_, out21_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_29_closeBudget_)
                                d_30_cg2_ = out19_
                                d_31_ci2_ = out20_
                                d_32_cc2_ = out21_
                                generated = d_30_cg2_
                                insideConstrainedOut = d_31_ci2_
                                currentConstrainedOut = d_32_cc2_
                                d_1_steps_ = maxSteps
                        elif (d_1_steps_) < (maxSteps):
                            d_33_closeBudget_: int
                            d_33_closeBudget_ = (maxSteps) - (d_1_steps_)
                            d_34_cg2_: _dafny.Seq
                            d_35_ci2_: bool
                            d_36_cc2_: _dafny.Seq
                            out22_: _dafny.Seq
                            out23_: bool
                            out24_: _dafny.Seq
                            out22_, out23_, out24_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_33_closeBudget_)
                            d_34_cg2_ = out22_
                            d_35_ci2_ = out23_
                            d_36_cc2_ = out24_
                            generated = d_34_cg2_
                            insideConstrainedOut = d_35_ci2_
                            currentConstrainedOut = d_36_cc2_
                            d_1_steps_ = maxSteps
                        raise _dafny.Break("0")
                    elif True:
                        d_37_ag_: _dafny.Seq
                        d_38_ai_: bool
                        d_39_ac_: _dafny.Seq
                        out25_: _dafny.Seq
                        out26_: bool
                        out27_: _dafny.Seq
                        out25_, out26_, out27_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_25_next_)
                        d_37_ag_ = out25_
                        d_38_ai_ = out26_
                        d_39_ac_ = out27_
                        generated = d_37_ag_
                        insideConstrainedOut = d_38_ai_
                        currentConstrainedOut = d_39_ac_
                        d_6_constrainedTokensGenerated_ = (d_6_constrainedTokensGenerated_) + (1)
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_40_closeBudget_: int
            d_40_closeBudget_ = (maxSteps) - (d_1_steps_)
            d_41_cg_: _dafny.Seq
            d_42_ci_: bool
            d_43_cc_: _dafny.Seq
            out28_: _dafny.Seq
            out29_: bool
            out30_: _dafny.Seq
            out28_, out29_, out30_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_40_closeBudget_)
            d_41_cg_ = out28_
            d_42_ci_ = out29_
            d_43_cc_ = out30_
            generated = d_41_cg_
            insideConstrainedOut = d_42_ci_
            currentConstrainedOut = d_43_cc_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

