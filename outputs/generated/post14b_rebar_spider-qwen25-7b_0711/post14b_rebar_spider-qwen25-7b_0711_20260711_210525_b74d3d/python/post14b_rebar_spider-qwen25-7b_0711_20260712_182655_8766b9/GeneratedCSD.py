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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "You must output a SQL query. Start your answer with << then write only the SQL query then >>. Keep the SQL simple and direct. Do not use unnecessary JOINs. Use only the tables mentioned in the schema that are needed. Select only the columns asked for.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_maxAttempts_: int
        d_2_maxAttempts_ = 15
        d_3_attempt_: int
        d_3_attempt_ = 0
        while (((d_3_attempt_) < (d_2_maxAttempts_)) and ((d_1_steps_) < (maxSteps))) and (not(insideConstrainedOut)):
            (d_0_helpers_).SafeBoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]), _dafny.BigRational('25e0'))
            d_4_next_: _dafny.Seq
            out0_: _dafny.Seq
            out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
            d_4_next_ = out0_
            d_1_steps_ = (d_1_steps_) + (1)
            if (d_4_next_) == (eosToken):
                cost = d_1_steps_
                return generated, insideConstrainedOut, currentConstrainedOut, cost
            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_4_next_]))
            if (d_4_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                d_5_og_: _dafny.Seq
                d_6_oi_: bool
                d_7_oc_: _dafny.Seq
                out1_: _dafny.Seq
                out2_: bool
                out3_: _dafny.Seq
                out1_, out2_, out3_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                d_5_og_ = out1_
                d_6_oi_ = out2_
                d_7_oc_ = out3_
                generated = d_5_og_
                insideConstrainedOut = d_6_oi_
                currentConstrainedOut = d_7_oc_
            d_3_attempt_ = (d_3_attempt_) + (1)
        if (not(insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
            d_8_og_: _dafny.Seq
            d_9_oi_: bool
            d_10_oc_: _dafny.Seq
            out4_: _dafny.Seq
            out5_: bool
            out6_: _dafny.Seq
            out4_, out5_, out6_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_8_og_ = out4_
            d_9_oi_ = out5_
            d_10_oc_ = out6_
            generated = d_8_og_
            insideConstrainedOut = d_9_oi_
            currentConstrainedOut = d_10_oc_
            d_1_steps_ = (d_1_steps_) + (1)
        d_11_minTokensBeforeClose_: int
        d_11_minTokensBeforeClose_ = 8
        d_12_constrainedTokensGenerated_: int
        d_12_constrainedTokensGenerated_ = len(currentConstrainedOut)
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        raise _dafny.Break("0")
                    d_13_isDeadEnd_: bool
                    out7_: bool
                    out7_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                    d_13_isDeadEnd_ = out7_
                    if d_13_isDeadEnd_:
                        d_14_rg_: _dafny.Seq
                        d_15_rc_: _dafny.Seq
                        out8_: _dafny.Seq
                        out9_: _dafny.Seq
                        out8_, out9_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                        d_14_rg_ = out8_
                        d_15_rc_ = out9_
                        generated = d_14_rg_
                        currentConstrainedOut = d_15_rc_
                        if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                            d_16_cg_: _dafny.Seq
                            d_17_ci_: bool
                            d_18_cc_: _dafny.Seq
                            out10_: _dafny.Seq
                            out11_: bool
                            out12_: _dafny.Seq
                            out10_, out11_, out12_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_16_cg_ = out10_
                            d_17_ci_ = out11_
                            d_18_cc_ = out12_
                            generated = d_16_cg_
                            insideConstrainedOut = d_17_ci_
                            currentConstrainedOut = d_18_cc_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif (d_1_steps_) < (maxSteps):
                            d_19_closeBudget_: int
                            d_19_closeBudget_ = (maxSteps) - (d_1_steps_)
                            d_20_cg2_: _dafny.Seq
                            d_21_ci2_: bool
                            d_22_cc2_: _dafny.Seq
                            out13_: _dafny.Seq
                            out14_: bool
                            out15_: _dafny.Seq
                            out13_, out14_, out15_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_19_closeBudget_)
                            d_20_cg2_ = out13_
                            d_21_ci2_ = out14_
                            d_22_cc2_ = out15_
                            generated = d_20_cg2_
                            insideConstrainedOut = d_21_ci2_
                            currentConstrainedOut = d_22_cc2_
                            d_1_steps_ = maxSteps
                        raise _dafny.Break("0")
                    if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_12_constrainedTokensGenerated_) >= (d_11_minTokensBeforeClose_)):
                        d_23_cg_: _dafny.Seq
                        d_24_ci_: bool
                        d_25_cc_: _dafny.Seq
                        d_26_closed_: bool
                        out16_: _dafny.Seq
                        out17_: bool
                        out18_: _dafny.Seq
                        out19_: bool
                        out16_, out17_, out18_, out19_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                        d_23_cg_ = out16_
                        d_24_ci_ = out17_
                        d_25_cc_ = out18_
                        d_26_closed_ = out19_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if d_26_closed_:
                            generated = d_23_cg_
                            insideConstrainedOut = d_24_ci_
                            currentConstrainedOut = d_25_cc_
                            raise _dafny.Break("0")
                        if (d_1_steps_) >= (maxSteps):
                            raise _dafny.Break("0")
                    d_27_constrainedPrompt_: _dafny.Seq
                    d_27_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                    d_28_next_: _dafny.Seq
                    out20_: _dafny.Seq
                    out20_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_27_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                    d_28_next_ = out20_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if (d_28_next_) == (eosToken):
                        if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                            d_29_cg_: _dafny.Seq
                            d_30_ci_: bool
                            d_31_cc_: _dafny.Seq
                            out21_: _dafny.Seq
                            out22_: bool
                            out23_: _dafny.Seq
                            out21_, out22_, out23_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_29_cg_ = out21_
                            d_30_ci_ = out22_
                            d_31_cc_ = out23_
                            generated = d_29_cg_
                            insideConstrainedOut = d_30_ci_
                            currentConstrainedOut = d_31_cc_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif (d_1_steps_) < (maxSteps):
                            d_32_closeBudget_: int
                            d_32_closeBudget_ = (maxSteps) - (d_1_steps_)
                            d_33_cg2_: _dafny.Seq
                            d_34_ci2_: bool
                            d_35_cc2_: _dafny.Seq
                            out24_: _dafny.Seq
                            out25_: bool
                            out26_: _dafny.Seq
                            out24_, out25_, out26_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_32_closeBudget_)
                            d_33_cg2_ = out24_
                            d_34_ci2_ = out25_
                            d_35_cc2_ = out26_
                            generated = d_33_cg2_
                            insideConstrainedOut = d_34_ci2_
                            currentConstrainedOut = d_35_cc2_
                            d_1_steps_ = maxSteps
                        raise _dafny.Break("0")
                    elif True:
                        d_36_ag_: _dafny.Seq
                        d_37_ai_: bool
                        d_38_ac_: _dafny.Seq
                        out27_: _dafny.Seq
                        out28_: bool
                        out29_: _dafny.Seq
                        out27_, out28_, out29_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_28_next_)
                        d_36_ag_ = out27_
                        d_37_ai_ = out28_
                        d_38_ac_ = out29_
                        generated = d_36_ag_
                        insideConstrainedOut = d_37_ai_
                        currentConstrainedOut = d_38_ac_
                        d_12_constrainedTokensGenerated_ = (d_12_constrainedTokensGenerated_) + (1)
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_39_closeBudget_: int
            d_39_closeBudget_ = (maxSteps) - (d_1_steps_)
            d_40_cg_: _dafny.Seq
            d_41_ci_: bool
            d_42_cc_: _dafny.Seq
            out30_: _dafny.Seq
            out31_: bool
            out32_: _dafny.Seq
            out30_, out31_, out32_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_39_closeBudget_)
            d_40_cg_ = out30_
            d_41_ci_ = out31_
            d_42_cc_ = out32_
            generated = d_40_cg_
            insideConstrainedOut = d_41_ci_
            currentConstrainedOut = d_42_cc_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

