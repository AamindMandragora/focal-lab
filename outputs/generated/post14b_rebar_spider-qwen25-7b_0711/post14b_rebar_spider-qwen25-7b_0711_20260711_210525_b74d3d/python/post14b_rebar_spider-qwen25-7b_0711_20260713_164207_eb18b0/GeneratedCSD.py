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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate a valid SQL query. Write simple, concise SQL. Use subqueries for aggregation comparisons. Use INTERSECT or UNION for set operations. Avoid unnecessary JOINs. Output ONLY the SQL query between << and >>.")))
        d_1_steps_: int
        d_1_steps_ = 0
        if (not(insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
            (d_0_helpers_).SafeBoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]), _dafny.BigRational('2e1'))
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
                while (((d_6_attempt_) < (3)) and ((d_1_steps_) < (maxSteps))) and (not(insideConstrainedOut)):
                    (d_0_helpers_).SafeBoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]), _dafny.BigRational('2e1'))
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
        d_14_minTokensBeforeClose_ = 10
        d_15_constrainedTokensGenerated_: int
        d_15_constrainedTokensGenerated_ = len(currentConstrainedOut)
        d_16_lastToken_: _dafny.Seq
        d_16_lastToken_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
        d_17_consecutiveRepeats_: int
        d_17_consecutiveRepeats_ = 0
        d_18_maxConstrainedTokens_: int
        d_18_maxConstrainedTokens_ = 120
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        raise _dafny.Break("0")
                    if (d_17_consecutiveRepeats_) >= (3):
                        d_19_rg_: _dafny.Seq
                        d_20_rc_: _dafny.Seq
                        out11_: _dafny.Seq
                        out12_: _dafny.Seq
                        out11_, out12_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                        d_19_rg_ = out11_
                        d_20_rc_ = out12_
                        generated = d_19_rg_
                        currentConstrainedOut = d_20_rc_
                        d_17_consecutiveRepeats_ = 0
                        if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                            d_21_cg_: _dafny.Seq
                            d_22_ci_: bool
                            d_23_cc_: _dafny.Seq
                            out13_: _dafny.Seq
                            out14_: bool
                            out15_: _dafny.Seq
                            out13_, out14_, out15_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_21_cg_ = out13_
                            d_22_ci_ = out14_
                            d_23_cc_ = out15_
                            generated = d_21_cg_
                            insideConstrainedOut = d_22_ci_
                            currentConstrainedOut = d_23_cc_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif (d_1_steps_) < (maxSteps):
                            d_24_closeBudget_: int
                            d_24_closeBudget_ = (maxSteps) - (d_1_steps_)
                            d_25_cg2_: _dafny.Seq
                            d_26_ci2_: bool
                            d_27_cc2_: _dafny.Seq
                            out16_: _dafny.Seq
                            out17_: bool
                            out18_: _dafny.Seq
                            out16_, out17_, out18_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_24_closeBudget_)
                            d_25_cg2_ = out16_
                            d_26_ci2_ = out17_
                            d_27_cc2_ = out18_
                            generated = d_25_cg2_
                            insideConstrainedOut = d_26_ci2_
                            currentConstrainedOut = d_27_cc2_
                            d_1_steps_ = maxSteps
                        raise _dafny.Break("0")
                    if ((d_15_constrainedTokensGenerated_) >= (d_18_maxConstrainedTokens_)) and ((d_1_steps_) < (maxSteps)):
                        d_28_rg_: _dafny.Seq
                        d_29_rc_: _dafny.Seq
                        out19_: _dafny.Seq
                        out20_: _dafny.Seq
                        out19_, out20_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                        d_28_rg_ = out19_
                        d_29_rc_ = out20_
                        generated = d_28_rg_
                        currentConstrainedOut = d_29_rc_
                        if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                            d_30_cg_: _dafny.Seq
                            d_31_ci_: bool
                            d_32_cc_: _dafny.Seq
                            out21_: _dafny.Seq
                            out22_: bool
                            out23_: _dafny.Seq
                            out21_, out22_, out23_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_30_cg_ = out21_
                            d_31_ci_ = out22_
                            d_32_cc_ = out23_
                            generated = d_30_cg_
                            insideConstrainedOut = d_31_ci_
                            currentConstrainedOut = d_32_cc_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif (d_1_steps_) < (maxSteps):
                            d_33_closeBudget_: int
                            d_33_closeBudget_ = (maxSteps) - (d_1_steps_)
                            d_34_cg2_: _dafny.Seq
                            d_35_ci2_: bool
                            d_36_cc2_: _dafny.Seq
                            out24_: _dafny.Seq
                            out25_: bool
                            out26_: _dafny.Seq
                            out24_, out25_, out26_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_33_closeBudget_)
                            d_34_cg2_ = out24_
                            d_35_ci2_ = out25_
                            d_36_cc2_ = out26_
                            generated = d_34_cg2_
                            insideConstrainedOut = d_35_ci2_
                            currentConstrainedOut = d_36_cc2_
                            d_1_steps_ = maxSteps
                        raise _dafny.Break("0")
                    d_37_isDeadEnd_: bool
                    out27_: bool
                    out27_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                    d_37_isDeadEnd_ = out27_
                    if d_37_isDeadEnd_:
                        d_38_rg_: _dafny.Seq
                        d_39_rc_: _dafny.Seq
                        out28_: _dafny.Seq
                        out29_: _dafny.Seq
                        out28_, out29_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                        d_38_rg_ = out28_
                        d_39_rc_ = out29_
                        generated = d_38_rg_
                        currentConstrainedOut = d_39_rc_
                        d_17_consecutiveRepeats_ = 0
                        if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                            d_40_cg_: _dafny.Seq
                            d_41_ci_: bool
                            d_42_cc_: _dafny.Seq
                            out30_: _dafny.Seq
                            out31_: bool
                            out32_: _dafny.Seq
                            out30_, out31_, out32_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_40_cg_ = out30_
                            d_41_ci_ = out31_
                            d_42_cc_ = out32_
                            generated = d_40_cg_
                            insideConstrainedOut = d_41_ci_
                            currentConstrainedOut = d_42_cc_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif (d_1_steps_) < (maxSteps):
                            d_43_closeBudget_: int
                            d_43_closeBudget_ = (maxSteps) - (d_1_steps_)
                            d_44_cg2_: _dafny.Seq
                            d_45_ci2_: bool
                            d_46_cc2_: _dafny.Seq
                            out33_: _dafny.Seq
                            out34_: bool
                            out35_: _dafny.Seq
                            out33_, out34_, out35_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_43_closeBudget_)
                            d_44_cg2_ = out33_
                            d_45_ci2_ = out34_
                            d_46_cc2_ = out35_
                            generated = d_44_cg2_
                            insideConstrainedOut = d_45_ci2_
                            currentConstrainedOut = d_46_cc2_
                            d_1_steps_ = maxSteps
                        raise _dafny.Break("0")
                    if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_15_constrainedTokensGenerated_) >= (d_14_minTokensBeforeClose_)):
                        d_47_cg_: _dafny.Seq
                        d_48_ci_: bool
                        d_49_cc_: _dafny.Seq
                        d_50_closed_: bool
                        out36_: _dafny.Seq
                        out37_: bool
                        out38_: _dafny.Seq
                        out39_: bool
                        out36_, out37_, out38_, out39_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                        d_47_cg_ = out36_
                        d_48_ci_ = out37_
                        d_49_cc_ = out38_
                        d_50_closed_ = out39_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if d_50_closed_:
                            generated = d_47_cg_
                            insideConstrainedOut = d_48_ci_
                            currentConstrainedOut = d_49_cc_
                            raise _dafny.Break("0")
                        if (d_1_steps_) >= (maxSteps):
                            raise _dafny.Break("0")
                    d_51_constrainedPrompt_: _dafny.Seq
                    d_51_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                    d_52_next_: _dafny.Seq
                    out40_: _dafny.Seq
                    out40_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_51_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 20, eosToken)
                    d_52_next_ = out40_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if (d_52_next_) == (eosToken):
                        if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                            d_53_cg_: _dafny.Seq
                            d_54_ci_: bool
                            d_55_cc_: _dafny.Seq
                            out41_: _dafny.Seq
                            out42_: bool
                            out43_: _dafny.Seq
                            out41_, out42_, out43_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_53_cg_ = out41_
                            d_54_ci_ = out42_
                            d_55_cc_ = out43_
                            generated = d_53_cg_
                            insideConstrainedOut = d_54_ci_
                            currentConstrainedOut = d_55_cc_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif (d_1_steps_) < (maxSteps):
                            d_56_closeBudget_: int
                            d_56_closeBudget_ = (maxSteps) - (d_1_steps_)
                            d_57_cg2_: _dafny.Seq
                            d_58_ci2_: bool
                            d_59_cc2_: _dafny.Seq
                            out44_: _dafny.Seq
                            out45_: bool
                            out46_: _dafny.Seq
                            out44_, out45_, out46_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_56_closeBudget_)
                            d_57_cg2_ = out44_
                            d_58_ci2_ = out45_
                            d_59_cc2_ = out46_
                            generated = d_57_cg2_
                            insideConstrainedOut = d_58_ci2_
                            currentConstrainedOut = d_59_cc2_
                            d_1_steps_ = maxSteps
                        raise _dafny.Break("0")
                    elif True:
                        if (d_52_next_) == (d_16_lastToken_):
                            d_17_consecutiveRepeats_ = (d_17_consecutiveRepeats_) + (1)
                        elif True:
                            d_16_lastToken_ = d_52_next_
                            d_17_consecutiveRepeats_ = 0
                        d_60_ag_: _dafny.Seq
                        d_61_ai_: bool
                        d_62_ac_: _dafny.Seq
                        out47_: _dafny.Seq
                        out48_: bool
                        out49_: _dafny.Seq
                        out47_, out48_, out49_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_52_next_)
                        d_60_ag_ = out47_
                        d_61_ai_ = out48_
                        d_62_ac_ = out49_
                        generated = d_60_ag_
                        insideConstrainedOut = d_61_ai_
                        currentConstrainedOut = d_62_ac_
                        d_15_constrainedTokensGenerated_ = (d_15_constrainedTokensGenerated_) + (1)
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_63_closeBudget_: int
            d_63_closeBudget_ = (maxSteps) - (d_1_steps_)
            d_64_cg_: _dafny.Seq
            d_65_ci_: bool
            d_66_cc_: _dafny.Seq
            out50_: _dafny.Seq
            out51_: bool
            out52_: _dafny.Seq
            out50_, out51_, out52_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_63_closeBudget_)
            d_64_cg_ = out50_
            d_65_ci_ = out51_
            d_66_cc_ = out52_
            generated = d_64_cg_
            insideConstrainedOut = d_65_ci_
            currentConstrainedOut = d_66_cc_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

