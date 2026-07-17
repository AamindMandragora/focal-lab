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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Write a complete SQL SELECT query that answers the natural language question. Use exact table and column names from the database schema. For counting use COUNT(*) or COUNT(col). For aggregation use GROUP BY. For ordering use ORDER BY. For multiple tables use JOIN ... ON. For top-N use LIMIT N. Output only the SQL query.")))
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
        d_5_minTokensBeforeClose_ = 20
        d_6_constrainedTokensGenerated_: int
        d_6_constrainedTokensGenerated_ = len(currentConstrainedOut)
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        raise _dafny.Break("0")
                    if ((d_1_steps_) + (3)) >= (maxSteps):
                        if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                            d_7_cg_: _dafny.Seq
                            d_8_ci_: bool
                            d_9_cc_: _dafny.Seq
                            out3_: _dafny.Seq
                            out4_: bool
                            out5_: _dafny.Seq
                            out3_, out4_, out5_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_7_cg_ = out3_
                            d_8_ci_ = out4_
                            d_9_cc_ = out5_
                            generated = d_7_cg_
                            insideConstrainedOut = d_8_ci_
                            currentConstrainedOut = d_9_cc_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif (d_1_steps_) < (maxSteps):
                            d_10_closeBudget_: int
                            d_10_closeBudget_ = (maxSteps) - (d_1_steps_)
                            d_11_cg2_: _dafny.Seq
                            d_12_ci2_: bool
                            d_13_cc2_: _dafny.Seq
                            out6_: _dafny.Seq
                            out7_: bool
                            out8_: _dafny.Seq
                            out6_, out7_, out8_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_10_closeBudget_)
                            d_11_cg2_ = out6_
                            d_12_ci2_ = out7_
                            d_13_cc2_ = out8_
                            generated = d_11_cg2_
                            insideConstrainedOut = d_12_ci2_
                            currentConstrainedOut = d_13_cc2_
                            d_1_steps_ = maxSteps
                        raise _dafny.Break("0")
                    d_14_isDeadEnd_: bool
                    out9_: bool
                    out9_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                    d_14_isDeadEnd_ = out9_
                    if d_14_isDeadEnd_:
                        d_15_rg_: _dafny.Seq
                        d_16_rc_: _dafny.Seq
                        out10_: _dafny.Seq
                        out11_: _dafny.Seq
                        out10_, out11_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                        d_15_rg_ = out10_
                        d_16_rc_ = out11_
                        generated = d_15_rg_
                        currentConstrainedOut = d_16_rc_
                        if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                            d_17_cg_: _dafny.Seq
                            d_18_ci_: bool
                            d_19_cc_: _dafny.Seq
                            out12_: _dafny.Seq
                            out13_: bool
                            out14_: _dafny.Seq
                            out12_, out13_, out14_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_17_cg_ = out12_
                            d_18_ci_ = out13_
                            d_19_cc_ = out14_
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
                            out15_: _dafny.Seq
                            out16_: bool
                            out17_: _dafny.Seq
                            out15_, out16_, out17_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_20_closeBudget_)
                            d_21_cg2_ = out15_
                            d_22_ci2_ = out16_
                            d_23_cc2_ = out17_
                            generated = d_21_cg2_
                            insideConstrainedOut = d_22_ci2_
                            currentConstrainedOut = d_23_cc2_
                            d_1_steps_ = maxSteps
                        raise _dafny.Break("0")
                    if (d_6_constrainedTokensGenerated_) > (50):
                        (d_0_helpers_).MaskTokensInPrefix(lm, currentConstrainedOut)
                    if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_6_constrainedTokensGenerated_) >= (d_5_minTokensBeforeClose_)):
                        d_24_cg_: _dafny.Seq
                        d_25_ci_: bool
                        d_26_cc_: _dafny.Seq
                        d_27_closed_: bool
                        out18_: _dafny.Seq
                        out19_: bool
                        out20_: _dafny.Seq
                        out21_: bool
                        out18_, out19_, out20_, out21_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                        d_24_cg_ = out18_
                        d_25_ci_ = out19_
                        d_26_cc_ = out20_
                        d_27_closed_ = out21_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if d_27_closed_:
                            generated = d_24_cg_
                            insideConstrainedOut = d_25_ci_
                            currentConstrainedOut = d_26_cc_
                            raise _dafny.Break("0")
                        if (d_1_steps_) >= (maxSteps):
                            raise _dafny.Break("0")
                    d_28_validCount_: int
                    out22_: int
                    out22_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                    d_28_validCount_ = out22_
                    d_29_next_: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                    if (d_28_validCount_) <= (5):
                        out23_: _dafny.Seq
                        out23_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, prompt, currentConstrainedOut, validTokenGroups, _dafny.BigRational('8e0'), 5, eosToken)
                        d_29_next_ = out23_
                    elif (d_28_validCount_) <= (20):
                        out24_: _dafny.Seq
                        out24_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, prompt, currentConstrainedOut, validTokenGroups, _dafny.BigRational('5e0'), 20, eosToken)
                        d_29_next_ = out24_
                    elif True:
                        out25_: _dafny.Seq
                        out25_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, prompt, currentConstrainedOut, validTokenGroups, _dafny.BigRational('3e0'), 40, eosToken)
                        d_29_next_ = out25_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if (d_29_next_) == (eosToken):
                        if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                            d_30_cg_: _dafny.Seq
                            d_31_ci_: bool
                            d_32_cc_: _dafny.Seq
                            out26_: _dafny.Seq
                            out27_: bool
                            out28_: _dafny.Seq
                            out26_, out27_, out28_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_30_cg_ = out26_
                            d_31_ci_ = out27_
                            d_32_cc_ = out28_
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
                            out29_: _dafny.Seq
                            out30_: bool
                            out31_: _dafny.Seq
                            out29_, out30_, out31_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_33_closeBudget_)
                            d_34_cg2_ = out29_
                            d_35_ci2_ = out30_
                            d_36_cc2_ = out31_
                            generated = d_34_cg2_
                            insideConstrainedOut = d_35_ci2_
                            currentConstrainedOut = d_36_cc2_
                            d_1_steps_ = maxSteps
                        raise _dafny.Break("0")
                    elif True:
                        d_37_ag_: _dafny.Seq
                        d_38_ai_: bool
                        d_39_ac_: _dafny.Seq
                        out32_: _dafny.Seq
                        out33_: bool
                        out34_: _dafny.Seq
                        out32_, out33_, out34_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_29_next_)
                        d_37_ag_ = out32_
                        d_38_ai_ = out33_
                        d_39_ac_ = out34_
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
            out35_: _dafny.Seq
            out36_: bool
            out37_: _dafny.Seq
            out35_, out36_, out37_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_40_closeBudget_)
            d_41_cg_ = out35_
            d_42_ci_ = out36_
            d_43_cc_ = out37_
            generated = d_41_cg_
            insideConstrainedOut = d_42_ci_
            currentConstrainedOut = d_43_cc_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

