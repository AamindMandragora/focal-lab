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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Translate the natural language question to a complete and valid SQL query. Output only the SQL query with no explanation.")))
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
        d_5_minTokensBeforeClose_ = 8
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
                        if (((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_6_constrainedTokensGenerated_) >= (d_5_minTokensBeforeClose_))) and ((d_1_steps_) < (maxSteps)):
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
                    if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_6_constrainedTokensGenerated_) >= (d_5_minTokensBeforeClose_)):
                        d_17_cg_: _dafny.Seq
                        d_18_ci_: bool
                        d_19_cc_: _dafny.Seq
                        d_20_closed_: bool
                        out12_: _dafny.Seq
                        out13_: bool
                        out14_: _dafny.Seq
                        out15_: bool
                        out12_, out13_, out14_, out15_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                        d_17_cg_ = out12_
                        d_18_ci_ = out13_
                        d_19_cc_ = out14_
                        d_20_closed_ = out15_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if d_20_closed_:
                            generated = d_17_cg_
                            insideConstrainedOut = d_18_ci_
                            currentConstrainedOut = d_19_cc_
                            raise _dafny.Break("0")
                    d_21_constrainedPrompt_: _dafny.Seq
                    d_21_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                    d_22_next_: _dafny.Seq
                    out16_: _dafny.Seq
                    out16_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_21_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 20, eosToken)
                    d_22_next_ = out16_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if (d_22_next_) == (eosToken):
                        if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                            d_23_cg_: _dafny.Seq
                            d_24_ci_: bool
                            d_25_cc_: _dafny.Seq
                            out17_: _dafny.Seq
                            out18_: bool
                            out19_: _dafny.Seq
                            out17_, out18_, out19_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_23_cg_ = out17_
                            d_24_ci_ = out18_
                            d_25_cc_ = out19_
                            generated = d_23_cg_
                            insideConstrainedOut = d_24_ci_
                            currentConstrainedOut = d_25_cc_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif (d_1_steps_) < (maxSteps):
                            d_26_closeBudget_: int
                            d_26_closeBudget_ = (maxSteps) - (d_1_steps_)
                            d_27_cg2_: _dafny.Seq
                            d_28_ci2_: bool
                            d_29_cc2_: _dafny.Seq
                            out20_: _dafny.Seq
                            out21_: bool
                            out22_: _dafny.Seq
                            out20_, out21_, out22_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_26_closeBudget_)
                            d_27_cg2_ = out20_
                            d_28_ci2_ = out21_
                            d_29_cc2_ = out22_
                            generated = d_27_cg2_
                            insideConstrainedOut = d_28_ci2_
                            currentConstrainedOut = d_29_cc2_
                            d_1_steps_ = maxSteps
                        raise _dafny.Break("0")
                    elif True:
                        d_30_ag_: _dafny.Seq
                        d_31_ai_: bool
                        d_32_ac_: _dafny.Seq
                        out23_: _dafny.Seq
                        out24_: bool
                        out25_: _dafny.Seq
                        out23_, out24_, out25_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_22_next_)
                        d_30_ag_ = out23_
                        d_31_ai_ = out24_
                        d_32_ac_ = out25_
                        generated = d_30_ag_
                        insideConstrainedOut = d_31_ai_
                        currentConstrainedOut = d_32_ac_
                        d_6_constrainedTokensGenerated_ = (d_6_constrainedTokensGenerated_) + (1)
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_33_closeBudget_: int
            d_33_closeBudget_ = (maxSteps) - (d_1_steps_)
            d_34_cg_: _dafny.Seq
            d_35_ci_: bool
            d_36_cc_: _dafny.Seq
            out26_: _dafny.Seq
            out27_: bool
            out28_: _dafny.Seq
            out26_, out27_, out28_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_33_closeBudget_)
            d_34_cg_ = out26_
            d_35_ci_ = out27_
            d_36_cc_ = out28_
            generated = d_34_cg_
            insideConstrainedOut = d_35_ci_
            currentConstrainedOut = d_36_cc_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

