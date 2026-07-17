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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Translate the question to a valid SQL query. Output format: <<SQL_QUERY>>. Start with << then write complete valid SQL, then >>.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_chunkBudget_: int
        d_2_chunkBudget_ = _dafny.euclidian_division(maxSteps, 2)
        if (d_2_chunkBudget_) == (0):
            d_2_chunkBudget_ = 1
        if (not(insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
            d_3_generatedOut_: _dafny.Seq
            d_4_stoppedOnOpenSpan_: bool
            d_5_stoppedOnEos_: bool
            d_6_stepsUsed_: int
            out0_: _dafny.Seq
            out1_: bool
            out2_: bool
            out3_: int
            out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_2_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
            d_3_generatedOut_ = out0_
            d_4_stoppedOnOpenSpan_ = out1_
            d_5_stoppedOnEos_ = out2_
            d_6_stepsUsed_ = out3_
            generated = d_3_generatedOut_
            d_1_steps_ = (d_1_steps_) + (d_6_stepsUsed_)
            if d_4_stoppedOnOpenSpan_:
                d_7_og_: _dafny.Seq
                d_8_oi_: bool
                d_9_oc_: _dafny.Seq
                out4_: _dafny.Seq
                out5_: bool
                out6_: _dafny.Seq
                out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                d_7_og_ = out4_
                d_8_oi_ = out5_
                d_9_oc_ = out6_
                generated = d_7_og_
                insideConstrainedOut = d_8_oi_
                currentConstrainedOut = d_9_oc_
            elif d_5_stoppedOnEos_:
                cost = d_1_steps_
                return generated, insideConstrainedOut, currentConstrainedOut, cost
            elif True:
                if (d_1_steps_) < (maxSteps):
                    d_10_og_: _dafny.Seq
                    d_11_oi_: bool
                    d_12_oc_: _dafny.Seq
                    out7_: _dafny.Seq
                    out8_: bool
                    out9_: _dafny.Seq
                    out7_, out8_, out9_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                    d_10_og_ = out7_
                    d_11_oi_ = out8_
                    d_12_oc_ = out9_
                    generated = d_10_og_
                    insideConstrainedOut = d_11_oi_
                    currentConstrainedOut = d_12_oc_
                    d_1_steps_ = (d_1_steps_) + (1)
        d_13_minTokensBeforeClose_: int
        d_13_minTokensBeforeClose_ = 5
        d_14_constrainedTokensGenerated_: int
        d_14_constrainedTokensGenerated_ = len(currentConstrainedOut)
        with _dafny.label("0"):
            while ((d_1_steps_) < (maxSteps)) and (insideConstrainedOut):
                with _dafny.c_label("0"):
                    d_15_isDeadEnd_: bool
                    out10_: bool
                    out10_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                    d_15_isDeadEnd_ = out10_
                    if d_15_isDeadEnd_:
                        d_16_rg_: _dafny.Seq
                        d_17_rc_: _dafny.Seq
                        out11_: _dafny.Seq
                        out12_: _dafny.Seq
                        out11_, out12_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                        d_16_rg_ = out11_
                        d_17_rc_ = out12_
                        generated = d_16_rg_
                        currentConstrainedOut = d_17_rc_
                        if (((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_14_constrainedTokensGenerated_) >= (d_13_minTokensBeforeClose_))) and ((d_1_steps_) < (maxSteps)):
                            d_18_cg_: _dafny.Seq
                            d_19_ci_: bool
                            d_20_cc_: _dafny.Seq
                            out13_: _dafny.Seq
                            out14_: bool
                            out15_: _dafny.Seq
                            out13_, out14_, out15_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_18_cg_ = out13_
                            d_19_ci_ = out14_
                            d_20_cc_ = out15_
                            generated = d_18_cg_
                            insideConstrainedOut = d_19_ci_
                            currentConstrainedOut = d_20_cc_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif (d_1_steps_) < (maxSteps):
                            d_21_closeBudget_: int
                            d_21_closeBudget_ = (maxSteps) - (d_1_steps_)
                            d_22_cg2_: _dafny.Seq
                            d_23_ci2_: bool
                            d_24_cc2_: _dafny.Seq
                            out16_: _dafny.Seq
                            out17_: bool
                            out18_: _dafny.Seq
                            out16_, out17_, out18_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_21_closeBudget_)
                            d_22_cg2_ = out16_
                            d_23_ci2_ = out17_
                            d_24_cc2_ = out18_
                            generated = d_22_cg2_
                            insideConstrainedOut = d_23_ci2_
                            currentConstrainedOut = d_24_cc2_
                            d_1_steps_ = maxSteps
                        raise _dafny.Break("0")
                    if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_14_constrainedTokensGenerated_) >= (d_13_minTokensBeforeClose_)):
                        d_25_cg_: _dafny.Seq
                        d_26_ci_: bool
                        d_27_cc_: _dafny.Seq
                        d_28_closed_: bool
                        out19_: _dafny.Seq
                        out20_: bool
                        out21_: _dafny.Seq
                        out22_: bool
                        out19_, out20_, out21_, out22_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                        d_25_cg_ = out19_
                        d_26_ci_ = out20_
                        d_27_cc_ = out21_
                        d_28_closed_ = out22_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if d_28_closed_:
                            generated = d_25_cg_
                            insideConstrainedOut = d_26_ci_
                            currentConstrainedOut = d_27_cc_
                            raise _dafny.Break("0")
                    d_29_constrainedPrompt_: _dafny.Seq
                    d_29_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                    d_30_next_: _dafny.Seq
                    out23_: _dafny.Seq
                    out23_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_29_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 20, eosToken)
                    d_30_next_ = out23_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if (d_30_next_) == (eosToken):
                        if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                            d_31_cg_: _dafny.Seq
                            d_32_ci_: bool
                            d_33_cc_: _dafny.Seq
                            out24_: _dafny.Seq
                            out25_: bool
                            out26_: _dafny.Seq
                            out24_, out25_, out26_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_31_cg_ = out24_
                            d_32_ci_ = out25_
                            d_33_cc_ = out26_
                            generated = d_31_cg_
                            insideConstrainedOut = d_32_ci_
                            currentConstrainedOut = d_33_cc_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif (d_1_steps_) < (maxSteps):
                            d_34_closeBudget_: int
                            d_34_closeBudget_ = (maxSteps) - (d_1_steps_)
                            d_35_cg2_: _dafny.Seq
                            d_36_ci2_: bool
                            d_37_cc2_: _dafny.Seq
                            out27_: _dafny.Seq
                            out28_: bool
                            out29_: _dafny.Seq
                            out27_, out28_, out29_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_34_closeBudget_)
                            d_35_cg2_ = out27_
                            d_36_ci2_ = out28_
                            d_37_cc2_ = out29_
                            generated = d_35_cg2_
                            insideConstrainedOut = d_36_ci2_
                            currentConstrainedOut = d_37_cc2_
                            d_1_steps_ = maxSteps
                        raise _dafny.Break("0")
                    elif True:
                        d_38_ag_: _dafny.Seq
                        d_39_ai_: bool
                        d_40_ac_: _dafny.Seq
                        out30_: _dafny.Seq
                        out31_: bool
                        out32_: _dafny.Seq
                        out30_, out31_, out32_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_30_next_)
                        d_38_ag_ = out30_
                        d_39_ai_ = out31_
                        d_40_ac_ = out32_
                        generated = d_38_ag_
                        insideConstrainedOut = d_39_ai_
                        currentConstrainedOut = d_40_ac_
                        d_14_constrainedTokensGenerated_ = (d_14_constrainedTokensGenerated_) + (1)
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_41_closeBudget_: int
            d_41_closeBudget_ = (maxSteps) - (d_1_steps_)
            d_42_cg_: _dafny.Seq
            d_43_ci_: bool
            d_44_cc_: _dafny.Seq
            out33_: _dafny.Seq
            out34_: bool
            out35_: _dafny.Seq
            out33_, out34_, out35_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_41_closeBudget_)
            d_42_cg_ = out33_
            d_43_ci_ = out34_
            d_44_cc_ = out35_
            generated = d_42_cg_
            insideConstrainedOut = d_43_ci_
            currentConstrainedOut = d_44_cc_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

