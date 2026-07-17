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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Output a valid SQL query. Begin your response with << immediately, then write the complete SQL query, then >>. Do not output anything after >>.")))
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
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        raise _dafny.Break("0")
                    if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_15_constrainedTokensGenerated_) >= (d_14_minTokensBeforeClose_)):
                        d_16_cg_: _dafny.Seq
                        d_17_ci_: bool
                        d_18_cc_: _dafny.Seq
                        d_19_closed_: bool
                        out11_: _dafny.Seq
                        out12_: bool
                        out13_: _dafny.Seq
                        out14_: bool
                        out11_, out12_, out13_, out14_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                        d_16_cg_ = out11_
                        d_17_ci_ = out12_
                        d_18_cc_ = out13_
                        d_19_closed_ = out14_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if d_19_closed_:
                            generated = d_16_cg_
                            insideConstrainedOut = d_17_ci_
                            currentConstrainedOut = d_18_cc_
                            raise _dafny.Break("0")
                        if (d_1_steps_) >= (maxSteps):
                            raise _dafny.Break("0")
                    d_20_constrainedPrompt_: _dafny.Seq
                    d_20_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                    d_21_next_: _dafny.Seq
                    out15_: _dafny.Seq
                    out15_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_20_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 20, eosToken)
                    d_21_next_ = out15_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if (d_21_next_) == (eosToken):
                        if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                            d_22_cg_: _dafny.Seq
                            d_23_ci_: bool
                            d_24_cc_: _dafny.Seq
                            out16_: _dafny.Seq
                            out17_: bool
                            out18_: _dafny.Seq
                            out16_, out17_, out18_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_22_cg_ = out16_
                            d_23_ci_ = out17_
                            d_24_cc_ = out18_
                            generated = d_22_cg_
                            insideConstrainedOut = d_23_ci_
                            currentConstrainedOut = d_24_cc_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif (d_1_steps_) < (maxSteps):
                            d_25_closeBudget_: int
                            d_25_closeBudget_ = (maxSteps) - (d_1_steps_)
                            d_26_cg2_: _dafny.Seq
                            d_27_ci2_: bool
                            d_28_cc2_: _dafny.Seq
                            out19_: _dafny.Seq
                            out20_: bool
                            out21_: _dafny.Seq
                            out19_, out20_, out21_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_25_closeBudget_)
                            d_26_cg2_ = out19_
                            d_27_ci2_ = out20_
                            d_28_cc2_ = out21_
                            generated = d_26_cg2_
                            insideConstrainedOut = d_27_ci2_
                            currentConstrainedOut = d_28_cc2_
                            d_1_steps_ = maxSteps
                        raise _dafny.Break("0")
                    elif True:
                        d_29_ag_: _dafny.Seq
                        d_30_ai_: bool
                        d_31_ac_: _dafny.Seq
                        out22_: _dafny.Seq
                        out23_: bool
                        out24_: _dafny.Seq
                        out22_, out23_, out24_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_21_next_)
                        d_29_ag_ = out22_
                        d_30_ai_ = out23_
                        d_31_ac_ = out24_
                        generated = d_29_ag_
                        insideConstrainedOut = d_30_ai_
                        currentConstrainedOut = d_31_ac_
                        d_15_constrainedTokensGenerated_ = (d_15_constrainedTokensGenerated_) + (1)
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_32_closeBudget_: int
            d_32_closeBudget_ = (maxSteps) - (d_1_steps_)
            d_33_cg_: _dafny.Seq
            d_34_ci_: bool
            d_35_cc_: _dafny.Seq
            out25_: _dafny.Seq
            out26_: bool
            out27_: _dafny.Seq
            out25_, out26_, out27_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_32_closeBudget_)
            d_33_cg_ = out25_
            d_34_ci_ = out26_
            d_35_cc_ = out27_
            generated = d_33_cg_
            insideConstrainedOut = d_34_ci_
            currentConstrainedOut = d_35_cc_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

