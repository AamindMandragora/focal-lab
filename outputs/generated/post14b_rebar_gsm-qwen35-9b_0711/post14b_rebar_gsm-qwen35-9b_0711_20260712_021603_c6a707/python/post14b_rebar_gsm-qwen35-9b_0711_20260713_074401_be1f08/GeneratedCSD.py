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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the problem step by step. Write your final answer as one symbolic expression inside << >>. Use exact variable names from the problem (the names written in {braces}, without the braces themselves). Do NOT write {variable_names} inside << >>. Use // for integer division when converting time units. Use int() when the result must be an integer. Do not use ** in the expression."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_preambleLimit_: int
        if (maxSteps) > (50):
            d_3_preambleLimit_ = (maxSteps) - (50)
        elif True:
            d_3_preambleLimit_ = 0
        while (d_2_steps_) < (d_3_preambleLimit_):
            if not(insideConstrainedOut):
                d_4_next_: _dafny.Seq
                out0_: _dafny.Seq
                out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                d_4_next_ = out0_
                d_2_steps_ = (d_2_steps_) + (1)
                if (d_4_next_) == (eosToken):
                    cost = d_2_steps_
                    return generated, insideConstrainedOut, currentConstrainedOut, cost
                elif True:
                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_4_next_]))
                    if (d_4_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                        d_5_eg_: _dafny.Seq
                        d_6_ei_: bool
                        d_7_ec_: _dafny.Seq
                        out1_: _dafny.Seq
                        out2_: bool
                        out3_: _dafny.Seq
                        out1_, out2_, out3_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                        d_5_eg_ = out1_
                        d_6_ei_ = out2_
                        d_7_ec_ = out3_
                        generated = d_5_eg_
                        insideConstrainedOut = d_6_ei_
                        currentConstrainedOut = d_7_ec_
            elif True:
                d_8_cg_: _dafny.Seq
                d_9_ci_: bool
                d_10_cc_: _dafny.Seq
                d_11_closed_: bool
                out4_: _dafny.Seq
                out5_: bool
                out6_: _dafny.Seq
                out7_: bool
                out4_, out5_, out6_, out7_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                d_8_cg_ = out4_
                d_9_ci_ = out5_
                d_10_cc_ = out6_
                d_11_closed_ = out7_
                d_2_steps_ = (d_2_steps_) + (1)
                if d_11_closed_:
                    generated = d_8_cg_
                    insideConstrainedOut = d_9_ci_
                    currentConstrainedOut = d_10_cc_
                elif True:
                    if (d_2_steps_) < (maxSteps):
                        d_12_constrainedPrompt_: _dafny.Seq
                        d_12_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_13_next_: _dafny.Seq
                        out8_: _dafny.Seq
                        out8_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_12_constrainedPrompt_, currentConstrainedOut, eosToken)
                        d_13_next_ = out8_
                        d_2_steps_ = (d_2_steps_) + (1)
                        if (d_13_next_) == (eosToken):
                            insideConstrainedOut = False
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                            cost = d_2_steps_
                            return generated, insideConstrainedOut, currentConstrainedOut, cost
                        elif True:
                            d_14_ag_: _dafny.Seq
                            d_15_ai_: bool
                            d_16_ac_: _dafny.Seq
                            out9_: _dafny.Seq
                            out10_: bool
                            out11_: _dafny.Seq
                            out9_, out10_, out11_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_13_next_)
                            d_14_ag_ = out9_
                            d_15_ai_ = out10_
                            d_16_ac_ = out11_
                            generated = d_14_ag_
                            insideConstrainedOut = d_15_ai_
                            currentConstrainedOut = d_16_ac_
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_17_closeBudget1_: int
            if ((maxSteps) - (d_2_steps_)) > (20):
                d_17_closeBudget1_ = 20
            elif True:
                d_17_closeBudget1_ = (maxSteps) - (d_2_steps_)
            d_18_cg1_: _dafny.Seq
            d_19_ci1_: bool
            d_20_cc1_: _dafny.Seq
            out12_: _dafny.Seq
            out13_: bool
            out14_: _dafny.Seq
            out12_, out13_, out14_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_17_closeBudget1_)
            d_18_cg1_ = out12_
            d_19_ci1_ = out13_
            d_20_cc1_ = out14_
            generated = d_18_cg1_
            insideConstrainedOut = d_19_ci1_
            currentConstrainedOut = d_20_cc1_
            d_2_steps_ = (d_2_steps_) + (d_17_closeBudget1_)
        if ((d_2_steps_) < (maxSteps)) and (not(insideConstrainedOut)):
            d_21_og_: _dafny.Seq
            d_22_oi_: bool
            d_23_oc_: _dafny.Seq
            out15_: _dafny.Seq
            out16_: bool
            out17_: _dafny.Seq
            out15_, out16_, out17_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_21_og_ = out15_
            d_22_oi_ = out16_
            d_23_oc_ = out17_
            generated = d_21_og_
            insideConstrainedOut = d_22_oi_
            currentConstrainedOut = d_23_oc_
            d_2_steps_ = (d_2_steps_) + (1)
        d_24_minTokens_: int
        d_24_minTokens_ = 3
        d_25_preSteps_: int
        d_25_preSteps_ = 0
        while (((d_25_preSteps_) < (d_24_minTokens_)) and ((d_2_steps_) < (maxSteps))) and (insideConstrainedOut):
            d_26_constrainedPrompt_: _dafny.Seq
            d_26_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
            d_27_next_: _dafny.Seq
            out18_: _dafny.Seq
            out18_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_26_constrainedPrompt_, currentConstrainedOut, eosToken)
            d_27_next_ = out18_
            d_2_steps_ = (d_2_steps_) + (1)
            d_25_preSteps_ = (d_25_preSteps_) + (1)
            if (d_27_next_) == (eosToken):
                insideConstrainedOut = False
                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
            elif True:
                d_28_ag_: _dafny.Seq
                d_29_ai_: bool
                d_30_ac_: _dafny.Seq
                out19_: _dafny.Seq
                out20_: bool
                out21_: _dafny.Seq
                out19_, out20_, out21_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_27_next_)
                d_28_ag_ = out19_
                d_29_ai_ = out20_
                d_30_ac_ = out21_
                generated = d_28_ag_
                insideConstrainedOut = d_29_ai_
                currentConstrainedOut = d_30_ac_
        while (((d_2_steps_) + (1)) < (maxSteps)) and (insideConstrainedOut):
            d_31_cg_: _dafny.Seq
            d_32_ci_: bool
            d_33_cc_: _dafny.Seq
            d_34_closed_: bool
            out22_: _dafny.Seq
            out23_: bool
            out24_: _dafny.Seq
            out25_: bool
            out22_, out23_, out24_, out25_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
            d_31_cg_ = out22_
            d_32_ci_ = out23_
            d_33_cc_ = out24_
            d_34_closed_ = out25_
            d_2_steps_ = (d_2_steps_) + (1)
            if d_34_closed_:
                generated = d_31_cg_
                insideConstrainedOut = d_32_ci_
                currentConstrainedOut = d_33_cc_
            elif True:
                if (d_2_steps_) < (maxSteps):
                    d_35_constrainedPrompt_: _dafny.Seq
                    d_35_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                    d_36_next_: _dafny.Seq
                    out26_: _dafny.Seq
                    out26_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_35_constrainedPrompt_, currentConstrainedOut, eosToken)
                    d_36_next_ = out26_
                    d_2_steps_ = (d_2_steps_) + (1)
                    if (d_36_next_) == (eosToken):
                        insideConstrainedOut = False
                        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        d_37_ag_: _dafny.Seq
                        d_38_ai_: bool
                        d_39_ac_: _dafny.Seq
                        out27_: _dafny.Seq
                        out28_: bool
                        out29_: _dafny.Seq
                        out27_, out28_, out29_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_36_next_)
                        d_37_ag_ = out27_
                        d_38_ai_ = out28_
                        d_39_ac_ = out29_
                        generated = d_37_ag_
                        insideConstrainedOut = d_38_ai_
                        currentConstrainedOut = d_39_ac_
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_40_closeBudget_: int
            d_40_closeBudget_ = (maxSteps) - (d_2_steps_)
            d_41_cg_: _dafny.Seq
            d_42_ci_: bool
            d_43_cc_: _dafny.Seq
            out30_: _dafny.Seq
            out31_: bool
            out32_: _dafny.Seq
            out30_, out31_, out32_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_40_closeBudget_)
            d_41_cg_ = out30_
            d_42_ci_ = out31_
            d_43_cc_ = out32_
            generated = d_41_cg_
            insideConstrainedOut = d_42_ci_
            currentConstrainedOut = d_43_cc_
            d_2_steps_ = maxSteps
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

