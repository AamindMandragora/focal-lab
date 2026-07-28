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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the problem step by step. For your final answer, write one symbolic expression inside << >>. Use exact variable names from the problem (the text inside {} braces, without the braces). Use integer division // (not /) when dividing time units like minutes to hours. Use int() around the final expression when computing monetary amounts or integer results. Do not use ** (use * instead). Do not use {} braces in the expression."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_preambleLimit_: int
        if (maxSteps) > (100):
            d_3_preambleLimit_ = (maxSteps) - (100)
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
                    if (d_2_steps_) < (d_3_preambleLimit_):
                        d_12_constrainedPrompt_: _dafny.Seq
                        d_12_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_13_next_: _dafny.Seq
                        out8_: _dafny.Seq
                        out8_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_12_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('15e-1'), eosToken)
                        d_13_next_ = out8_
                        d_2_steps_ = (d_2_steps_) + (1)
                        if (d_13_next_) == (eosToken):
                            insideConstrainedOut = False
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
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
        if ((d_2_steps_) < (maxSteps)) and (not(insideConstrainedOut)):
            d_17_og_: _dafny.Seq
            d_18_oi_: bool
            d_19_oc_: _dafny.Seq
            out12_: _dafny.Seq
            out13_: bool
            out14_: _dafny.Seq
            out12_, out13_, out14_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_17_og_ = out12_
            d_18_oi_ = out13_
            d_19_oc_ = out14_
            generated = d_17_og_
            insideConstrainedOut = d_18_oi_
            currentConstrainedOut = d_19_oc_
            d_2_steps_ = (d_2_steps_) + (1)
        d_20_minTokensToGenerate_: int
        d_20_minTokensToGenerate_ = 3
        d_21_preCloseSteps_: int
        d_21_preCloseSteps_ = 0
        while (((d_21_preCloseSteps_) < (d_20_minTokensToGenerate_)) and ((d_2_steps_) < (maxSteps))) and (insideConstrainedOut):
            d_22_constrainedPrompt_: _dafny.Seq
            d_22_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
            d_23_next_: _dafny.Seq
            out15_: _dafny.Seq
            out15_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_22_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('15e-1'), eosToken)
            d_23_next_ = out15_
            d_2_steps_ = (d_2_steps_) + (1)
            d_21_preCloseSteps_ = (d_21_preCloseSteps_) + (1)
            if (d_23_next_) == (eosToken):
                insideConstrainedOut = False
                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
            elif True:
                d_24_ag_: _dafny.Seq
                d_25_ai_: bool
                d_26_ac_: _dafny.Seq
                out16_: _dafny.Seq
                out17_: bool
                out18_: _dafny.Seq
                out16_, out17_, out18_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_23_next_)
                d_24_ag_ = out16_
                d_25_ai_ = out17_
                d_26_ac_ = out18_
                generated = d_24_ag_
                insideConstrainedOut = d_25_ai_
                currentConstrainedOut = d_26_ac_
        while (((d_2_steps_) + (1)) < (maxSteps)) and (insideConstrainedOut):
            d_27_cg_: _dafny.Seq
            d_28_ci_: bool
            d_29_cc_: _dafny.Seq
            d_30_closed_: bool
            out19_: _dafny.Seq
            out20_: bool
            out21_: _dafny.Seq
            out22_: bool
            out19_, out20_, out21_, out22_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
            d_27_cg_ = out19_
            d_28_ci_ = out20_
            d_29_cc_ = out21_
            d_30_closed_ = out22_
            d_2_steps_ = (d_2_steps_) + (1)
            if d_30_closed_:
                generated = d_27_cg_
                insideConstrainedOut = d_28_ci_
                currentConstrainedOut = d_29_cc_
            elif True:
                if (d_2_steps_) < (maxSteps):
                    d_31_constrainedPrompt_: _dafny.Seq
                    d_31_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                    d_32_next_: _dafny.Seq
                    out23_: _dafny.Seq
                    out23_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_31_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('15e-1'), eosToken)
                    d_32_next_ = out23_
                    d_2_steps_ = (d_2_steps_) + (1)
                    if (d_32_next_) == (eosToken):
                        insideConstrainedOut = False
                        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        d_33_ag_: _dafny.Seq
                        d_34_ai_: bool
                        d_35_ac_: _dafny.Seq
                        out24_: _dafny.Seq
                        out25_: bool
                        out26_: _dafny.Seq
                        out24_, out25_, out26_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_32_next_)
                        d_33_ag_ = out24_
                        d_34_ai_ = out25_
                        d_35_ac_ = out26_
                        generated = d_33_ag_
                        insideConstrainedOut = d_34_ai_
                        currentConstrainedOut = d_35_ac_
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_36_closeBudget_: int
            d_36_closeBudget_ = (maxSteps) - (d_2_steps_)
            d_37_cg_: _dafny.Seq
            d_38_ci_: bool
            d_39_cc_: _dafny.Seq
            out27_: _dafny.Seq
            out28_: bool
            out29_: _dafny.Seq
            out27_, out28_, out29_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_36_closeBudget_)
            d_37_cg_ = out27_
            d_38_ci_ = out28_
            d_39_cc_ = out29_
            generated = d_37_cg_
            insideConstrainedOut = d_38_ci_
            currentConstrainedOut = d_39_cc_
            d_2_steps_ = maxSteps
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

