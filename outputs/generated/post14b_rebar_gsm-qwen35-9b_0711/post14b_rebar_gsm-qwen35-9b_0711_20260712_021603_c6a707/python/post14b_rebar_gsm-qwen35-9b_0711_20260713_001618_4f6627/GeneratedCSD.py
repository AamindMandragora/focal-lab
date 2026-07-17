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
        if (maxSteps) > (50):
            d_3_preambleLimit_ = (maxSteps) - (50)
        elif True:
            d_3_preambleLimit_ = 0
        if not(insideConstrainedOut):
            while (d_2_steps_) < (d_3_preambleLimit_):
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
        if ((d_2_steps_) < (maxSteps)) and (not(insideConstrainedOut)):
            d_5_og_: _dafny.Seq
            d_6_oi_: bool
            d_7_oc_: _dafny.Seq
            out1_: _dafny.Seq
            out2_: bool
            out3_: _dafny.Seq
            out1_, out2_, out3_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_5_og_ = out1_
            d_6_oi_ = out2_
            d_7_oc_ = out3_
            generated = d_5_og_
            insideConstrainedOut = d_6_oi_
            currentConstrainedOut = d_7_oc_
            d_2_steps_ = (d_2_steps_) + (1)
        d_8_minTokensToGenerate_: int
        d_8_minTokensToGenerate_ = 3
        d_9_preCloseSteps_: int
        d_9_preCloseSteps_ = 0
        while (((d_9_preCloseSteps_) < (d_8_minTokensToGenerate_)) and ((d_2_steps_) < (maxSteps))) and (insideConstrainedOut):
            d_10_constrainedPrompt_: _dafny.Seq
            d_10_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
            d_11_next_: _dafny.Seq
            out4_: _dafny.Seq
            out4_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_10_constrainedPrompt_, currentConstrainedOut, eosToken)
            d_11_next_ = out4_
            d_2_steps_ = (d_2_steps_) + (1)
            d_9_preCloseSteps_ = (d_9_preCloseSteps_) + (1)
            if (d_11_next_) == (eosToken):
                insideConstrainedOut = False
                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
            elif True:
                d_12_ag_: _dafny.Seq
                d_13_ai_: bool
                d_14_ac_: _dafny.Seq
                out5_: _dafny.Seq
                out6_: bool
                out7_: _dafny.Seq
                out5_, out6_, out7_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_11_next_)
                d_12_ag_ = out5_
                d_13_ai_ = out6_
                d_14_ac_ = out7_
                generated = d_12_ag_
                insideConstrainedOut = d_13_ai_
                currentConstrainedOut = d_14_ac_
        while (((d_2_steps_) + (5)) < (maxSteps)) and (insideConstrainedOut):
            d_15_cg_: _dafny.Seq
            d_16_ci_: bool
            d_17_cc_: _dafny.Seq
            d_18_closed_: bool
            out8_: _dafny.Seq
            out9_: bool
            out10_: _dafny.Seq
            out11_: bool
            out8_, out9_, out10_, out11_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
            d_15_cg_ = out8_
            d_16_ci_ = out9_
            d_17_cc_ = out10_
            d_18_closed_ = out11_
            d_2_steps_ = (d_2_steps_) + (1)
            if d_18_closed_:
                generated = d_15_cg_
                insideConstrainedOut = d_16_ci_
                currentConstrainedOut = d_17_cc_
            elif True:
                if (d_2_steps_) < (maxSteps):
                    d_19_constrainedPrompt_: _dafny.Seq
                    d_19_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                    d_20_next_: _dafny.Seq
                    out12_: _dafny.Seq
                    out12_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_19_constrainedPrompt_, currentConstrainedOut, eosToken)
                    d_20_next_ = out12_
                    d_2_steps_ = (d_2_steps_) + (1)
                    if (d_20_next_) == (eosToken):
                        insideConstrainedOut = False
                        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        d_21_ag_: _dafny.Seq
                        d_22_ai_: bool
                        d_23_ac_: _dafny.Seq
                        out13_: _dafny.Seq
                        out14_: bool
                        out15_: _dafny.Seq
                        out13_, out14_, out15_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_20_next_)
                        d_21_ag_ = out13_
                        d_22_ai_ = out14_
                        d_23_ac_ = out15_
                        generated = d_21_ag_
                        insideConstrainedOut = d_22_ai_
                        currentConstrainedOut = d_23_ac_
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_24_closeBudget_: int
            d_24_closeBudget_ = (maxSteps) - (d_2_steps_)
            d_25_cg_: _dafny.Seq
            d_26_ci_: bool
            d_27_cc_: _dafny.Seq
            out16_: _dafny.Seq
            out17_: bool
            out18_: _dafny.Seq
            out16_, out17_, out18_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_24_closeBudget_)
            d_25_cg_ = out16_
            d_26_ci_ = out17_
            d_27_cc_ = out18_
            generated = d_25_cg_
            insideConstrainedOut = d_26_ci_
            currentConstrainedOut = d_27_cc_
            d_2_steps_ = maxSteps
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

