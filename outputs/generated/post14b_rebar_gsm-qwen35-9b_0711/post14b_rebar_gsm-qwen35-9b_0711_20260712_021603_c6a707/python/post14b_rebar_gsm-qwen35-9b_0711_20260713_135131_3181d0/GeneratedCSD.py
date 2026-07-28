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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the problem step by step. Write your final answer using exactly this format: <<expression>>. Use the exact variable names from the problem (the words inside { } in the problem text, WITHOUT the curly braces). NEVER use { or } inside << >>. Use // for integer division when the result must be a whole number (like converting minutes to hours). Use int() around the expression ONLY for integer counts of objects or monetary amounts - do NOT use int() for exponential growth formulas. For exponential growth use ** for exponentiation (e.g. n * (r+1)**d). The expression should be a single formula."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_hitEosInPreamble_: bool
        d_3_hitEosInPreamble_ = False
        d_4_preambleLimit_: int
        if (maxSteps) > (50):
            d_4_preambleLimit_ = (maxSteps) - (50)
        elif True:
            d_4_preambleLimit_ = 0
        if not(insideConstrainedOut):
            with _dafny.label("0_0"):
                while (d_2_steps_) < (d_4_preambleLimit_):
                    with _dafny.c_label("0_0"):
                        d_5_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_5_next_ = out0_
                        d_2_steps_ = (d_2_steps_) + (1)
                        if (d_5_next_) == (eosToken):
                            d_3_hitEosInPreamble_ = True
                            raise _dafny.Break("0_0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_5_next_]))
                        pass
                pass
        if ((d_2_steps_) < (maxSteps)) and (not(insideConstrainedOut)):
            if ((len(generated)) > (0)) and (((generated)[(len(generated)) - (1)]) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))):
                d_6_og_: _dafny.Seq
                d_7_oi_: bool
                d_8_oc_: _dafny.Seq
                out1_: _dafny.Seq
                out2_: bool
                out3_: _dafny.Seq
                out1_, out2_, out3_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                d_6_og_ = out1_
                d_7_oi_ = out2_
                d_8_oc_ = out3_
                generated = d_6_og_
                insideConstrainedOut = d_7_oi_
                currentConstrainedOut = d_8_oc_
            elif True:
                d_9_og_: _dafny.Seq
                d_10_oi_: bool
                d_11_oc_: _dafny.Seq
                out4_: _dafny.Seq
                out5_: bool
                out6_: _dafny.Seq
                out4_, out5_, out6_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                d_9_og_ = out4_
                d_10_oi_ = out5_
                d_11_oc_ = out6_
                generated = d_9_og_
                insideConstrainedOut = d_10_oi_
                currentConstrainedOut = d_11_oc_
                d_2_steps_ = (d_2_steps_) + (1)
        d_12_minTokensToGenerate_: int
        d_12_minTokensToGenerate_ = 3
        d_13_preCloseSteps_: int
        d_13_preCloseSteps_ = 0
        while (((d_13_preCloseSteps_) < (d_12_minTokensToGenerate_)) and ((d_2_steps_) < (maxSteps))) and (insideConstrainedOut):
            d_14_constrainedPrompt_: _dafny.Seq
            d_14_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
            d_15_next_: _dafny.Seq
            out7_: _dafny.Seq
            out7_ = (d_0_helpers_).SafeTemperatureConstrainedStep(lm, parser, d_14_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('3e-1'), eosToken)
            d_15_next_ = out7_
            d_2_steps_ = (d_2_steps_) + (1)
            d_13_preCloseSteps_ = (d_13_preCloseSteps_) + (1)
            if (d_15_next_) == (eosToken):
                insideConstrainedOut = False
                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
            elif True:
                d_16_ag_: _dafny.Seq
                d_17_ai_: bool
                d_18_ac_: _dafny.Seq
                out8_: _dafny.Seq
                out9_: bool
                out10_: _dafny.Seq
                out8_, out9_, out10_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_15_next_)
                d_16_ag_ = out8_
                d_17_ai_ = out9_
                d_18_ac_ = out10_
                generated = d_16_ag_
                insideConstrainedOut = d_17_ai_
                currentConstrainedOut = d_18_ac_
        while (((d_2_steps_) + (1)) < (maxSteps)) and (insideConstrainedOut):
            d_19_cg_: _dafny.Seq
            d_20_ci_: bool
            d_21_cc_: _dafny.Seq
            d_22_closed_: bool
            out11_: _dafny.Seq
            out12_: bool
            out13_: _dafny.Seq
            out14_: bool
            out11_, out12_, out13_, out14_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
            d_19_cg_ = out11_
            d_20_ci_ = out12_
            d_21_cc_ = out13_
            d_22_closed_ = out14_
            d_2_steps_ = (d_2_steps_) + (1)
            if d_22_closed_:
                generated = d_19_cg_
                insideConstrainedOut = d_20_ci_
                currentConstrainedOut = d_21_cc_
            elif True:
                if (d_2_steps_) < (maxSteps):
                    d_23_constrainedPrompt_: _dafny.Seq
                    d_23_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                    d_24_next_: _dafny.Seq
                    out15_: _dafny.Seq
                    out15_ = (d_0_helpers_).SafeTemperatureConstrainedStep(lm, parser, d_23_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('3e-1'), eosToken)
                    d_24_next_ = out15_
                    d_2_steps_ = (d_2_steps_) + (1)
                    if (d_24_next_) == (eosToken):
                        insideConstrainedOut = False
                        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        d_25_ag_: _dafny.Seq
                        d_26_ai_: bool
                        d_27_ac_: _dafny.Seq
                        out16_: _dafny.Seq
                        out17_: bool
                        out18_: _dafny.Seq
                        out16_, out17_, out18_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_24_next_)
                        d_25_ag_ = out16_
                        d_26_ai_ = out17_
                        d_27_ac_ = out18_
                        generated = d_25_ag_
                        insideConstrainedOut = d_26_ai_
                        currentConstrainedOut = d_27_ac_
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_28_closeBudget_: int
            d_28_closeBudget_ = (maxSteps) - (d_2_steps_)
            d_29_cg_: _dafny.Seq
            d_30_ci_: bool
            d_31_cc_: _dafny.Seq
            out19_: _dafny.Seq
            out20_: bool
            out21_: _dafny.Seq
            out19_, out20_, out21_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_28_closeBudget_)
            d_29_cg_ = out19_
            d_30_ci_ = out20_
            d_31_cc_ = out21_
            generated = d_29_cg_
            insideConstrainedOut = d_30_ci_
            currentConstrainedOut = d_31_cc_
            d_2_steps_ = maxSteps
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

