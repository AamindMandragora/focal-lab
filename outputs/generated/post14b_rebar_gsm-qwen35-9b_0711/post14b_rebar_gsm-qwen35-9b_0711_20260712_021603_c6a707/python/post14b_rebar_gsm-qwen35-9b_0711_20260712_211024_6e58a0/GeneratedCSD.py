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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the problem step by step. At the very end, write your final symbolic answer inside << >>. Use the exact variable names from the problem (text inside {} without the braces). Use * for multiplication, // for integer division of discrete items. Do not use ** or {} in the expression."))
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
        d_5_openCount_: int
        out1_: int
        out1_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))
        d_5_openCount_ = out1_
        d_6_closeCount_: int
        out2_: int
        out2_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")))
        d_6_closeCount_ = out2_
        if ((d_5_openCount_) >= (1)) and ((d_6_closeCount_) >= (1)):
            with _dafny.label("1_0"):
                while (d_2_steps_) < (maxSteps):
                    with _dafny.c_label("1_0"):
                        d_7_next_: _dafny.Seq
                        out3_: _dafny.Seq
                        out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_7_next_ = out3_
                        d_2_steps_ = (d_2_steps_) + (1)
                        if (d_7_next_) == (eosToken):
                            raise _dafny.Break("1_0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_7_next_]))
                        pass
                pass
            cost = d_2_steps_
            return generated, insideConstrainedOut, currentConstrainedOut, cost
        if ((d_2_steps_) < (maxSteps)) and (not(insideConstrainedOut)):
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
            d_2_steps_ = (d_2_steps_) + (1)
        d_11_minTokensToGenerate_: int
        d_11_minTokensToGenerate_ = 3
        d_12_preCloseSteps_: int
        d_12_preCloseSteps_ = 0
        while (((d_12_preCloseSteps_) < (d_11_minTokensToGenerate_)) and ((d_2_steps_) < (maxSteps))) and (insideConstrainedOut):
            d_13_constrainedPrompt_: _dafny.Seq
            d_13_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
            d_14_next_: _dafny.Seq
            out7_: _dafny.Seq
            out7_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_13_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('15e-1'), eosToken)
            d_14_next_ = out7_
            d_2_steps_ = (d_2_steps_) + (1)
            d_12_preCloseSteps_ = (d_12_preCloseSteps_) + (1)
            if (d_14_next_) == (eosToken):
                insideConstrainedOut = False
                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
            elif True:
                d_15_ag_: _dafny.Seq
                d_16_ai_: bool
                d_17_ac_: _dafny.Seq
                out8_: _dafny.Seq
                out9_: bool
                out10_: _dafny.Seq
                out8_, out9_, out10_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_14_next_)
                d_15_ag_ = out8_
                d_16_ai_ = out9_
                d_17_ac_ = out10_
                generated = d_15_ag_
                insideConstrainedOut = d_16_ai_
                currentConstrainedOut = d_17_ac_
        while (((d_2_steps_) + (1)) < (maxSteps)) and (insideConstrainedOut):
            d_18_cg_: _dafny.Seq
            d_19_ci_: bool
            d_20_cc_: _dafny.Seq
            d_21_closed_: bool
            out11_: _dafny.Seq
            out12_: bool
            out13_: _dafny.Seq
            out14_: bool
            out11_, out12_, out13_, out14_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
            d_18_cg_ = out11_
            d_19_ci_ = out12_
            d_20_cc_ = out13_
            d_21_closed_ = out14_
            d_2_steps_ = (d_2_steps_) + (1)
            if d_21_closed_:
                generated = d_18_cg_
                insideConstrainedOut = d_19_ci_
                currentConstrainedOut = d_20_cc_
            elif True:
                if (d_2_steps_) < (maxSteps):
                    d_22_constrainedPrompt_: _dafny.Seq
                    d_22_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                    d_23_next_: _dafny.Seq
                    out15_: _dafny.Seq
                    out15_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_22_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('15e-1'), eosToken)
                    d_23_next_ = out15_
                    d_2_steps_ = (d_2_steps_) + (1)
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
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_27_closeBudget_: int
            d_27_closeBudget_ = (maxSteps) - (d_2_steps_)
            d_28_cg_: _dafny.Seq
            d_29_ci_: bool
            d_30_cc_: _dafny.Seq
            out19_: _dafny.Seq
            out20_: bool
            out21_: _dafny.Seq
            out19_, out20_, out21_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_27_closeBudget_)
            d_28_cg_ = out19_
            d_29_ci_ = out20_
            d_30_cc_ = out21_
            generated = d_28_cg_
            insideConstrainedOut = d_29_ci_
            currentConstrainedOut = d_30_cc_
            d_2_steps_ = maxSteps
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

