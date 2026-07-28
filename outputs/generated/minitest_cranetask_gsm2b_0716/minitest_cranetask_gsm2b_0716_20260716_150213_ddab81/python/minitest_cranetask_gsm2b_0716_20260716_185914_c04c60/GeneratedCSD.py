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
        (d_0_helpers_).AppendTaskGuidance(lm, ((((_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step using ALL symbolic variable names from the problem. "))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Write intermediate expressions in << >>. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "End with: The final answer is <<expr>> ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "where expr combines ALL relevant variables with +, -, *, /, //, %, int(). ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Do NOT answer with just one variable name or a single number."))))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_freeBudget_: int
        if (maxSteps) > (300):
            d_2_freeBudget_ = (maxSteps) - (300)
        elif True:
            d_2_freeBudget_ = _dafny.euclidian_division((maxSteps) * (2), 3)
        with _dafny.label("0"):
            while ((d_1_steps_) < (d_2_freeBudget_)) and (not(insideConstrainedOut)):
                with _dafny.c_label("0"):
                    d_3_next_: _dafny.Seq
                    out0_: _dafny.Seq
                    out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                    d_3_next_ = out0_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if (d_3_next_) == (eosToken):
                        raise _dafny.Break("0")
                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_3_next_]))
                    if VerifiedDecoderAgent.default__.RenderedEndsWith(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                        insideConstrainedOut = True
                        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    pass
            pass
        if (not(insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
            d_4_og_: _dafny.Seq
            d_5_oi_: bool
            d_6_oc_: _dafny.Seq
            out1_: _dafny.Seq
            out2_: bool
            out3_: _dafny.Seq
            out1_, out2_, out3_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_4_og_ = out1_
            d_5_oi_ = out2_
            d_6_oc_ = out3_
            generated = d_4_og_
            insideConstrainedOut = d_5_oi_
            currentConstrainedOut = d_6_oc_
            d_1_steps_ = (d_1_steps_) + (1)
        d_7_minExprTokens_: int
        d_7_minExprTokens_ = 25
        d_8_exprSteps_: int
        d_8_exprSteps_ = 0
        with _dafny.label("1"):
            while (((d_1_steps_) < (maxSteps)) and (insideConstrainedOut)) and ((d_8_exprSteps_) < (d_7_minExprTokens_)):
                with _dafny.c_label("1"):
                    d_9_constrainedPrompt_: _dafny.Seq
                    d_9_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                    d_10_next_: _dafny.Seq
                    out4_: _dafny.Seq
                    out4_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_9_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('6e0'), _dafny.SeqWithoutIsStrInference([eosToken]), _dafny.BigRational('8e0'), 12, eosToken)
                    d_10_next_ = out4_
                    d_1_steps_ = (d_1_steps_) + (1)
                    d_8_exprSteps_ = (d_8_exprSteps_) + (1)
                    if (d_10_next_) == (eosToken):
                        d_11_rg_: _dafny.Seq
                        d_12_rc_: _dafny.Seq
                        out5_: _dafny.Seq
                        out6_: _dafny.Seq
                        out5_, out6_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                        d_11_rg_ = out5_
                        d_12_rc_ = out6_
                        generated = d_11_rg_
                        currentConstrainedOut = d_12_rc_
                        if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                            d_13_cg_: _dafny.Seq
                            d_14_ci_: bool
                            d_15_cc_: _dafny.Seq
                            out7_: _dafny.Seq
                            out8_: bool
                            out9_: _dafny.Seq
                            out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_13_cg_ = out7_
                            d_14_ci_ = out8_
                            d_15_cc_ = out9_
                            generated = d_13_cg_
                            insideConstrainedOut = d_14_ci_
                            currentConstrainedOut = d_15_cc_
                            d_1_steps_ = (d_1_steps_) + (1)
                        raise _dafny.Break("1")
                    d_16_ag_: _dafny.Seq
                    d_17_ai_: bool
                    d_18_ac_: _dafny.Seq
                    out10_: _dafny.Seq
                    out11_: bool
                    out12_: _dafny.Seq
                    out10_, out11_, out12_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_10_next_)
                    d_16_ag_ = out10_
                    d_17_ai_ = out11_
                    d_18_ac_ = out12_
                    generated = d_16_ag_
                    insideConstrainedOut = d_17_ai_
                    currentConstrainedOut = d_18_ac_
                    pass
            pass
        with _dafny.label("2"):
            while ((d_1_steps_) < (maxSteps)) and (insideConstrainedOut):
                with _dafny.c_label("2"):
                    d_19_constrainedPrompt_: _dafny.Seq
                    d_19_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                    d_20_next_: _dafny.Seq
                    out13_: _dafny.Seq
                    out13_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_19_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                    d_20_next_ = out13_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if (d_20_next_) == (eosToken):
                        d_21_rg_: _dafny.Seq
                        d_22_rc_: _dafny.Seq
                        out14_: _dafny.Seq
                        out15_: _dafny.Seq
                        out14_, out15_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                        d_21_rg_ = out14_
                        d_22_rc_ = out15_
                        generated = d_21_rg_
                        currentConstrainedOut = d_22_rc_
                        if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                            d_23_cg_: _dafny.Seq
                            d_24_ci_: bool
                            d_25_cc_: _dafny.Seq
                            out16_: _dafny.Seq
                            out17_: bool
                            out18_: _dafny.Seq
                            out16_, out17_, out18_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_23_cg_ = out16_
                            d_24_ci_ = out17_
                            d_25_cc_ = out18_
                            generated = d_23_cg_
                            insideConstrainedOut = d_24_ci_
                            currentConstrainedOut = d_25_cc_
                            d_1_steps_ = (d_1_steps_) + (1)
                        raise _dafny.Break("2")
                    d_26_ag_: _dafny.Seq
                    d_27_ai_: bool
                    d_28_ac_: _dafny.Seq
                    out19_: _dafny.Seq
                    out20_: bool
                    out21_: _dafny.Seq
                    out19_, out20_, out21_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_20_next_)
                    d_26_ag_ = out19_
                    d_27_ai_ = out20_
                    d_28_ac_ = out21_
                    generated = d_26_ag_
                    insideConstrainedOut = d_27_ai_
                    currentConstrainedOut = d_28_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_29_rg_: _dafny.Seq
            d_30_rc_: _dafny.Seq
            out22_: _dafny.Seq
            out23_: _dafny.Seq
            out22_, out23_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
            d_29_rg_ = out22_
            d_30_rc_ = out23_
            generated = d_29_rg_
            currentConstrainedOut = d_30_rc_
            if (parser).IsCompletePrefix(currentConstrainedOut):
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
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

