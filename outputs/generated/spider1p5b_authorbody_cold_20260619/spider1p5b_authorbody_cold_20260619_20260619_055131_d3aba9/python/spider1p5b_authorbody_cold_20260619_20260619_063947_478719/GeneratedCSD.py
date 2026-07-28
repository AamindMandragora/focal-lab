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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Output format is exactly: SQL: <<YOUR SQL QUERY HERE>> with no semicolons, no extra text, no markdown. The SQL goes between << and >>. Example: SQL: <<SELECT col FROM table>>"))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_maxPreambleSteps_: int
        d_3_maxPreambleSteps_ = 15
        d_4_closeBudgetReserve_: int
        d_4_closeBudgetReserve_ = 200
        with _dafny.label("0"):
            while (((d_2_steps_) < (maxSteps)) and (not(insideConstrainedOut))) and ((d_2_steps_) < (d_3_maxPreambleSteps_)):
                with _dafny.c_label("0"):
                    d_5_next_: _dafny.Seq
                    out0_: _dafny.Seq
                    out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                    d_5_next_ = out0_
                    d_2_steps_ = (d_2_steps_) + (1)
                    if (d_5_next_) == (eosToken):
                        raise _dafny.Break("0")
                    elif True:
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_5_next_]))
                        if (d_5_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                            d_6_eg_: _dafny.Seq
                            d_7_ei_: bool
                            d_8_ec_: _dafny.Seq
                            out1_: _dafny.Seq
                            out2_: bool
                            out3_: _dafny.Seq
                            out1_, out2_, out3_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                            d_6_eg_ = out1_
                            d_7_ei_ = out2_
                            d_8_ec_ = out3_
                            generated = d_6_eg_
                            insideConstrainedOut = d_7_ei_
                            currentConstrainedOut = d_8_ec_
                    pass
            pass
        if (not(insideConstrainedOut)) and ((d_2_steps_) < (maxSteps)):
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
        with _dafny.label("1"):
            while ((d_2_steps_) < (maxSteps)) and (insideConstrainedOut):
                with _dafny.c_label("1"):
                    d_12_remainingBudget_: int
                    d_12_remainingBudget_ = (maxSteps) - (d_2_steps_)
                    if (d_12_remainingBudget_) <= (d_4_closeBudgetReserve_):
                        d_13_closeBudget_: int
                        d_13_closeBudget_ = d_12_remainingBudget_
                        d_14_cg_: _dafny.Seq
                        d_15_ci_: bool
                        d_16_cc_: _dafny.Seq
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: _dafny.Seq
                        out7_, out8_, out9_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_13_closeBudget_)
                        d_14_cg_ = out7_
                        d_15_ci_ = out8_
                        d_16_cc_ = out9_
                        generated = d_14_cg_
                        insideConstrainedOut = d_15_ci_
                        currentConstrainedOut = d_16_cc_
                        d_2_steps_ = (d_2_steps_) + (d_13_closeBudget_)
                        raise _dafny.Break("1")
                    d_17_cg_: _dafny.Seq
                    d_18_ci_: bool
                    d_19_cc_: _dafny.Seq
                    d_20_closed_: bool
                    out10_: _dafny.Seq
                    out11_: bool
                    out12_: _dafny.Seq
                    out13_: bool
                    out10_, out11_, out12_, out13_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                    d_17_cg_ = out10_
                    d_18_ci_ = out11_
                    d_19_cc_ = out12_
                    d_20_closed_ = out13_
                    d_2_steps_ = (d_2_steps_) + (1)
                    if d_20_closed_:
                        generated = d_17_cg_
                        insideConstrainedOut = d_18_ci_
                        currentConstrainedOut = d_19_cc_
                        raise _dafny.Break("1")
                    d_21_constrainedPrompt_: _dafny.Seq
                    d_21_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                    d_22_next_: _dafny.Seq
                    out14_: _dafny.Seq
                    out14_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_21_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                    d_22_next_ = out14_
                    if (d_22_next_) == (eosToken):
                        d_23_remaining_: int
                        d_23_remaining_ = (maxSteps) - (d_2_steps_)
                        if (d_23_remaining_) > (0):
                            d_24_cg2_: _dafny.Seq
                            d_25_ci2_: bool
                            d_26_cc2_: _dafny.Seq
                            out15_: _dafny.Seq
                            out16_: bool
                            out17_: _dafny.Seq
                            out15_, out16_, out17_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_23_remaining_)
                            d_24_cg2_ = out15_
                            d_25_ci2_ = out16_
                            d_26_cc2_ = out17_
                            generated = d_24_cg2_
                            insideConstrainedOut = d_25_ci2_
                            currentConstrainedOut = d_26_cc2_
                            d_2_steps_ = (d_2_steps_) + (d_23_remaining_)
                        raise _dafny.Break("1")
                    elif True:
                        d_27_ag_: _dafny.Seq
                        d_28_ai_: bool
                        d_29_ac_: _dafny.Seq
                        out18_: _dafny.Seq
                        out19_: bool
                        out20_: _dafny.Seq
                        out18_, out19_, out20_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_22_next_)
                        d_27_ag_ = out18_
                        d_28_ai_ = out19_
                        d_29_ac_ = out20_
                        generated = d_27_ag_
                        insideConstrainedOut = d_28_ai_
                        currentConstrainedOut = d_29_ac_
                    pass
            pass
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

