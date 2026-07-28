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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "You must output exactly: SQL: <<YOUR SQL QUERY HERE>> with no other text. The SQL query goes inside << and >>. Example: SQL: <<SELECT * FROM table>>. Now generate the SQL query inside << >> delimiters."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_narrowThreshold_: int
        d_3_narrowThreshold_ = 12
        d_4_closeBudgetReserve_: int
        d_4_closeBudgetReserve_ = 200
        d_5_maxPreambleTokens_: int
        d_5_maxPreambleTokens_ = 5
        d_6_preambleCount_: int
        d_6_preambleCount_ = 0
        with _dafny.label("0"):
            while (((d_2_steps_) < (maxSteps)) and (not(insideConstrainedOut))) and ((d_6_preambleCount_) < (d_5_maxPreambleTokens_)):
                with _dafny.c_label("0"):
                    d_7_next_: _dafny.Seq
                    out0_: _dafny.Seq
                    out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                    d_7_next_ = out0_
                    d_2_steps_ = (d_2_steps_) + (1)
                    d_6_preambleCount_ = (d_6_preambleCount_) + (1)
                    if (d_7_next_) == (eosToken):
                        raise _dafny.Break("0")
                    elif True:
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_7_next_]))
                        if (d_7_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                            d_8_eg_: _dafny.Seq
                            d_9_ei_: bool
                            d_10_ec_: _dafny.Seq
                            out1_: _dafny.Seq
                            out2_: bool
                            out3_: _dafny.Seq
                            out1_, out2_, out3_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                            d_8_eg_ = out1_
                            d_9_ei_ = out2_
                            d_10_ec_ = out3_
                            generated = d_8_eg_
                            insideConstrainedOut = d_9_ei_
                            currentConstrainedOut = d_10_ec_
                    pass
            pass
        if (not(insideConstrainedOut)) and ((d_2_steps_) < (maxSteps)):
            d_11_og_: _dafny.Seq
            d_12_oi_: bool
            d_13_oc_: _dafny.Seq
            out4_: _dafny.Seq
            out5_: bool
            out6_: _dafny.Seq
            out4_, out5_, out6_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_11_og_ = out4_
            d_12_oi_ = out5_
            d_13_oc_ = out6_
            d_2_steps_ = (d_2_steps_) + (1)
            generated = d_11_og_
            insideConstrainedOut = d_12_oi_
            currentConstrainedOut = d_13_oc_
        with _dafny.label("1"):
            while ((d_2_steps_) < (maxSteps)) and (insideConstrainedOut):
                with _dafny.c_label("1"):
                    d_14_remainingBudget_: int
                    d_14_remainingBudget_ = (maxSteps) - (d_2_steps_)
                    if (d_14_remainingBudget_) <= (d_4_closeBudgetReserve_):
                        if (d_14_remainingBudget_) > (0):
                            d_15_cg_: _dafny.Seq
                            d_16_ci_: bool
                            d_17_cc_: _dafny.Seq
                            out7_: _dafny.Seq
                            out8_: bool
                            out9_: _dafny.Seq
                            out7_, out8_, out9_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_14_remainingBudget_)
                            d_15_cg_ = out7_
                            d_16_ci_ = out8_
                            d_17_cc_ = out9_
                            generated = d_15_cg_
                            insideConstrainedOut = d_16_ci_
                            currentConstrainedOut = d_17_cc_
                            d_2_steps_ = (d_2_steps_) + (d_14_remainingBudget_)
                        elif True:
                            d_2_steps_ = maxSteps
                        raise _dafny.Break("1")
                    elif True:
                        d_18_cg_: _dafny.Seq
                        d_19_ci_: bool
                        d_20_cc_: _dafny.Seq
                        d_21_closed_: bool
                        out10_: _dafny.Seq
                        out11_: bool
                        out12_: _dafny.Seq
                        out13_: bool
                        out10_, out11_, out12_, out13_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                        d_18_cg_ = out10_
                        d_19_ci_ = out11_
                        d_20_cc_ = out12_
                        d_21_closed_ = out13_
                        d_2_steps_ = (d_2_steps_) + (1)
                        if d_21_closed_:
                            generated = d_18_cg_
                            insideConstrainedOut = d_19_ci_
                            currentConstrainedOut = d_20_cc_
                            raise _dafny.Break("1")
                        elif True:
                            d_22_constrainedPrompt_: _dafny.Seq
                            d_22_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_23_next_: _dafny.Seq
                            out14_: _dafny.Seq
                            out14_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_22_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_3_narrowThreshold_, eosToken)
                            d_23_next_ = out14_
                            if (d_23_next_) == (eosToken):
                                d_24_remaining_: int
                                d_24_remaining_ = (maxSteps) - (d_2_steps_)
                                if (d_24_remaining_) > (0):
                                    d_25_cg2_: _dafny.Seq
                                    d_26_ci2_: bool
                                    d_27_cc2_: _dafny.Seq
                                    out15_: _dafny.Seq
                                    out16_: bool
                                    out17_: _dafny.Seq
                                    out15_, out16_, out17_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_24_remaining_)
                                    d_25_cg2_ = out15_
                                    d_26_ci2_ = out16_
                                    d_27_cc2_ = out17_
                                    generated = d_25_cg2_
                                    insideConstrainedOut = d_26_ci2_
                                    currentConstrainedOut = d_27_cc2_
                                    d_2_steps_ = (d_2_steps_) + (d_24_remaining_)
                                raise _dafny.Break("1")
                            elif True:
                                d_28_ag_: _dafny.Seq
                                d_29_ai_: bool
                                d_30_ac_: _dafny.Seq
                                out18_: _dafny.Seq
                                out19_: bool
                                out20_: _dafny.Seq
                                out18_, out19_, out20_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_23_next_)
                                d_28_ag_ = out18_
                                d_29_ai_ = out19_
                                d_30_ac_ = out20_
                                generated = d_28_ag_
                                insideConstrainedOut = d_29_ai_
                                currentConstrainedOut = d_30_ac_
                    pass
            pass
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

