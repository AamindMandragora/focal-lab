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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve this math word problem step by step using the variable names from the problem statement. After completing your reasoning, write: The answer is <<EXPR>> where EXPR is a complete arithmetic expression with at least two operands and one operator. Good examples: <<n * (mult + 1)>>, <<initial_amount - quantity * unit_price>>, <<count * (n1 + n2 + n3 + n4 + n5)>>. Bad example: <<n>> (a single variable with no operators is wrong). Always include the full computation formula inside << >>.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_hasCompletedSpan_: bool
        d_2_hasCompletedSpan_ = False
        d_3_freeLimit_: int
        d_3_freeLimit_ = _dafny.euclidian_division((maxSteps) * (3), 4)
        if ((d_3_freeLimit_) == (0)) and ((maxSteps) > (0)):
            d_3_freeLimit_ = 1
        d_4_closeReserve_: int
        d_4_closeReserve_ = 50
        if (d_4_closeReserve_) > (maxSteps):
            d_4_closeReserve_ = maxSteps
        d_5_phase2Limit_: int
        d_5_phase2Limit_ = maxSteps
        if (maxSteps) >= (d_4_closeReserve_):
            d_5_phase2Limit_ = (maxSteps) - (d_4_closeReserve_)
        if (d_5_phase2Limit_) < (d_3_freeLimit_):
            d_5_phase2Limit_ = d_3_freeLimit_
        if (d_5_phase2Limit_) > (maxSteps):
            d_5_phase2Limit_ = maxSteps
        d_6_minSpanTokens_: int
        d_6_minSpanTokens_ = 3
        d_7_opTokens_: _dafny.Seq
        d_7_opTokens_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "+")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "-")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "*")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "/")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "(")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ")"))])
        with _dafny.label("0"):
            while ((d_1_steps_) < (d_3_freeLimit_)) and (not(insideConstrainedOut)):
                with _dafny.c_label("0"):
                    d_8_next_: _dafny.Seq
                    out0_: _dafny.Seq
                    out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                    d_8_next_ = out0_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if (d_8_next_) == (eosToken):
                        raise _dafny.Break("0")
                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_8_next_]))
                    if VerifiedDecoderAgent.default__.RenderedEndsWith(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                        d_9_eg_: _dafny.Seq
                        d_10_ei_: bool
                        d_11_ec_: _dafny.Seq
                        out1_: _dafny.Seq
                        out2_: bool
                        out3_: _dafny.Seq
                        out1_, out2_, out3_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                        d_9_eg_ = out1_
                        d_10_ei_ = out2_
                        d_11_ec_ = out3_
                        generated = d_9_eg_
                        insideConstrainedOut = d_10_ei_
                        currentConstrainedOut = d_11_ec_
                    pass
            pass
        with _dafny.label("1"):
            while (insideConstrainedOut) and ((d_1_steps_) < (d_5_phase2Limit_)):
                with _dafny.c_label("1"):
                    if ((len(currentConstrainedOut)) >= (d_6_minSpanTokens_)) and ((parser).IsCompletePrefix(currentConstrainedOut)):
                        d_12_cg_: _dafny.Seq
                        d_13_ci_: bool
                        d_14_cc_: _dafny.Seq
                        out4_: _dafny.Seq
                        out5_: bool
                        out6_: _dafny.Seq
                        out4_, out5_, out6_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_12_cg_ = out4_
                        d_13_ci_ = out5_
                        d_14_cc_ = out6_
                        d_1_steps_ = (d_1_steps_) + (1)
                        generated = d_12_cg_
                        insideConstrainedOut = d_13_ci_
                        currentConstrainedOut = d_14_cc_
                        d_2_hasCompletedSpan_ = True
                    elif True:
                        d_15_cp_: _dafny.Seq
                        d_15_cp_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_16_next_: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                        if (len(currentConstrainedOut)) < (d_6_minSpanTokens_):
                            out7_: _dafny.Seq
                            out7_ = (d_0_helpers_).SafeBoostedConstrainedStep(lm, parser, d_15_cp_, currentConstrainedOut, d_7_opTokens_, _dafny.BigRational('4e0'), eosToken)
                            d_16_next_ = out7_
                        elif True:
                            out8_: _dafny.Seq
                            out8_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_15_cp_, currentConstrainedOut, eosToken)
                            d_16_next_ = out8_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_16_next_) == (eosToken):
                            raise _dafny.Break("1")
                        d_17_ag_: _dafny.Seq
                        d_18_ai_: bool
                        d_19_ac_: _dafny.Seq
                        out9_: _dafny.Seq
                        out10_: bool
                        out11_: _dafny.Seq
                        out9_, out10_, out11_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_16_next_)
                        d_17_ag_ = out9_
                        d_18_ai_ = out10_
                        d_19_ac_ = out11_
                        generated = d_17_ag_
                        insideConstrainedOut = d_18_ai_
                        currentConstrainedOut = d_19_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_20_rem3_: int
            d_20_rem3_ = (maxSteps) - (d_1_steps_)
            d_21_wg_: _dafny.Seq
            d_22_wi_: bool
            d_23_wc_: _dafny.Seq
            out12_: _dafny.Seq
            out13_: bool
            out14_: _dafny.Seq
            out12_, out13_, out14_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_20_rem3_)
            d_21_wg_ = out12_
            d_22_wi_ = out13_
            d_23_wc_ = out14_
            generated = d_21_wg_
            insideConstrainedOut = d_22_wi_
            currentConstrainedOut = d_23_wc_
            d_1_steps_ = (d_1_steps_) + (d_20_rem3_)
            if not(insideConstrainedOut):
                d_2_hasCompletedSpan_ = True
        if ((not(insideConstrainedOut)) and (not(d_2_hasCompletedSpan_))) and ((d_1_steps_) < (maxSteps)):
            if ((d_1_steps_) + (2)) <= (maxSteps):
                d_24_fg_: _dafny.Seq
                d_25_fi_: bool
                d_26_fc_: _dafny.Seq
                out15_: _dafny.Seq
                out16_: bool
                out17_: _dafny.Seq
                out15_, out16_, out17_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                d_24_fg_ = out15_
                d_25_fi_ = out16_
                d_26_fc_ = out17_
                generated = d_24_fg_
                insideConstrainedOut = d_25_fi_
                currentConstrainedOut = d_26_fc_
                d_1_steps_ = (d_1_steps_) + (1)
                d_27_phase4Limit_: int
                d_27_phase4Limit_ = maxSteps
                if ((maxSteps) - (d_1_steps_)) >= (d_4_closeReserve_):
                    d_27_phase4Limit_ = (maxSteps) - (d_4_closeReserve_)
                with _dafny.label("8_0_0"):
                    while (insideConstrainedOut) and ((d_1_steps_) < (d_27_phase4Limit_)):
                        with _dafny.c_label("8_0_0"):
                            if ((len(currentConstrainedOut)) >= (d_6_minSpanTokens_)) and ((parser).IsCompletePrefix(currentConstrainedOut)):
                                d_28_cg4_: _dafny.Seq
                                d_29_ci4_: bool
                                d_30_cc4_: _dafny.Seq
                                out18_: _dafny.Seq
                                out19_: bool
                                out20_: _dafny.Seq
                                out18_, out19_, out20_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_28_cg4_ = out18_
                                d_29_ci4_ = out19_
                                d_30_cc4_ = out20_
                                d_1_steps_ = (d_1_steps_) + (1)
                                generated = d_28_cg4_
                                insideConstrainedOut = d_29_ci4_
                                currentConstrainedOut = d_30_cc4_
                                d_2_hasCompletedSpan_ = True
                            elif True:
                                d_31_cp4_: _dafny.Seq
                                d_31_cp4_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_32_next4_: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                                if (len(currentConstrainedOut)) < (d_6_minSpanTokens_):
                                    out21_: _dafny.Seq
                                    out21_ = (d_0_helpers_).SafeBoostedConstrainedStep(lm, parser, d_31_cp4_, currentConstrainedOut, d_7_opTokens_, _dafny.BigRational('4e0'), eosToken)
                                    d_32_next4_ = out21_
                                elif True:
                                    out22_: _dafny.Seq
                                    out22_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_31_cp4_, currentConstrainedOut, eosToken)
                                    d_32_next4_ = out22_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_32_next4_) == (eosToken):
                                    raise _dafny.Break("8_0_0")
                                d_33_ag4_: _dafny.Seq
                                d_34_ai4_: bool
                                d_35_ac4_: _dafny.Seq
                                out23_: _dafny.Seq
                                out24_: bool
                                out25_: _dafny.Seq
                                out23_, out24_, out25_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_32_next4_)
                                d_33_ag4_ = out23_
                                d_34_ai4_ = out24_
                                d_35_ac4_ = out25_
                                generated = d_33_ag4_
                                insideConstrainedOut = d_34_ai4_
                                currentConstrainedOut = d_35_ac4_
                            pass
                    pass
                if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
                    d_36_rem4_: int
                    d_36_rem4_ = (maxSteps) - (d_1_steps_)
                    d_37_wg4_: _dafny.Seq
                    d_38_wi4_: bool
                    d_39_wc4_: _dafny.Seq
                    out26_: _dafny.Seq
                    out27_: bool
                    out28_: _dafny.Seq
                    out26_, out27_, out28_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_36_rem4_)
                    d_37_wg4_ = out26_
                    d_38_wi4_ = out27_
                    d_39_wc4_ = out28_
                    generated = d_37_wg4_
                    insideConstrainedOut = d_38_wi4_
                    currentConstrainedOut = d_39_wc4_
                    d_1_steps_ = (d_1_steps_) + (d_36_rem4_)
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_40_rem5_: int
            d_40_rem5_ = (maxSteps) - (d_1_steps_)
            if (d_40_rem5_) > (0):
                d_41_wg5_: _dafny.Seq
                d_42_wi5_: bool
                d_43_wc5_: _dafny.Seq
                out29_: _dafny.Seq
                out30_: bool
                out31_: _dafny.Seq
                out29_, out30_, out31_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_40_rem5_)
                d_41_wg5_ = out29_
                d_42_wi5_ = out30_
                d_43_wc5_ = out31_
                generated = d_41_wg5_
                insideConstrainedOut = d_42_wi5_
                currentConstrainedOut = d_43_wc5_
                d_1_steps_ = (d_1_steps_) + (d_40_rem5_)
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

