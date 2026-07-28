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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. For each calculation, write the symbolic expression inside << >> delimiters using the exact variable names from the problem (no curly braces). Use only +, -, *, /, //, %, int(). Do NOT use ** or ^. Example: <<n1 + n2 * price>>, <<int(total // n)>>, <<(a - b) * c>>. Put the final numerical expression inside the LAST << >> block.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_forceOpenThreshold_: int
        d_2_forceOpenThreshold_ = 100
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_3_remainingBudget_: int
                        d_3_remainingBudget_ = (maxSteps) - (d_1_steps_)
                        if (d_3_remainingBudget_) <= (3):
                            raise _dafny.Break("0")
                        elif (d_3_remainingBudget_) <= (d_2_forceOpenThreshold_):
                            d_4_og_: _dafny.Seq
                            d_5_oi_: bool
                            d_6_oc_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_4_og_ = out0_
                            d_5_oi_ = out1_
                            d_6_oc_ = out2_
                            generated = d_4_og_
                            insideConstrainedOut = d_5_oi_
                            currentConstrainedOut = d_6_oc_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_1_steps_) < (maxSteps):
                                d_7_rem2_: int
                                d_7_rem2_ = (maxSteps) - (d_1_steps_)
                                d_8_cb2_: int
                                if (d_7_rem2_) < (80):
                                    d_8_cb2_ = d_7_rem2_
                                elif True:
                                    d_8_cb2_ = 80
                                if (d_8_cb2_) > (0):
                                    d_9_cg2_: _dafny.Seq
                                    d_10_ci2_: bool
                                    d_11_cc2_: _dafny.Seq
                                    out3_: _dafny.Seq
                                    out4_: bool
                                    out5_: _dafny.Seq
                                    out3_, out4_, out5_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_8_cb2_)
                                    d_9_cg2_ = out3_
                                    d_10_ci2_ = out4_
                                    d_11_cc2_ = out5_
                                    generated = d_9_cg2_
                                    insideConstrainedOut = d_10_ci2_
                                    currentConstrainedOut = d_11_cc2_
                                    d_1_steps_ = (d_1_steps_) + (d_8_cb2_)
                        elif True:
                            d_12_next_: _dafny.Seq
                            out6_: _dafny.Seq
                            out6_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_12_next_ = out6_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_12_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_12_next_]))
                                if (d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    d_13_og2_: _dafny.Seq
                                    d_14_oi2_: bool
                                    d_15_oc2_: _dafny.Seq
                                    out7_: _dafny.Seq
                                    out8_: bool
                                    out9_: _dafny.Seq
                                    out7_, out8_, out9_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                    d_13_og2_ = out7_
                                    d_14_oi2_ = out8_
                                    d_15_oc2_ = out9_
                                    generated = d_13_og2_
                                    insideConstrainedOut = d_14_oi2_
                                    currentConstrainedOut = d_15_oc2_
                    elif True:
                        d_16_cg_: _dafny.Seq
                        d_17_ci_: bool
                        d_18_cc_: _dafny.Seq
                        d_19_closed_: bool
                        out10_: _dafny.Seq
                        out11_: bool
                        out12_: _dafny.Seq
                        out13_: bool
                        out10_, out11_, out12_, out13_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                        d_16_cg_ = out10_
                        d_17_ci_ = out11_
                        d_18_cc_ = out12_
                        d_19_closed_ = out13_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if d_19_closed_:
                            generated = d_16_cg_
                            insideConstrainedOut = d_17_ci_
                            currentConstrainedOut = d_18_cc_
                        elif True:
                            d_20_constrainedPrompt_: _dafny.Seq
                            d_20_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_21_next_: _dafny.Seq
                            out14_: _dafny.Seq
                            out14_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_20_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_21_next_ = out14_
                            if (d_21_next_) == (eosToken):
                                if (d_1_steps_) < (maxSteps):
                                    d_22_remEos_: int
                                    d_22_remEos_ = (maxSteps) - (d_1_steps_)
                                    d_23_cbEos_: int
                                    if (d_22_remEos_) < (30):
                                        d_23_cbEos_ = d_22_remEos_
                                    elif True:
                                        d_23_cbEos_ = 30
                                    if (d_23_cbEos_) > (0):
                                        d_24_cgE_: _dafny.Seq
                                        d_25_ciE_: bool
                                        d_26_ccE_: _dafny.Seq
                                        out15_: _dafny.Seq
                                        out16_: bool
                                        out17_: _dafny.Seq
                                        out15_, out16_, out17_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_23_cbEos_)
                                        d_24_cgE_ = out15_
                                        d_25_ciE_ = out16_
                                        d_26_ccE_ = out17_
                                        generated = d_24_cgE_
                                        insideConstrainedOut = d_25_ciE_
                                        currentConstrainedOut = d_26_ccE_
                                        d_1_steps_ = (d_1_steps_) + (d_23_cbEos_)
                                raise _dafny.Break("0")
                            elif True:
                                d_27_valid_: bool
                                out18_: bool
                                out18_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_21_next_)
                                d_27_valid_ = out18_
                                if d_27_valid_:
                                    d_28_ag_: _dafny.Seq
                                    d_29_ai_: bool
                                    d_30_ac_: _dafny.Seq
                                    out19_: _dafny.Seq
                                    out20_: bool
                                    out21_: _dafny.Seq
                                    out19_, out20_, out21_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_21_next_)
                                    d_28_ag_ = out19_
                                    d_29_ai_ = out20_
                                    d_30_ac_ = out21_
                                    generated = d_28_ag_
                                    insideConstrainedOut = d_29_ai_
                                    currentConstrainedOut = d_30_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_31_remainingA_: int
            d_31_remainingA_ = (maxSteps) - (d_1_steps_)
            d_32_closeBudgetA_: int
            if (d_31_remainingA_) < (60):
                d_32_closeBudgetA_ = d_31_remainingA_
            elif True:
                d_32_closeBudgetA_ = 60
            if (d_32_closeBudgetA_) > (0):
                d_33_cgA_: _dafny.Seq
                d_34_ciA_: bool
                d_35_ccA_: _dafny.Seq
                out22_: _dafny.Seq
                out23_: bool
                out24_: _dafny.Seq
                out22_, out23_, out24_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_32_closeBudgetA_)
                d_33_cgA_ = out22_
                d_34_ciA_ = out23_
                d_35_ccA_ = out24_
                generated = d_33_cgA_
                insideConstrainedOut = d_34_ciA_
                currentConstrainedOut = d_35_ccA_
                d_1_steps_ = (d_1_steps_) + (d_32_closeBudgetA_)
        if (not(insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
            d_36_genStr_: _dafny.Seq
            d_36_genStr_ = VerifiedDecoderAgent.CSDHelpers.PrefixToString(generated)
            d_37_openCount_: int
            d_37_openCount_ = VerifiedDecoderAgent.CSDHelpers.CountSubstring(d_36_genStr_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))
            if (d_37_openCount_) == (0):
                d_38_remainingB_: int
                d_38_remainingB_ = (maxSteps) - (d_1_steps_)
                if (d_38_remainingB_) >= (5):
                    d_39_ogB_: _dafny.Seq
                    d_40_oiB_: bool
                    d_41_ocB_: _dafny.Seq
                    out25_: _dafny.Seq
                    out26_: bool
                    out27_: _dafny.Seq
                    out25_, out26_, out27_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                    d_39_ogB_ = out25_
                    d_40_oiB_ = out26_
                    d_41_ocB_ = out27_
                    generated = d_39_ogB_
                    insideConstrainedOut = d_40_oiB_
                    currentConstrainedOut = d_41_ocB_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if (d_1_steps_) < (maxSteps):
                        d_42_remainingB2_: int
                        d_42_remainingB2_ = (maxSteps) - (d_1_steps_)
                        d_43_closeBudgetB_: int
                        if (d_42_remainingB2_) < (80):
                            d_43_closeBudgetB_ = d_42_remainingB2_
                        elif True:
                            d_43_closeBudgetB_ = 80
                        if (d_43_closeBudgetB_) > (0):
                            d_44_cgB_: _dafny.Seq
                            d_45_ciB_: bool
                            d_46_ccB_: _dafny.Seq
                            out28_: _dafny.Seq
                            out29_: bool
                            out30_: _dafny.Seq
                            out28_, out29_, out30_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_43_closeBudgetB_)
                            d_44_cgB_ = out28_
                            d_45_ciB_ = out29_
                            d_46_ccB_ = out30_
                            generated = d_44_cgB_
                            insideConstrainedOut = d_45_ciB_
                            currentConstrainedOut = d_46_ccB_
                            d_1_steps_ = (d_1_steps_) + (d_43_closeBudgetB_)
        if (d_1_steps_) > (maxSteps):
            cost = maxSteps
        elif True:
            cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

