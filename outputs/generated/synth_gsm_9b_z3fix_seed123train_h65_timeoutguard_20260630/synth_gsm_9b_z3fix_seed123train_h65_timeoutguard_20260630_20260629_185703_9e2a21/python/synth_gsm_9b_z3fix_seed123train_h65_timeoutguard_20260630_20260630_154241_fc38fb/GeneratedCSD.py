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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. Write intermediate symbolic expressions and the final answer inside << >> delimiters. Use plain variable names from the problem (no curly braces, no {n} notation). Allowed operators: +, -, *, /, //, %, int(). Do NOT use ** or ^. Example: <<n * price>>, <<int(total // n)>>, <<(a - b) * c + d>>.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_spanBudget_: int
        d_2_spanBudget_ = 55
        d_3_reasoningBudget_: int
        d_3_reasoningBudget_ = 550
        d_4_forceOpenThreshold_: int
        d_4_forceOpenThreshold_ = 100
        d_5_lastTokenWasLt_: bool
        d_5_lastTokenWasLt_ = False
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_6_remainingBudget_: int
                        d_6_remainingBudget_ = (maxSteps) - (d_1_steps_)
                        if (d_6_remainingBudget_) <= (3):
                            raise _dafny.Break("0")
                        elif ((d_6_remainingBudget_) <= (d_4_forceOpenThreshold_)) or ((d_1_steps_) >= (d_3_reasoningBudget_)):
                            d_7_og_: _dafny.Seq
                            d_8_oi_: bool
                            d_9_oc_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_7_og_ = out0_
                            d_8_oi_ = out1_
                            d_9_oc_ = out2_
                            generated = d_7_og_
                            insideConstrainedOut = d_8_oi_
                            currentConstrainedOut = d_9_oc_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_5_lastTokenWasLt_ = False
                            if (d_1_steps_) < (maxSteps):
                                d_10_rem_: int
                                d_10_rem_ = (maxSteps) - (d_1_steps_)
                                d_11_cb_: int
                                if (d_10_rem_) < (d_2_spanBudget_):
                                    d_11_cb_ = d_10_rem_
                                elif True:
                                    d_11_cb_ = d_2_spanBudget_
                                if (d_11_cb_) > (0):
                                    d_12_cg_: _dafny.Seq
                                    d_13_ci_: bool
                                    d_14_cc_: _dafny.Seq
                                    out3_: _dafny.Seq
                                    out4_: bool
                                    out5_: _dafny.Seq
                                    out3_, out4_, out5_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_11_cb_)
                                    d_12_cg_ = out3_
                                    d_13_ci_ = out4_
                                    d_14_cc_ = out5_
                                    generated = d_12_cg_
                                    insideConstrainedOut = d_13_ci_
                                    currentConstrainedOut = d_14_cc_
                                    d_1_steps_ = (d_1_steps_) + (d_11_cb_)
                        elif True:
                            d_15_next_: _dafny.Seq
                            out6_: _dafny.Seq
                            out6_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_15_next_ = out6_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_15_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                if (d_15_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_15_next_]))
                                    d_5_lastTokenWasLt_ = False
                                    d_16_og2_: _dafny.Seq
                                    d_17_oi2_: bool
                                    d_18_oc2_: _dafny.Seq
                                    out7_: _dafny.Seq
                                    out8_: bool
                                    out9_: _dafny.Seq
                                    out7_, out8_, out9_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                    d_16_og2_ = out7_
                                    d_17_oi2_ = out8_
                                    d_18_oc2_ = out9_
                                    generated = d_16_og2_
                                    insideConstrainedOut = d_17_oi2_
                                    currentConstrainedOut = d_18_oc2_
                                    d_19_rem2_: int
                                    d_19_rem2_ = (maxSteps) - (d_1_steps_)
                                    if (d_19_rem2_) > (0):
                                        d_20_cb2_: int
                                        if (d_19_rem2_) < (d_2_spanBudget_):
                                            d_20_cb2_ = d_19_rem2_
                                        elif True:
                                            d_20_cb2_ = d_2_spanBudget_
                                        d_21_cg2_: _dafny.Seq
                                        d_22_ci2_: bool
                                        d_23_cc2_: _dafny.Seq
                                        out10_: _dafny.Seq
                                        out11_: bool
                                        out12_: _dafny.Seq
                                        out10_, out11_, out12_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_20_cb2_)
                                        d_21_cg2_ = out10_
                                        d_22_ci2_ = out11_
                                        d_23_cc2_ = out12_
                                        generated = d_21_cg2_
                                        insideConstrainedOut = d_22_ci2_
                                        currentConstrainedOut = d_23_cc2_
                                        d_1_steps_ = (d_1_steps_) + (d_20_cb2_)
                                elif ((d_15_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<")))) and (d_5_lastTokenWasLt_):
                                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_15_next_]))
                                    d_5_lastTokenWasLt_ = False
                                    d_24_og3_: _dafny.Seq
                                    d_25_oi3_: bool
                                    d_26_oc3_: _dafny.Seq
                                    out13_: _dafny.Seq
                                    out14_: bool
                                    out15_: _dafny.Seq
                                    out13_, out14_, out15_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                    d_24_og3_ = out13_
                                    d_25_oi3_ = out14_
                                    d_26_oc3_ = out15_
                                    generated = d_24_og3_
                                    insideConstrainedOut = d_25_oi3_
                                    currentConstrainedOut = d_26_oc3_
                                    d_27_rem3_: int
                                    d_27_rem3_ = (maxSteps) - (d_1_steps_)
                                    if (d_27_rem3_) > (0):
                                        d_28_cb3_: int
                                        if (d_27_rem3_) < (d_2_spanBudget_):
                                            d_28_cb3_ = d_27_rem3_
                                        elif True:
                                            d_28_cb3_ = d_2_spanBudget_
                                        d_29_cg3_: _dafny.Seq
                                        d_30_ci3_: bool
                                        d_31_cc3_: _dafny.Seq
                                        out16_: _dafny.Seq
                                        out17_: bool
                                        out18_: _dafny.Seq
                                        out16_, out17_, out18_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_28_cb3_)
                                        d_29_cg3_ = out16_
                                        d_30_ci3_ = out17_
                                        d_31_cc3_ = out18_
                                        generated = d_29_cg3_
                                        insideConstrainedOut = d_30_ci3_
                                        currentConstrainedOut = d_31_cc3_
                                        d_1_steps_ = (d_1_steps_) + (d_28_cb3_)
                                elif True:
                                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_15_next_]))
                                    d_5_lastTokenWasLt_ = (d_15_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<")))
                    elif True:
                        d_32_constrainedPrompt_: _dafny.Seq
                        d_32_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            d_33_rem4_: int
                            d_33_rem4_ = (maxSteps) - (d_1_steps_)
                            if (d_33_rem4_) == (0):
                                raise _dafny.Break("0")
                            d_34_cb4_: int
                            if (d_33_rem4_) < (20):
                                d_34_cb4_ = d_33_rem4_
                            elif True:
                                d_34_cb4_ = 20
                            d_35_cg4_: _dafny.Seq
                            d_36_ci4_: bool
                            d_37_cc4_: _dafny.Seq
                            out19_: _dafny.Seq
                            out20_: bool
                            out21_: _dafny.Seq
                            out19_, out20_, out21_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_34_cb4_)
                            d_35_cg4_ = out19_
                            d_36_ci4_ = out20_
                            d_37_cc4_ = out21_
                            generated = d_35_cg4_
                            insideConstrainedOut = d_36_ci4_
                            currentConstrainedOut = d_37_cc4_
                            d_1_steps_ = (d_1_steps_) + (d_34_cb4_)
                            d_5_lastTokenWasLt_ = False
                        elif True:
                            d_38_next_: _dafny.Seq
                            out22_: _dafny.Seq
                            out22_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_32_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_38_next_ = out22_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_38_next_) == (eosToken):
                                d_39_rem5_: int
                                d_39_rem5_ = (maxSteps) - (d_1_steps_)
                                if (d_39_rem5_) == (0):
                                    raise _dafny.Break("0")
                                d_40_cb5_: int
                                if (d_39_rem5_) < (25):
                                    d_40_cb5_ = d_39_rem5_
                                elif True:
                                    d_40_cb5_ = 25
                                d_41_cg5_: _dafny.Seq
                                d_42_ci5_: bool
                                d_43_cc5_: _dafny.Seq
                                out23_: _dafny.Seq
                                out24_: bool
                                out25_: _dafny.Seq
                                out23_, out24_, out25_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_40_cb5_)
                                d_41_cg5_ = out23_
                                d_42_ci5_ = out24_
                                d_43_cc5_ = out25_
                                generated = d_41_cg5_
                                insideConstrainedOut = d_42_ci5_
                                currentConstrainedOut = d_43_cc5_
                                d_1_steps_ = (d_1_steps_) + (d_40_cb5_)
                                raise _dafny.Break("0")
                            elif True:
                                d_44_valid_: bool
                                out26_: bool
                                out26_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_38_next_)
                                d_44_valid_ = out26_
                                if d_44_valid_:
                                    d_45_ag_: _dafny.Seq
                                    d_46_ai_: bool
                                    d_47_ac_: _dafny.Seq
                                    out27_: _dafny.Seq
                                    out28_: bool
                                    out29_: _dafny.Seq
                                    out27_, out28_, out29_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_38_next_)
                                    d_45_ag_ = out27_
                                    d_46_ai_ = out28_
                                    d_47_ac_ = out29_
                                    generated = d_45_ag_
                                    insideConstrainedOut = d_46_ai_
                                    currentConstrainedOut = d_47_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_48_remainingA_: int
            d_48_remainingA_ = (maxSteps) - (d_1_steps_)
            d_49_closeBudgetA_: int
            if (d_48_remainingA_) < (50):
                d_49_closeBudgetA_ = d_48_remainingA_
            elif True:
                d_49_closeBudgetA_ = 50
            if (d_49_closeBudgetA_) > (0):
                d_50_cgA_: _dafny.Seq
                d_51_ciA_: bool
                d_52_ccA_: _dafny.Seq
                out30_: _dafny.Seq
                out31_: bool
                out32_: _dafny.Seq
                out30_, out31_, out32_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_49_closeBudgetA_)
                d_50_cgA_ = out30_
                d_51_ciA_ = out31_
                d_52_ccA_ = out32_
                generated = d_50_cgA_
                insideConstrainedOut = d_51_ciA_
                currentConstrainedOut = d_52_ccA_
                d_1_steps_ = (d_1_steps_) + (d_49_closeBudgetA_)
        if (not(insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
            d_53_genStr_: _dafny.Seq
            d_53_genStr_ = VerifiedDecoderAgent.CSDHelpers.PrefixToString(generated)
            d_54_openCount_: int
            d_54_openCount_ = VerifiedDecoderAgent.CSDHelpers.CountSubstring(d_53_genStr_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))
            if (d_54_openCount_) == (0):
                d_55_remainingB_: int
                d_55_remainingB_ = (maxSteps) - (d_1_steps_)
                if (d_55_remainingB_) >= (5):
                    d_56_ogB_: _dafny.Seq
                    d_57_oiB_: bool
                    d_58_ocB_: _dafny.Seq
                    out33_: _dafny.Seq
                    out34_: bool
                    out35_: _dafny.Seq
                    out33_, out34_, out35_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                    d_56_ogB_ = out33_
                    d_57_oiB_ = out34_
                    d_58_ocB_ = out35_
                    generated = d_56_ogB_
                    insideConstrainedOut = d_57_oiB_
                    currentConstrainedOut = d_58_ocB_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if (d_1_steps_) < (maxSteps):
                        d_59_remainingB2_: int
                        d_59_remainingB2_ = (maxSteps) - (d_1_steps_)
                        d_60_closeBudgetB_: int
                        if (d_59_remainingB2_) < (80):
                            d_60_closeBudgetB_ = d_59_remainingB2_
                        elif True:
                            d_60_closeBudgetB_ = 80
                        if (d_60_closeBudgetB_) > (0):
                            d_61_cgB_: _dafny.Seq
                            d_62_ciB_: bool
                            d_63_ccB_: _dafny.Seq
                            out36_: _dafny.Seq
                            out37_: bool
                            out38_: _dafny.Seq
                            out36_, out37_, out38_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_60_closeBudgetB_)
                            d_61_cgB_ = out36_
                            d_62_ciB_ = out37_
                            d_63_ccB_ = out38_
                            generated = d_61_cgB_
                            insideConstrainedOut = d_62_ciB_
                            currentConstrainedOut = d_63_ccB_
                            d_1_steps_ = (d_1_steps_) + (d_60_closeBudgetB_)
        if (d_1_steps_) > (maxSteps):
            cost = maxSteps
        elif True:
            cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

