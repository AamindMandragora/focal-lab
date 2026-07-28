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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. Write each intermediate calculation and the final answer inside <<expression>> delimiters. Use plain variable names (no curly braces), small numbers, and operators +, -, *, /, //, %, int(). Do not use ** or ^. Examples: <<n * price>>, <<int(a + b)>>, <<(x - y) * z // 60>>. The final answer must be the last <<expression>>.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_spanSteps_: int
        d_2_spanSteps_ = 0
        d_3_spanBudget_: int
        d_3_spanBudget_ = 60
        d_4_forceOpenThreshold_: int
        d_4_forceOpenThreshold_ = 150
        d_5_finalSpanBudget_: int
        d_5_finalSpanBudget_ = 130
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_6_remainingBudget_: int
                        d_6_remainingBudget_ = (maxSteps) - (d_1_steps_)
                        if (d_6_remainingBudget_) <= (3):
                            raise _dafny.Break("0")
                        elif (d_6_remainingBudget_) <= (d_4_forceOpenThreshold_):
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
                            d_2_spanSteps_ = 0
                            if (d_1_steps_) < (maxSteps):
                                d_10_rem_: int
                                d_10_rem_ = (maxSteps) - (d_1_steps_)
                                d_11_fb_: int
                                if (d_10_rem_) < (d_5_finalSpanBudget_):
                                    d_11_fb_ = d_10_rem_
                                elif True:
                                    d_11_fb_ = d_5_finalSpanBudget_
                                if (d_11_fb_) > (0):
                                    d_12_cg_: _dafny.Seq
                                    d_13_ci_: bool
                                    d_14_cc_: _dafny.Seq
                                    out3_: _dafny.Seq
                                    out4_: bool
                                    out5_: _dafny.Seq
                                    out3_, out4_, out5_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_11_fb_)
                                    d_12_cg_ = out3_
                                    d_13_ci_ = out4_
                                    d_14_cc_ = out5_
                                    generated = d_12_cg_
                                    insideConstrainedOut = d_13_ci_
                                    currentConstrainedOut = d_14_cc_
                                    d_1_steps_ = (d_1_steps_) + (d_11_fb_)
                                    d_2_spanSteps_ = 0
                            raise _dafny.Break("0")
                        elif True:
                            d_15_next_: _dafny.Seq
                            out6_: _dafny.Seq
                            out6_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_15_next_ = out6_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_15_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_15_next_]))
                    elif (d_2_spanSteps_) >= (d_3_spanBudget_):
                        d_16_remainingSteps_: int
                        d_16_remainingSteps_ = (maxSteps) - (d_1_steps_)
                        if (d_16_remainingSteps_) == (0):
                            raise _dafny.Break("0")
                        d_17_closeBudget2_: int
                        if (d_16_remainingSteps_) < (30):
                            d_17_closeBudget2_ = d_16_remainingSteps_
                        elif True:
                            d_17_closeBudget2_ = 30
                        d_18_cg2_: _dafny.Seq
                        d_19_ci2_: bool
                        d_20_cc2_: _dafny.Seq
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: _dafny.Seq
                        out7_, out8_, out9_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_17_closeBudget2_)
                        d_18_cg2_ = out7_
                        d_19_ci2_ = out8_
                        d_20_cc2_ = out9_
                        generated = d_18_cg2_
                        insideConstrainedOut = d_19_ci2_
                        currentConstrainedOut = d_20_cc2_
                        d_1_steps_ = (d_1_steps_) + (d_17_closeBudget2_)
                        d_2_spanSteps_ = 0
                    elif True:
                        d_21_constrainedPrompt_: _dafny.Seq
                        d_21_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_22_next_: _dafny.Seq
                        out10_: _dafny.Seq
                        out10_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_21_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('2e0'), eosToken)
                        d_22_next_ = out10_
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_2_spanSteps_ = (d_2_spanSteps_) + (1)
                        if (d_22_next_) == (eosToken):
                            d_23_remainingSteps_: int
                            d_23_remainingSteps_ = (maxSteps) - (d_1_steps_)
                            if (d_23_remainingSteps_) == (0):
                                raise _dafny.Break("0")
                            d_24_closeBudget3_: int
                            if (d_23_remainingSteps_) < (20):
                                d_24_closeBudget3_ = d_23_remainingSteps_
                            elif True:
                                d_24_closeBudget3_ = 20
                            d_25_cg3_: _dafny.Seq
                            d_26_ci3_: bool
                            d_27_cc3_: _dafny.Seq
                            out11_: _dafny.Seq
                            out12_: bool
                            out13_: _dafny.Seq
                            out11_, out12_, out13_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_24_closeBudget3_)
                            d_25_cg3_ = out11_
                            d_26_ci3_ = out12_
                            d_27_cc3_ = out13_
                            generated = d_25_cg3_
                            insideConstrainedOut = d_26_ci3_
                            currentConstrainedOut = d_27_cc3_
                            d_1_steps_ = (d_1_steps_) + (d_24_closeBudget3_)
                            d_2_spanSteps_ = 0
                            raise _dafny.Break("0")
                        elif True:
                            d_28_isComplete_: bool
                            d_28_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                            if d_28_isComplete_:
                                d_29_remainingSteps2_: int
                                d_29_remainingSteps2_ = (maxSteps) - (d_1_steps_)
                                if (d_29_remainingSteps2_) == (0):
                                    raise _dafny.Break("0")
                                d_30_closeBudget4_: int
                                if (d_29_remainingSteps2_) < (20):
                                    d_30_closeBudget4_ = d_29_remainingSteps2_
                                elif True:
                                    d_30_closeBudget4_ = 20
                                d_31_cg4_: _dafny.Seq
                                d_32_ci4_: bool
                                d_33_cc4_: _dafny.Seq
                                out14_: _dafny.Seq
                                out15_: bool
                                out16_: _dafny.Seq
                                out14_, out15_, out16_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_30_closeBudget4_)
                                d_31_cg4_ = out14_
                                d_32_ci4_ = out15_
                                d_33_cc4_ = out16_
                                generated = d_31_cg4_
                                insideConstrainedOut = d_32_ci4_
                                currentConstrainedOut = d_33_cc4_
                                d_1_steps_ = (d_1_steps_) + (d_30_closeBudget4_)
                                d_2_spanSteps_ = 0
                            elif True:
                                d_34_ag_: _dafny.Seq
                                d_35_ai_: bool
                                d_36_ac_: _dafny.Seq
                                out17_: _dafny.Seq
                                out18_: bool
                                out19_: _dafny.Seq
                                out17_, out18_, out19_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_22_next_)
                                d_34_ag_ = out17_
                                d_35_ai_ = out18_
                                d_36_ac_ = out19_
                                generated = d_34_ag_
                                insideConstrainedOut = d_35_ai_
                                currentConstrainedOut = d_36_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_37_remainingA_: int
            d_37_remainingA_ = (maxSteps) - (d_1_steps_)
            d_38_closeBudgetA_: int
            if (d_37_remainingA_) < (60):
                d_38_closeBudgetA_ = d_37_remainingA_
            elif True:
                d_38_closeBudgetA_ = 60
            if (d_38_closeBudgetA_) > (0):
                d_39_cgA_: _dafny.Seq
                d_40_ciA_: bool
                d_41_ccA_: _dafny.Seq
                out20_: _dafny.Seq
                out21_: bool
                out22_: _dafny.Seq
                out20_, out21_, out22_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_38_closeBudgetA_)
                d_39_cgA_ = out20_
                d_40_ciA_ = out21_
                d_41_ccA_ = out22_
                generated = d_39_cgA_
                insideConstrainedOut = d_40_ciA_
                currentConstrainedOut = d_41_ccA_
                d_1_steps_ = (d_1_steps_) + (d_38_closeBudgetA_)
        d_42_genStr_: _dafny.Seq
        d_42_genStr_ = VerifiedDecoderAgent.CSDHelpers.PrefixToString(generated)
        d_43_openCount_: int
        d_43_openCount_ = VerifiedDecoderAgent.CSDHelpers.CountSubstring(d_42_genStr_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))
        if (((d_43_openCount_) == (0)) and (not(insideConstrainedOut))) and ((d_1_steps_) < (maxSteps)):
            d_44_remainingB_: int
            d_44_remainingB_ = (maxSteps) - (d_1_steps_)
            if (d_44_remainingB_) >= (5):
                d_45_ogB_: _dafny.Seq
                d_46_oiB_: bool
                d_47_ocB_: _dafny.Seq
                out23_: _dafny.Seq
                out24_: bool
                out25_: _dafny.Seq
                out23_, out24_, out25_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                d_45_ogB_ = out23_
                d_46_oiB_ = out24_
                d_47_ocB_ = out25_
                generated = d_45_ogB_
                insideConstrainedOut = d_46_oiB_
                currentConstrainedOut = d_47_ocB_
                d_1_steps_ = (d_1_steps_) + (1)
                if (d_1_steps_) < (maxSteps):
                    d_48_remainingB2_: int
                    d_48_remainingB2_ = (maxSteps) - (d_1_steps_)
                    d_49_closeBudgetB_: int
                    if (d_48_remainingB2_) < (100):
                        d_49_closeBudgetB_ = d_48_remainingB2_
                    elif True:
                        d_49_closeBudgetB_ = 100
                    if (d_49_closeBudgetB_) > (0):
                        d_50_cgB_: _dafny.Seq
                        d_51_ciB_: bool
                        d_52_ccB_: _dafny.Seq
                        out26_: _dafny.Seq
                        out27_: bool
                        out28_: _dafny.Seq
                        out26_, out27_, out28_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_49_closeBudgetB_)
                        d_50_cgB_ = out26_
                        d_51_ciB_ = out27_
                        d_52_ccB_ = out28_
                        generated = d_50_cgB_
                        insideConstrainedOut = d_51_ciB_
                        currentConstrainedOut = d_52_ccB_
                        d_1_steps_ = (d_1_steps_) + (d_49_closeBudgetB_)
        if (d_1_steps_) > (maxSteps):
            cost = maxSteps
        elif True:
            cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

