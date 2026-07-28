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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the problem step by step using the symbolic variable names given. Express intermediate calculations using the variable names. End with: The final answer is <<EXPRESSION>> where EXPRESSION is a valid arithmetic expression using the variables (no curly braces, no ** operator, use ^ for powers)."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_narrowThreshold_: int
        d_3_narrowThreshold_ = 12
        d_4_freePhase_: int
        if (maxSteps) >= (5):
            d_4_freePhase_ = _dafny.euclidian_division((maxSteps) * (82), 100)
        elif True:
            d_4_freePhase_ = _dafny.euclidian_division(maxSteps, 2)
        d_5_forcedSpanDone_: bool
        d_5_forcedSpanDone_ = False
        with _dafny.label("0"):
            while (d_2_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_6_budgetLeft_: int
                        d_6_budgetLeft_ = (maxSteps) - (d_2_steps_)
                        d_7_shouldForce_: bool
                        d_7_shouldForce_ = (not(d_5_forcedSpanDone_)) and (((d_2_steps_) >= (d_4_freePhase_)) or ((d_6_budgetLeft_) <= (30)))
                        if (d_7_shouldForce_) and ((d_6_budgetLeft_) >= (3)):
                            d_8_og_: _dafny.Seq
                            d_9_oi_: bool
                            d_10_oc_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_8_og_ = out0_
                            d_9_oi_ = out1_
                            d_10_oc_ = out2_
                            generated = d_8_og_
                            insideConstrainedOut = d_9_oi_
                            currentConstrainedOut = d_10_oc_
                            d_2_steps_ = (d_2_steps_) + (1)
                            d_5_forcedSpanDone_ = True
                        elif (d_7_shouldForce_) and ((d_6_budgetLeft_) < (3)):
                            raise _dafny.Break("0")
                        elif True:
                            d_11_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_11_next_ = out3_
                            d_2_steps_ = (d_2_steps_) + (1)
                            if (d_11_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_11_next_]))
                                if (d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    out4_: _dafny.Seq
                                    out5_: bool
                                    out6_: _dafny.Seq
                                    out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                    generated = out4_
                                    insideConstrainedOut = out5_
                                    currentConstrainedOut = out6_
                    elif True:
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            d_12_cg_: _dafny.Seq
                            d_13_ci_: bool
                            d_14_cc_: _dafny.Seq
                            out7_: _dafny.Seq
                            out8_: bool
                            out9_: _dafny.Seq
                            out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_12_cg_ = out7_
                            d_13_ci_ = out8_
                            d_14_cc_ = out9_
                            generated = d_12_cg_
                            insideConstrainedOut = d_13_ci_
                            currentConstrainedOut = d_14_cc_
                            d_2_steps_ = (d_2_steps_) + (1)
                            if d_5_forcedSpanDone_:
                                raise _dafny.Break("0")
                        elif True:
                            d_15_budgetLeft2_: int
                            d_15_budgetLeft2_ = (maxSteps) - (d_2_steps_)
                            if (d_15_budgetLeft2_) <= (4):
                                d_16_closeBudget_: int
                                d_16_closeBudget_ = d_15_budgetLeft2_
                                d_17_cg_: _dafny.Seq
                                d_18_ci_: bool
                                d_19_cc_: _dafny.Seq
                                out10_: _dafny.Seq
                                out11_: bool
                                out12_: _dafny.Seq
                                out10_, out11_, out12_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_16_closeBudget_)
                                d_17_cg_ = out10_
                                d_18_ci_ = out11_
                                d_19_cc_ = out12_
                                generated = d_17_cg_
                                insideConstrainedOut = d_18_ci_
                                currentConstrainedOut = d_19_cc_
                                d_2_steps_ = (d_2_steps_) + (d_16_closeBudget_)
                                raise _dafny.Break("0")
                            elif True:
                                d_20_constrainedPrompt_: _dafny.Seq
                                d_20_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_21_next_: _dafny.Seq
                                out13_: _dafny.Seq
                                out13_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_20_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_3_narrowThreshold_, eosToken)
                                d_21_next_ = out13_
                                d_2_steps_ = (d_2_steps_) + (1)
                                if (d_21_next_) == (eosToken):
                                    if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_2_steps_) < (maxSteps)):
                                        d_22_cg_: _dafny.Seq
                                        d_23_ci_: bool
                                        d_24_cc_: _dafny.Seq
                                        out14_: _dafny.Seq
                                        out15_: bool
                                        out16_: _dafny.Seq
                                        out14_, out15_, out16_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                        d_22_cg_ = out14_
                                        d_23_ci_ = out15_
                                        d_24_cc_ = out16_
                                        generated = d_22_cg_
                                        insideConstrainedOut = d_23_ci_
                                        currentConstrainedOut = d_24_cc_
                                        d_2_steps_ = (d_2_steps_) + (1)
                                    elif (d_2_steps_) < (maxSteps):
                                        d_25_closeBudget_: int
                                        d_25_closeBudget_ = (maxSteps) - (d_2_steps_)
                                        d_26_cg_: _dafny.Seq
                                        d_27_ci_: bool
                                        d_28_cc_: _dafny.Seq
                                        out17_: _dafny.Seq
                                        out18_: bool
                                        out19_: _dafny.Seq
                                        out17_, out18_, out19_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_25_closeBudget_)
                                        d_26_cg_ = out17_
                                        d_27_ci_ = out18_
                                        d_28_cc_ = out19_
                                        generated = d_26_cg_
                                        insideConstrainedOut = d_27_ci_
                                        currentConstrainedOut = d_28_cc_
                                        d_2_steps_ = (d_2_steps_) + (d_25_closeBudget_)
                                    raise _dafny.Break("0")
                                elif True:
                                    d_29_ag_: _dafny.Seq
                                    d_30_ai_: bool
                                    d_31_ac_: _dafny.Seq
                                    out20_: _dafny.Seq
                                    out21_: bool
                                    out22_: _dafny.Seq
                                    out20_, out21_, out22_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_21_next_)
                                    d_29_ag_ = out20_
                                    d_30_ai_ = out21_
                                    d_31_ac_ = out22_
                                    generated = d_29_ag_
                                    insideConstrainedOut = d_30_ai_
                                    currentConstrainedOut = d_31_ac_
                    pass
            pass
        if ((not(insideConstrainedOut)) and (not(d_5_forcedSpanDone_))) and (((maxSteps) - (d_2_steps_)) >= (3)):
            d_32_og_: _dafny.Seq
            d_33_oi_: bool
            d_34_oc_: _dafny.Seq
            out23_: _dafny.Seq
            out24_: bool
            out25_: _dafny.Seq
            out23_, out24_, out25_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_32_og_ = out23_
            d_33_oi_ = out24_
            d_34_oc_ = out25_
            generated = d_32_og_
            insideConstrainedOut = d_33_oi_
            currentConstrainedOut = d_34_oc_
            d_2_steps_ = (d_2_steps_) + (1)
            d_35_innerSteps_: int
            d_35_innerSteps_ = 0
            d_36_innerBudget_: int
            d_36_innerBudget_ = (maxSteps) - (d_2_steps_)
            with _dafny.label("1_0"):
                while (d_35_innerSteps_) < (d_36_innerBudget_):
                    with _dafny.c_label("1_0"):
                        if not(insideConstrainedOut):
                            raise _dafny.Break("1_0")
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            d_37_cg_: _dafny.Seq
                            d_38_ci_: bool
                            d_39_cc_: _dafny.Seq
                            out26_: _dafny.Seq
                            out27_: bool
                            out28_: _dafny.Seq
                            out26_, out27_, out28_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_37_cg_ = out26_
                            d_38_ci_ = out27_
                            d_39_cc_ = out28_
                            generated = d_37_cg_
                            insideConstrainedOut = d_38_ci_
                            currentConstrainedOut = d_39_cc_
                            d_35_innerSteps_ = (d_35_innerSteps_) + (1)
                            raise _dafny.Break("1_0")
                        elif ((d_36_innerBudget_) - (d_35_innerSteps_)) <= (3):
                            d_40_closeBudget_: int
                            d_40_closeBudget_ = (d_36_innerBudget_) - (d_35_innerSteps_)
                            d_41_cg_: _dafny.Seq
                            d_42_ci_: bool
                            d_43_cc_: _dafny.Seq
                            out29_: _dafny.Seq
                            out30_: bool
                            out31_: _dafny.Seq
                            out29_, out30_, out31_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_40_closeBudget_)
                            d_41_cg_ = out29_
                            d_42_ci_ = out30_
                            d_43_cc_ = out31_
                            generated = d_41_cg_
                            insideConstrainedOut = d_42_ci_
                            currentConstrainedOut = d_43_cc_
                            d_35_innerSteps_ = (d_35_innerSteps_) + (d_40_closeBudget_)
                            raise _dafny.Break("1_0")
                        elif True:
                            d_44_constrainedPrompt_: _dafny.Seq
                            d_44_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_45_next_: _dafny.Seq
                            out32_: _dafny.Seq
                            out32_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_44_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_3_narrowThreshold_, eosToken)
                            d_45_next_ = out32_
                            d_35_innerSteps_ = (d_35_innerSteps_) + (1)
                            if (d_45_next_) == (eosToken):
                                if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_35_innerSteps_) < (d_36_innerBudget_)):
                                    d_46_cg_: _dafny.Seq
                                    d_47_ci_: bool
                                    d_48_cc_: _dafny.Seq
                                    out33_: _dafny.Seq
                                    out34_: bool
                                    out35_: _dafny.Seq
                                    out33_, out34_, out35_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_46_cg_ = out33_
                                    d_47_ci_ = out34_
                                    d_48_cc_ = out35_
                                    generated = d_46_cg_
                                    insideConstrainedOut = d_47_ci_
                                    currentConstrainedOut = d_48_cc_
                                    d_35_innerSteps_ = (d_35_innerSteps_) + (1)
                                elif (d_35_innerSteps_) < (d_36_innerBudget_):
                                    d_49_closeBudget_: int
                                    d_49_closeBudget_ = (d_36_innerBudget_) - (d_35_innerSteps_)
                                    d_50_cg_: _dafny.Seq
                                    d_51_ci_: bool
                                    d_52_cc_: _dafny.Seq
                                    out36_: _dafny.Seq
                                    out37_: bool
                                    out38_: _dafny.Seq
                                    out36_, out37_, out38_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_49_closeBudget_)
                                    d_50_cg_ = out36_
                                    d_51_ci_ = out37_
                                    d_52_cc_ = out38_
                                    generated = d_50_cg_
                                    insideConstrainedOut = d_51_ci_
                                    currentConstrainedOut = d_52_cc_
                                    d_35_innerSteps_ = (d_35_innerSteps_) + (d_49_closeBudget_)
                                raise _dafny.Break("1_0")
                            elif True:
                                d_53_ag_: _dafny.Seq
                                d_54_ai_: bool
                                d_55_ac_: _dafny.Seq
                                out39_: _dafny.Seq
                                out40_: bool
                                out41_: _dafny.Seq
                                out39_, out40_, out41_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_45_next_)
                                d_53_ag_ = out39_
                                d_54_ai_ = out40_
                                d_55_ac_ = out41_
                                generated = d_53_ag_
                                insideConstrainedOut = d_54_ai_
                                currentConstrainedOut = d_55_ac_
                        pass
                pass
            d_2_steps_ = (d_2_steps_) + (d_35_innerSteps_)
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

