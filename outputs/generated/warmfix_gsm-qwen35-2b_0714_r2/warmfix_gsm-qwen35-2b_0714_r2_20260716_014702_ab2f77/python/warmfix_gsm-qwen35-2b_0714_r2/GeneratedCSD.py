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
        if (maxSteps) == (0):
            return generated, insideConstrainedOut, currentConstrainedOut, cost
        d_1_guidance_: _dafny.Seq
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. At the very end, write exactly: The answer is <<EXPR>> where EXPR is a full arithmetic expression using only the symbolic variable names from the problem (like n, price, rate, count, etc.), numbers, and operators +, -, *, /, (, ). The expression must show the complete computation, not just a single number. No LaTeX, no {}, no **, no $."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_hasCompletedSpan_: bool
        d_3_hasCompletedSpan_ = False
        d_4_chunkSize_: int
        d_4_chunkSize_ = 40
        d_5_minInnerTokens_: int
        d_5_minInnerTokens_ = 6
        d_6_phase1Limit_: int
        d_6_phase1Limit_ = _dafny.euclidian_division((maxSteps) * (75), 100)
        if (d_6_phase1Limit_) == (0):
            d_6_phase1Limit_ = 1
        with _dafny.label("0"):
            while (((d_2_steps_) < (d_6_phase1Limit_)) and (not(insideConstrainedOut))) and (not(d_3_hasCompletedSpan_)):
                with _dafny.c_label("0"):
                    d_7_actualChunk_: int
                    d_7_actualChunk_ = d_4_chunkSize_
                    if ((d_2_steps_) + (d_7_actualChunk_)) > (d_6_phase1Limit_):
                        d_7_actualChunk_ = (d_6_phase1Limit_) - (d_2_steps_)
                    if (d_7_actualChunk_) == (0):
                        raise _dafny.Break("0")
                    d_8_cg_: _dafny.Seq
                    d_9_stoppedOnOpen_: bool
                    d_10_stoppedOnEos_: bool
                    d_11_stepsUsed_: int
                    out0_: _dafny.Seq
                    out1_: bool
                    out2_: bool
                    out3_: int
                    out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_7_actualChunk_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                    d_8_cg_ = out0_
                    d_9_stoppedOnOpen_ = out1_
                    d_10_stoppedOnEos_ = out2_
                    d_11_stepsUsed_ = out3_
                    generated = d_8_cg_
                    d_2_steps_ = (d_2_steps_) + (d_11_stepsUsed_)
                    if d_10_stoppedOnEos_:
                        raise _dafny.Break("0")
                    if d_9_stoppedOnOpen_:
                        d_12_eg_: _dafny.Seq
                        d_13_ei_: bool
                        d_14_ec_: _dafny.Seq
                        out4_: _dafny.Seq
                        out5_: bool
                        out6_: _dafny.Seq
                        out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                        d_12_eg_ = out4_
                        d_13_ei_ = out5_
                        d_14_ec_ = out6_
                        generated = d_12_eg_
                        insideConstrainedOut = d_13_ei_
                        currentConstrainedOut = d_14_ec_
                    pass
            pass
        d_15_innerStepLimit_: int
        d_15_innerStepLimit_ = 120
        d_16_innerSteps_: int
        d_16_innerSteps_ = 0
        d_17_innerTokenCount_: int
        d_17_innerTokenCount_ = 0
        with _dafny.label("1"):
            while (((insideConstrainedOut) and (not(d_3_hasCompletedSpan_))) and ((d_2_steps_) < (maxSteps))) and ((d_16_innerSteps_) < (d_15_innerStepLimit_)):
                with _dafny.c_label("1"):
                    if (d_17_innerTokenCount_) >= (d_5_minInnerTokens_):
                        d_18_cg_: _dafny.Seq
                        d_19_ci_: bool
                        d_20_cc_: _dafny.Seq
                        d_21_closed_: bool
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: _dafny.Seq
                        out10_: bool
                        out7_, out8_, out9_, out10_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                        d_18_cg_ = out7_
                        d_19_ci_ = out8_
                        d_20_cc_ = out9_
                        d_21_closed_ = out10_
                        if d_21_closed_:
                            generated = d_18_cg_
                            insideConstrainedOut = d_19_ci_
                            currentConstrainedOut = d_20_cc_
                            d_2_steps_ = (d_2_steps_) + (1)
                            d_3_hasCompletedSpan_ = True
                            raise _dafny.Break("1")
                    d_22_constrainedPrompt_: _dafny.Seq
                    d_22_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                    d_23_next_: _dafny.Seq
                    out11_: _dafny.Seq
                    out11_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_22_constrainedPrompt_, currentConstrainedOut, eosToken)
                    d_23_next_ = out11_
                    d_2_steps_ = (d_2_steps_) + (1)
                    d_16_innerSteps_ = (d_16_innerSteps_) + (1)
                    if (d_23_next_) == (eosToken):
                        raise _dafny.Break("1")
                    elif True:
                        d_24_ag_: _dafny.Seq
                        d_25_ai_: bool
                        d_26_ac_: _dafny.Seq
                        out12_: _dafny.Seq
                        out13_: bool
                        out14_: _dafny.Seq
                        out12_, out13_, out14_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_23_next_)
                        d_24_ag_ = out12_
                        d_25_ai_ = out13_
                        d_26_ac_ = out14_
                        generated = d_24_ag_
                        insideConstrainedOut = d_25_ai_
                        currentConstrainedOut = d_26_ac_
                        d_17_innerTokenCount_ = (d_17_innerTokenCount_) + (1)
                    pass
            pass
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_27_remaining3_: int
            d_27_remaining3_ = (maxSteps) - (d_2_steps_)
            d_28_spanBudget3_: int
            d_28_spanBudget3_ = d_27_remaining3_
            if (d_28_spanBudget3_) > (200):
                d_28_spanBudget3_ = 200
            if (d_28_spanBudget3_) > (0):
                d_29_wg_: _dafny.Seq
                d_30_wi_: bool
                d_31_wc_: _dafny.Seq
                out15_: _dafny.Seq
                out16_: bool
                out17_: _dafny.Seq
                out15_, out16_, out17_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_28_spanBudget3_)
                d_29_wg_ = out15_
                d_30_wi_ = out16_
                d_31_wc_ = out17_
                generated = d_29_wg_
                insideConstrainedOut = d_30_wi_
                currentConstrainedOut = d_31_wc_
                d_2_steps_ = (d_2_steps_) + (d_28_spanBudget3_)
                if not(insideConstrainedOut):
                    d_3_hasCompletedSpan_ = True
        with _dafny.label("2"):
            while (((d_2_steps_) < (d_6_phase1Limit_)) and (not(insideConstrainedOut))) and (not(d_3_hasCompletedSpan_)):
                with _dafny.c_label("2"):
                    d_32_next4_: _dafny.Seq
                    out18_: _dafny.Seq
                    out18_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                    d_32_next4_ = out18_
                    d_2_steps_ = (d_2_steps_) + (1)
                    if (d_32_next4_) == (eosToken):
                        raise _dafny.Break("2")
                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_32_next4_]))
                    if (d_32_next4_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                        d_33_eg4_: _dafny.Seq
                        d_34_ei4_: bool
                        d_35_ec4_: _dafny.Seq
                        out19_: _dafny.Seq
                        out20_: bool
                        out21_: _dafny.Seq
                        out19_, out20_, out21_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                        d_33_eg4_ = out19_
                        d_34_ei4_ = out20_
                        d_35_ec4_ = out21_
                        generated = d_33_eg4_
                        insideConstrainedOut = d_34_ei4_
                        currentConstrainedOut = d_35_ec4_
                    pass
            pass
        d_36_innerSteps4_: int
        d_36_innerSteps4_ = 0
        d_37_innerTokenCount4_: int
        d_37_innerTokenCount4_ = 0
        with _dafny.label("3"):
            while (((insideConstrainedOut) and (not(d_3_hasCompletedSpan_))) and ((d_2_steps_) < (maxSteps))) and ((d_36_innerSteps4_) < (d_15_innerStepLimit_)):
                with _dafny.c_label("3"):
                    if (d_37_innerTokenCount4_) >= (d_5_minInnerTokens_):
                        d_38_cg4_: _dafny.Seq
                        d_39_ci4_: bool
                        d_40_cc4_: _dafny.Seq
                        d_41_closed4_: bool
                        out22_: _dafny.Seq
                        out23_: bool
                        out24_: _dafny.Seq
                        out25_: bool
                        out22_, out23_, out24_, out25_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                        d_38_cg4_ = out22_
                        d_39_ci4_ = out23_
                        d_40_cc4_ = out24_
                        d_41_closed4_ = out25_
                        if d_41_closed4_:
                            generated = d_38_cg4_
                            insideConstrainedOut = d_39_ci4_
                            currentConstrainedOut = d_40_cc4_
                            d_2_steps_ = (d_2_steps_) + (1)
                            d_3_hasCompletedSpan_ = True
                            raise _dafny.Break("3")
                    d_42_constrainedPrompt4_: _dafny.Seq
                    d_42_constrainedPrompt4_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                    d_43_next4b_: _dafny.Seq
                    out26_: _dafny.Seq
                    out26_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_42_constrainedPrompt4_, currentConstrainedOut, eosToken)
                    d_43_next4b_ = out26_
                    d_2_steps_ = (d_2_steps_) + (1)
                    d_36_innerSteps4_ = (d_36_innerSteps4_) + (1)
                    if (d_43_next4b_) == (eosToken):
                        raise _dafny.Break("3")
                    elif True:
                        d_44_ag4_: _dafny.Seq
                        d_45_ai4_: bool
                        d_46_ac4_: _dafny.Seq
                        out27_: _dafny.Seq
                        out28_: bool
                        out29_: _dafny.Seq
                        out27_, out28_, out29_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_43_next4b_)
                        d_44_ag4_ = out27_
                        d_45_ai4_ = out28_
                        d_46_ac4_ = out29_
                        generated = d_44_ag4_
                        insideConstrainedOut = d_45_ai4_
                        currentConstrainedOut = d_46_ac4_
                        d_37_innerTokenCount4_ = (d_37_innerTokenCount4_) + (1)
                    pass
            pass
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_47_remaining4c_: int
            d_47_remaining4c_ = (maxSteps) - (d_2_steps_)
            d_48_spanBudget4c_: int
            d_48_spanBudget4c_ = d_47_remaining4c_
            if (d_48_spanBudget4c_) > (150):
                d_48_spanBudget4c_ = 150
            if (d_48_spanBudget4c_) > (0):
                d_49_wg4c_: _dafny.Seq
                d_50_wi4c_: bool
                d_51_wc4c_: _dafny.Seq
                out30_: _dafny.Seq
                out31_: bool
                out32_: _dafny.Seq
                out30_, out31_, out32_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_48_spanBudget4c_)
                d_49_wg4c_ = out30_
                d_50_wi4c_ = out31_
                d_51_wc4c_ = out32_
                generated = d_49_wg4c_
                insideConstrainedOut = d_50_wi4c_
                currentConstrainedOut = d_51_wc4c_
                d_2_steps_ = (d_2_steps_) + (d_48_spanBudget4c_)
                if not(insideConstrainedOut):
                    d_3_hasCompletedSpan_ = True
        if ((not(insideConstrainedOut)) and (not(d_3_hasCompletedSpan_))) and (((d_2_steps_) + (2)) <= (maxSteps)):
            d_52_fg_: _dafny.Seq
            d_53_fi_: bool
            d_54_fc_: _dafny.Seq
            out33_: _dafny.Seq
            out34_: bool
            out35_: _dafny.Seq
            out33_, out34_, out35_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_52_fg_ = out33_
            d_53_fi_ = out34_
            d_54_fc_ = out35_
            generated = d_52_fg_
            insideConstrainedOut = d_53_fi_
            currentConstrainedOut = d_54_fc_
            d_2_steps_ = (d_2_steps_) + (1)
            d_55_innerSteps5_: int
            d_55_innerSteps5_ = 0
            d_56_innerTokenCount5_: int
            d_56_innerTokenCount5_ = 0
            d_57_innerLimit5_: int
            d_57_innerLimit5_ = 80
            with _dafny.label("8_0"):
                while (((insideConstrainedOut) and (not(d_3_hasCompletedSpan_))) and ((d_2_steps_) < (maxSteps))) and ((d_55_innerSteps5_) < (d_57_innerLimit5_)):
                    with _dafny.c_label("8_0"):
                        if (d_56_innerTokenCount5_) >= (d_5_minInnerTokens_):
                            d_58_cg5_: _dafny.Seq
                            d_59_ci5_: bool
                            d_60_cc5_: _dafny.Seq
                            d_61_closed5_: bool
                            out36_: _dafny.Seq
                            out37_: bool
                            out38_: _dafny.Seq
                            out39_: bool
                            out36_, out37_, out38_, out39_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                            d_58_cg5_ = out36_
                            d_59_ci5_ = out37_
                            d_60_cc5_ = out38_
                            d_61_closed5_ = out39_
                            if d_61_closed5_:
                                generated = d_58_cg5_
                                insideConstrainedOut = d_59_ci5_
                                currentConstrainedOut = d_60_cc5_
                                d_2_steps_ = (d_2_steps_) + (1)
                                d_3_hasCompletedSpan_ = True
                                raise _dafny.Break("8_0")
                        d_62_constrainedPrompt5_: _dafny.Seq
                        d_62_constrainedPrompt5_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_63_next5_: _dafny.Seq
                        out40_: _dafny.Seq
                        out40_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_62_constrainedPrompt5_, currentConstrainedOut, eosToken)
                        d_63_next5_ = out40_
                        d_2_steps_ = (d_2_steps_) + (1)
                        d_55_innerSteps5_ = (d_55_innerSteps5_) + (1)
                        if (d_63_next5_) == (eosToken):
                            raise _dafny.Break("8_0")
                        elif True:
                            d_64_ag5_: _dafny.Seq
                            d_65_ai5_: bool
                            d_66_ac5_: _dafny.Seq
                            out41_: _dafny.Seq
                            out42_: bool
                            out43_: _dafny.Seq
                            out41_, out42_, out43_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_63_next5_)
                            d_64_ag5_ = out41_
                            d_65_ai5_ = out42_
                            d_66_ac5_ = out43_
                            generated = d_64_ag5_
                            insideConstrainedOut = d_65_ai5_
                            currentConstrainedOut = d_66_ac5_
                            d_56_innerTokenCount5_ = (d_56_innerTokenCount5_) + (1)
                        pass
                pass
            if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
                d_67_remainBudget5_: int
                d_67_remainBudget5_ = (maxSteps) - (d_2_steps_)
                if (d_67_remainBudget5_) > (60):
                    d_67_remainBudget5_ = 60
                if (d_67_remainBudget5_) > (0):
                    d_68_wg5_: _dafny.Seq
                    d_69_wi5_: bool
                    d_70_wc5_: _dafny.Seq
                    out44_: _dafny.Seq
                    out45_: bool
                    out46_: _dafny.Seq
                    out44_, out45_, out46_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_67_remainBudget5_)
                    d_68_wg5_ = out44_
                    d_69_wi5_ = out45_
                    d_70_wc5_ = out46_
                    generated = d_68_wg5_
                    insideConstrainedOut = d_69_wi5_
                    currentConstrainedOut = d_70_wc5_
                    d_2_steps_ = (d_2_steps_) + (d_67_remainBudget5_)
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_71_finalBudget_: int
            d_71_finalBudget_ = (maxSteps) - (d_2_steps_)
            if (d_71_finalBudget_) > (0):
                d_72_wgf_: _dafny.Seq
                d_73_wif_: bool
                d_74_wcf_: _dafny.Seq
                out47_: _dafny.Seq
                out48_: bool
                out49_: _dafny.Seq
                out47_, out48_, out49_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_71_finalBudget_)
                d_72_wgf_ = out47_
                d_73_wif_ = out48_
                d_74_wcf_ = out49_
                generated = d_72_wgf_
                insideConstrainedOut = d_73_wif_
                currentConstrainedOut = d_74_wcf_
                d_2_steps_ = (d_2_steps_) + (d_71_finalBudget_)
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

