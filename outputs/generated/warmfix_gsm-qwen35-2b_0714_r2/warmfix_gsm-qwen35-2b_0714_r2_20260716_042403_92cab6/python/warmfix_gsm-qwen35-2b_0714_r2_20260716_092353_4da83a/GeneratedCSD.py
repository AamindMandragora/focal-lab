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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. Write the final answer as: The answer is EXPR where EXPR is a plain arithmetic expression. Rules: (1) Use EXACTLY the variable names from the problem curly braces, without the braces. For example if the problem has {frac_1} write frac_1, if {n1} write n1, if {price} write price. (2) Use int() when multiplying an integer by a fraction: int(n * frac). (3) Only use: variable names, numbers, +, -, *, /, (, ), int(). (4) No LaTeX, no dollar signs, no backslashes, no curly braces in the expression. Good EXPR examples: n * price - discount, int(n * frac_1 * frac_2), count * (n1 + n2 + n3 + n4 + n5), usage * int(price * (1 + percent / 100)) + extra_price")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_hasCompletedSpan_: bool
        d_2_hasCompletedSpan_ = False
        d_3_chunkSize_: int
        d_3_chunkSize_ = 40
        d_4_innerStepLimit_: int
        d_4_innerStepLimit_ = 80
        d_5_phase1Limit_: int
        d_5_phase1Limit_ = _dafny.euclidian_division((maxSteps) * (75), 100)
        if ((d_5_phase1Limit_) == (0)) and ((maxSteps) > (0)):
            d_5_phase1Limit_ = 1
        with _dafny.label("0"):
            while (((d_1_steps_) < (d_5_phase1Limit_)) and (not(insideConstrainedOut))) and (not(d_2_hasCompletedSpan_)):
                with _dafny.c_label("0"):
                    d_6_actualChunk_: int
                    d_6_actualChunk_ = d_3_chunkSize_
                    if ((d_1_steps_) + (d_6_actualChunk_)) > (d_5_phase1Limit_):
                        d_6_actualChunk_ = (d_5_phase1Limit_) - (d_1_steps_)
                    if (d_6_actualChunk_) == (0):
                        raise _dafny.Break("0")
                    d_7_cg_: _dafny.Seq
                    d_8_soo_: bool
                    d_9_soe_: bool
                    d_10_su_: int
                    out0_: _dafny.Seq
                    out1_: bool
                    out2_: bool
                    out3_: int
                    out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_6_actualChunk_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                    d_7_cg_ = out0_
                    d_8_soo_ = out1_
                    d_9_soe_ = out2_
                    d_10_su_ = out3_
                    generated = d_7_cg_
                    d_1_steps_ = (d_1_steps_) + (d_10_su_)
                    if d_9_soe_:
                        raise _dafny.Break("0")
                    if d_8_soo_:
                        d_11_eg_: _dafny.Seq
                        d_12_ei_: bool
                        d_13_ec_: _dafny.Seq
                        out4_: _dafny.Seq
                        out5_: bool
                        out6_: _dafny.Seq
                        out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                        d_11_eg_ = out4_
                        d_12_ei_ = out5_
                        d_13_ec_ = out6_
                        generated = d_11_eg_
                        insideConstrainedOut = d_12_ei_
                        currentConstrainedOut = d_13_ec_
                    pass
            pass
        d_14_innerSteps_: int
        d_14_innerSteps_ = 0
        with _dafny.label("1"):
            while (((insideConstrainedOut) and (not(d_2_hasCompletedSpan_))) and ((d_1_steps_) < (maxSteps))) and ((d_14_innerSteps_) < (d_4_innerStepLimit_)):
                with _dafny.c_label("1"):
                    d_15_cg_: _dafny.Seq
                    d_16_ci_: bool
                    d_17_cc_: _dafny.Seq
                    d_18_closed_: bool
                    out7_: _dafny.Seq
                    out8_: bool
                    out9_: _dafny.Seq
                    out10_: bool
                    out7_, out8_, out9_, out10_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                    d_15_cg_ = out7_
                    d_16_ci_ = out8_
                    d_17_cc_ = out9_
                    d_18_closed_ = out10_
                    if d_18_closed_:
                        generated = d_15_cg_
                        insideConstrainedOut = d_16_ci_
                        currentConstrainedOut = d_17_cc_
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_2_hasCompletedSpan_ = True
                    elif True:
                        d_19_cp_: _dafny.Seq
                        d_19_cp_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_20_next_: _dafny.Seq
                        out11_: _dafny.Seq
                        out11_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_19_cp_, currentConstrainedOut, eosToken)
                        d_20_next_ = out11_
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_14_innerSteps_ = (d_14_innerSteps_) + (1)
                        if (d_20_next_) == (eosToken):
                            raise _dafny.Break("1")
                        d_21_ag_: _dafny.Seq
                        d_22_ai_: bool
                        d_23_ac_: _dafny.Seq
                        out12_: _dafny.Seq
                        out13_: bool
                        out14_: _dafny.Seq
                        out12_, out13_, out14_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_20_next_)
                        d_21_ag_ = out12_
                        d_22_ai_ = out13_
                        d_23_ac_ = out14_
                        generated = d_21_ag_
                        insideConstrainedOut = d_22_ai_
                        currentConstrainedOut = d_23_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_24_bud_: int
            d_24_bud_ = (maxSteps) - (d_1_steps_)
            d_25_wg_: _dafny.Seq
            d_26_wi_: bool
            d_27_wc_: _dafny.Seq
            out15_: _dafny.Seq
            out16_: bool
            out17_: _dafny.Seq
            out15_, out16_, out17_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_24_bud_)
            d_25_wg_ = out15_
            d_26_wi_ = out16_
            d_27_wc_ = out17_
            generated = d_25_wg_
            insideConstrainedOut = d_26_wi_
            currentConstrainedOut = d_27_wc_
            d_1_steps_ = (d_1_steps_) + (d_24_bud_)
            if not(insideConstrainedOut):
                d_2_hasCompletedSpan_ = True
        with _dafny.label("2"):
            while (((d_1_steps_) < (d_5_phase1Limit_)) and (not(insideConstrainedOut))) and (not(d_2_hasCompletedSpan_)):
                with _dafny.c_label("2"):
                    d_28_next_: _dafny.Seq
                    out18_: _dafny.Seq
                    out18_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                    d_28_next_ = out18_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if (d_28_next_) == (eosToken):
                        raise _dafny.Break("2")
                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_28_next_]))
                    if VerifiedDecoderAgent.default__.RenderedEndsWith(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                        d_29_eg_: _dafny.Seq
                        d_30_ei_: bool
                        d_31_ec_: _dafny.Seq
                        out19_: _dafny.Seq
                        out20_: bool
                        out21_: _dafny.Seq
                        out19_, out20_, out21_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                        d_29_eg_ = out19_
                        d_30_ei_ = out20_
                        d_31_ec_ = out21_
                        generated = d_29_eg_
                        insideConstrainedOut = d_30_ei_
                        currentConstrainedOut = d_31_ec_
                    pass
            pass
        d_32_innerSteps2_: int
        d_32_innerSteps2_ = 0
        with _dafny.label("3"):
            while (((insideConstrainedOut) and (not(d_2_hasCompletedSpan_))) and ((d_1_steps_) < (maxSteps))) and ((d_32_innerSteps2_) < (d_4_innerStepLimit_)):
                with _dafny.c_label("3"):
                    d_33_cg2_: _dafny.Seq
                    d_34_ci2_: bool
                    d_35_cc2_: _dafny.Seq
                    d_36_closed2_: bool
                    out22_: _dafny.Seq
                    out23_: bool
                    out24_: _dafny.Seq
                    out25_: bool
                    out22_, out23_, out24_, out25_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                    d_33_cg2_ = out22_
                    d_34_ci2_ = out23_
                    d_35_cc2_ = out24_
                    d_36_closed2_ = out25_
                    if d_36_closed2_:
                        generated = d_33_cg2_
                        insideConstrainedOut = d_34_ci2_
                        currentConstrainedOut = d_35_cc2_
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_2_hasCompletedSpan_ = True
                    elif True:
                        d_37_cp2_: _dafny.Seq
                        d_37_cp2_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_38_next2_: _dafny.Seq
                        out26_: _dafny.Seq
                        out26_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_37_cp2_, currentConstrainedOut, eosToken)
                        d_38_next2_ = out26_
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_32_innerSteps2_ = (d_32_innerSteps2_) + (1)
                        if (d_38_next2_) == (eosToken):
                            raise _dafny.Break("3")
                        d_39_ag2_: _dafny.Seq
                        d_40_ai2_: bool
                        d_41_ac2_: _dafny.Seq
                        out27_: _dafny.Seq
                        out28_: bool
                        out29_: _dafny.Seq
                        out27_, out28_, out29_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_38_next2_)
                        d_39_ag2_ = out27_
                        d_40_ai2_ = out28_
                        d_41_ac2_ = out29_
                        generated = d_39_ag2_
                        insideConstrainedOut = d_40_ai2_
                        currentConstrainedOut = d_41_ac2_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_42_bud2_: int
            d_42_bud2_ = (maxSteps) - (d_1_steps_)
            d_43_wg2_: _dafny.Seq
            d_44_wi2_: bool
            d_45_wc2_: _dafny.Seq
            out30_: _dafny.Seq
            out31_: bool
            out32_: _dafny.Seq
            out30_, out31_, out32_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_42_bud2_)
            d_43_wg2_ = out30_
            d_44_wi2_ = out31_
            d_45_wc2_ = out32_
            generated = d_43_wg2_
            insideConstrainedOut = d_44_wi2_
            currentConstrainedOut = d_45_wc2_
            d_1_steps_ = (d_1_steps_) + (d_42_bud2_)
            if not(insideConstrainedOut):
                d_2_hasCompletedSpan_ = True
        if ((not(insideConstrainedOut)) and (not(d_2_hasCompletedSpan_))) and (((d_1_steps_) + (2)) <= (maxSteps)):
            d_46_fg_: _dafny.Seq
            d_47_fi_: bool
            d_48_fc_: _dafny.Seq
            out33_: _dafny.Seq
            out34_: bool
            out35_: _dafny.Seq
            out33_, out34_, out35_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_46_fg_ = out33_
            d_47_fi_ = out34_
            d_48_fc_ = out35_
            generated = d_46_fg_
            insideConstrainedOut = d_47_fi_
            currentConstrainedOut = d_48_fc_
            d_1_steps_ = (d_1_steps_) + (1)
            d_49_innerSteps3_: int
            d_49_innerSteps3_ = 0
            d_50_innerLimit3_: int
            d_50_innerLimit3_ = 40
            with _dafny.label("7_0"):
                while (((insideConstrainedOut) and (not(d_2_hasCompletedSpan_))) and ((d_1_steps_) < (maxSteps))) and ((d_49_innerSteps3_) < (d_50_innerLimit3_)):
                    with _dafny.c_label("7_0"):
                        d_51_cg3_: _dafny.Seq
                        d_52_ci3_: bool
                        d_53_cc3_: _dafny.Seq
                        d_54_closed3_: bool
                        out36_: _dafny.Seq
                        out37_: bool
                        out38_: _dafny.Seq
                        out39_: bool
                        out36_, out37_, out38_, out39_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                        d_51_cg3_ = out36_
                        d_52_ci3_ = out37_
                        d_53_cc3_ = out38_
                        d_54_closed3_ = out39_
                        if d_54_closed3_:
                            generated = d_51_cg3_
                            insideConstrainedOut = d_52_ci3_
                            currentConstrainedOut = d_53_cc3_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_2_hasCompletedSpan_ = True
                        elif True:
                            d_55_cp3_: _dafny.Seq
                            d_55_cp3_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_56_next3_: _dafny.Seq
                            out40_: _dafny.Seq
                            out40_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_55_cp3_, currentConstrainedOut, eosToken)
                            d_56_next3_ = out40_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_49_innerSteps3_ = (d_49_innerSteps3_) + (1)
                            if (d_56_next3_) == (eosToken):
                                raise _dafny.Break("7_0")
                            d_57_ag3_: _dafny.Seq
                            d_58_ai3_: bool
                            d_59_ac3_: _dafny.Seq
                            out41_: _dafny.Seq
                            out42_: bool
                            out43_: _dafny.Seq
                            out41_, out42_, out43_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_56_next3_)
                            d_57_ag3_ = out41_
                            d_58_ai3_ = out42_
                            d_59_ac3_ = out43_
                            generated = d_57_ag3_
                            insideConstrainedOut = d_58_ai3_
                            currentConstrainedOut = d_59_ac3_
                        pass
                pass
            if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
                d_60_bud3_: int
                d_60_bud3_ = (maxSteps) - (d_1_steps_)
                d_61_wg3_: _dafny.Seq
                d_62_wi3_: bool
                d_63_wc3_: _dafny.Seq
                out44_: _dafny.Seq
                out45_: bool
                out46_: _dafny.Seq
                out44_, out45_, out46_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_60_bud3_)
                d_61_wg3_ = out44_
                d_62_wi3_ = out45_
                d_63_wc3_ = out46_
                generated = d_61_wg3_
                insideConstrainedOut = d_62_wi3_
                currentConstrainedOut = d_63_wc3_
                d_1_steps_ = (d_1_steps_) + (d_60_bud3_)
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_64_bud4_: int
            d_64_bud4_ = (maxSteps) - (d_1_steps_)
            if (d_64_bud4_) > (0):
                d_65_wg4_: _dafny.Seq
                d_66_wi4_: bool
                d_67_wc4_: _dafny.Seq
                out47_: _dafny.Seq
                out48_: bool
                out49_: _dafny.Seq
                out47_, out48_, out49_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_64_bud4_)
                d_65_wg4_ = out47_
                d_66_wi4_ = out48_
                d_67_wc4_ = out49_
                generated = d_65_wg4_
                insideConstrainedOut = d_66_wi4_
                currentConstrainedOut = d_67_wc4_
                d_1_steps_ = (d_1_steps_) + (d_64_bud4_)
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

