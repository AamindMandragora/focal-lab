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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. At the very end write exactly: The answer is EXPR where EXPR is a single arithmetic expression. Rules: (1) Use EXACTLY the variable names from the problem without braces, e.g. n1 not {n1}, frac_1 not {frac_1}. (2) Combine ALL relevant quantities with operators. (3) Use int() for integer truncation of fractions: int(n * frac). (4) Only use: variable names, numbers, +, -, *, /, (, ), int(). (5) No LaTeX, no dollar signs, no backslashes. Good examples: n * price - discount, int(n * frac_1) + base, (a + b) * rate / 60.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_warmupMax_: int
        d_2_warmupMax_ = 8
        d_3_phase1Limit_: int
        d_3_phase1Limit_ = _dafny.euclidian_division((maxSteps) * (4), 5)
        if ((d_3_phase1Limit_) == (0)) and ((maxSteps) > (0)):
            d_3_phase1Limit_ = 1
        with _dafny.label("0"):
            while ((d_1_steps_) < (d_3_phase1Limit_)) and (not(insideConstrainedOut)):
                with _dafny.c_label("0"):
                    d_4_chunkSz_: int
                    d_4_chunkSz_ = 40
                    if ((d_1_steps_) + (d_4_chunkSz_)) > (d_3_phase1Limit_):
                        d_4_chunkSz_ = (d_3_phase1Limit_) - (d_1_steps_)
                    if (d_4_chunkSz_) == (0):
                        raise _dafny.Break("0")
                    d_5_cg_: _dafny.Seq
                    d_6_soo_: bool
                    d_7_soe_: bool
                    d_8_su_: int
                    out0_: _dafny.Seq
                    out1_: bool
                    out2_: bool
                    out3_: int
                    out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_4_chunkSz_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                    d_5_cg_ = out0_
                    d_6_soo_ = out1_
                    d_7_soe_ = out2_
                    d_8_su_ = out3_
                    generated = d_5_cg_
                    d_1_steps_ = (d_1_steps_) + (d_8_su_)
                    if d_7_soe_:
                        raise _dafny.Break("0")
                    if d_6_soo_:
                        d_9_eg_: _dafny.Seq
                        d_10_ei_: bool
                        d_11_ec_: _dafny.Seq
                        out4_: _dafny.Seq
                        out5_: bool
                        out6_: _dafny.Seq
                        out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                        d_9_eg_ = out4_
                        d_10_ei_ = out5_
                        d_11_ec_ = out6_
                        generated = d_9_eg_
                        insideConstrainedOut = d_10_ei_
                        currentConstrainedOut = d_11_ec_
                    pass
            pass
        d_12_warmup1_: int
        d_12_warmup1_ = 0
        with _dafny.label("1"):
            while ((insideConstrainedOut) and ((d_1_steps_) < (maxSteps))) and ((d_12_warmup1_) < (d_2_warmupMax_)):
                with _dafny.c_label("1"):
                    d_13_cp_: _dafny.Seq
                    d_13_cp_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                    d_14_next_: _dafny.Seq
                    out7_: _dafny.Seq
                    out7_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_13_cp_, currentConstrainedOut, eosToken)
                    d_14_next_ = out7_
                    d_1_steps_ = (d_1_steps_) + (1)
                    d_12_warmup1_ = (d_12_warmup1_) + (1)
                    if (d_14_next_) == (eosToken):
                        raise _dafny.Break("1")
                    d_15_ag_: _dafny.Seq
                    d_16_ai_: bool
                    d_17_ac_: _dafny.Seq
                    out8_: _dafny.Seq
                    out9_: bool
                    out10_: _dafny.Seq
                    out8_, out9_, out10_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_14_next_)
                    d_15_ag_ = out8_
                    d_16_ai_ = out9_
                    d_17_ac_ = out10_
                    generated = d_15_ag_
                    insideConstrainedOut = d_16_ai_
                    currentConstrainedOut = d_17_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_18_bud1_: int
            d_18_bud1_ = (maxSteps) - (d_1_steps_)
            d_19_wg1_: _dafny.Seq
            d_20_wi1_: bool
            d_21_wc1_: _dafny.Seq
            out11_: _dafny.Seq
            out12_: bool
            out13_: _dafny.Seq
            out11_, out12_, out13_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_18_bud1_)
            d_19_wg1_ = out11_
            d_20_wi1_ = out12_
            d_21_wc1_ = out13_
            generated = d_19_wg1_
            insideConstrainedOut = d_20_wi1_
            currentConstrainedOut = d_21_wc1_
            d_1_steps_ = (d_1_steps_) + (d_18_bud1_)
        with _dafny.label("2"):
            while ((d_1_steps_) < (d_3_phase1Limit_)) and (not(insideConstrainedOut)):
                with _dafny.c_label("2"):
                    d_22_next_: _dafny.Seq
                    out14_: _dafny.Seq
                    out14_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                    d_22_next_ = out14_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if (d_22_next_) == (eosToken):
                        raise _dafny.Break("2")
                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_22_next_]))
                    if VerifiedDecoderAgent.default__.RenderedEndsWith(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                        d_23_eg_: _dafny.Seq
                        d_24_ei_: bool
                        d_25_ec_: _dafny.Seq
                        out15_: _dafny.Seq
                        out16_: bool
                        out17_: _dafny.Seq
                        out15_, out16_, out17_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                        d_23_eg_ = out15_
                        d_24_ei_ = out16_
                        d_25_ec_ = out17_
                        generated = d_23_eg_
                        insideConstrainedOut = d_24_ei_
                        currentConstrainedOut = d_25_ec_
                    pass
            pass
        d_26_warmup2_: int
        d_26_warmup2_ = 0
        with _dafny.label("3"):
            while ((insideConstrainedOut) and ((d_1_steps_) < (maxSteps))) and ((d_26_warmup2_) < (d_2_warmupMax_)):
                with _dafny.c_label("3"):
                    d_27_cp_: _dafny.Seq
                    d_27_cp_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                    d_28_next_: _dafny.Seq
                    out18_: _dafny.Seq
                    out18_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_27_cp_, currentConstrainedOut, eosToken)
                    d_28_next_ = out18_
                    d_1_steps_ = (d_1_steps_) + (1)
                    d_26_warmup2_ = (d_26_warmup2_) + (1)
                    if (d_28_next_) == (eosToken):
                        raise _dafny.Break("3")
                    d_29_ag_: _dafny.Seq
                    d_30_ai_: bool
                    d_31_ac_: _dafny.Seq
                    out19_: _dafny.Seq
                    out20_: bool
                    out21_: _dafny.Seq
                    out19_, out20_, out21_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_28_next_)
                    d_29_ag_ = out19_
                    d_30_ai_ = out20_
                    d_31_ac_ = out21_
                    generated = d_29_ag_
                    insideConstrainedOut = d_30_ai_
                    currentConstrainedOut = d_31_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_32_bud2_: int
            d_32_bud2_ = (maxSteps) - (d_1_steps_)
            d_33_wg2_: _dafny.Seq
            d_34_wi2_: bool
            d_35_wc2_: _dafny.Seq
            out22_: _dafny.Seq
            out23_: bool
            out24_: _dafny.Seq
            out22_, out23_, out24_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_32_bud2_)
            d_33_wg2_ = out22_
            d_34_wi2_ = out23_
            d_35_wc2_ = out24_
            generated = d_33_wg2_
            insideConstrainedOut = d_34_wi2_
            currentConstrainedOut = d_35_wc2_
            d_1_steps_ = (d_1_steps_) + (d_32_bud2_)
        if (not(insideConstrainedOut)) and (((d_1_steps_) + (2)) <= (maxSteps)):
            d_36_fg_: _dafny.Seq
            d_37_fi_: bool
            d_38_fc_: _dafny.Seq
            out25_: _dafny.Seq
            out26_: bool
            out27_: _dafny.Seq
            out25_, out26_, out27_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_36_fg_ = out25_
            d_37_fi_ = out26_
            d_38_fc_ = out27_
            generated = d_36_fg_
            insideConstrainedOut = d_37_fi_
            currentConstrainedOut = d_38_fc_
            d_1_steps_ = (d_1_steps_) + (1)
            d_39_warmup3_: int
            d_39_warmup3_ = 0
            d_40_warmupMax3_: int
            d_40_warmupMax3_ = 4
            with _dafny.label("7_0"):
                while ((insideConstrainedOut) and ((d_1_steps_) < (maxSteps))) and ((d_39_warmup3_) < (d_40_warmupMax3_)):
                    with _dafny.c_label("7_0"):
                        d_41_cp_: _dafny.Seq
                        d_41_cp_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_42_next_: _dafny.Seq
                        out28_: _dafny.Seq
                        out28_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_41_cp_, currentConstrainedOut, eosToken)
                        d_42_next_ = out28_
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_39_warmup3_ = (d_39_warmup3_) + (1)
                        if (d_42_next_) == (eosToken):
                            raise _dafny.Break("7_0")
                        d_43_ag_: _dafny.Seq
                        d_44_ai_: bool
                        d_45_ac_: _dafny.Seq
                        out29_: _dafny.Seq
                        out30_: bool
                        out31_: _dafny.Seq
                        out29_, out30_, out31_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_42_next_)
                        d_43_ag_ = out29_
                        d_44_ai_ = out30_
                        d_45_ac_ = out31_
                        generated = d_43_ag_
                        insideConstrainedOut = d_44_ai_
                        currentConstrainedOut = d_45_ac_
                        pass
                pass
            if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
                d_46_bud3_: int
                d_46_bud3_ = (maxSteps) - (d_1_steps_)
                d_47_wg3_: _dafny.Seq
                d_48_wi3_: bool
                d_49_wc3_: _dafny.Seq
                out32_: _dafny.Seq
                out33_: bool
                out34_: _dafny.Seq
                out32_, out33_, out34_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_46_bud3_)
                d_47_wg3_ = out32_
                d_48_wi3_ = out33_
                d_49_wc3_ = out34_
                generated = d_47_wg3_
                insideConstrainedOut = d_48_wi3_
                currentConstrainedOut = d_49_wc3_
                d_1_steps_ = (d_1_steps_) + (d_46_bud3_)
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_50_bud4_: int
            d_50_bud4_ = (maxSteps) - (d_1_steps_)
            if (d_50_bud4_) > (0):
                d_51_wg4_: _dafny.Seq
                d_52_wi4_: bool
                d_53_wc4_: _dafny.Seq
                out35_: _dafny.Seq
                out36_: bool
                out37_: _dafny.Seq
                out35_, out36_, out37_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_50_bud4_)
                d_51_wg4_ = out35_
                d_52_wi4_ = out36_
                d_53_wc4_ = out37_
                generated = d_51_wg4_
                insideConstrainedOut = d_52_wi4_
                currentConstrainedOut = d_53_wc4_
                d_1_steps_ = (d_1_steps_) + (d_50_bud4_)
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

