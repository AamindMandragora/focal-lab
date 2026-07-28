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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Reason step by step. At the end write: The answer is EXPR where EXPR uses only variable names from the problem without curly braces, numbers, +, -, *, /, (, ), int(). Use int() for truncating fractions, for example int(n * frac). No LaTeX, no dollar signs, no backslashes, no curly braces in EXPR.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_hasCompletedSpan_: bool
        d_2_hasCompletedSpan_ = False
        d_3_phase1Limit_: int
        d_3_phase1Limit_ = _dafny.euclidian_division((maxSteps) * (65), 100)
        if ((d_3_phase1Limit_) == (0)) and ((maxSteps) > (0)):
            d_3_phase1Limit_ = 1
        d_4_phase3Limit_: int
        d_4_phase3Limit_ = (d_3_phase1Limit_) + (80)
        if (d_4_phase3Limit_) > (maxSteps):
            d_4_phase3Limit_ = maxSteps
        with _dafny.label("0"):
            while (((d_1_steps_) < (d_3_phase1Limit_)) and (not(insideConstrainedOut))) and (not(d_2_hasCompletedSpan_)):
                with _dafny.c_label("0"):
                    d_5_chunkSz_: int
                    d_5_chunkSz_ = 40
                    if ((d_1_steps_) + (d_5_chunkSz_)) > (d_3_phase1Limit_):
                        d_5_chunkSz_ = (d_3_phase1Limit_) - (d_1_steps_)
                    if (d_5_chunkSz_) == (0):
                        raise _dafny.Break("0")
                    d_6_cg_: _dafny.Seq
                    d_7_soo_: bool
                    d_8_soe_: bool
                    d_9_su_: int
                    out0_: _dafny.Seq
                    out1_: bool
                    out2_: bool
                    out3_: int
                    out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_5_chunkSz_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                    d_6_cg_ = out0_
                    d_7_soo_ = out1_
                    d_8_soe_ = out2_
                    d_9_su_ = out3_
                    generated = d_6_cg_
                    d_1_steps_ = (d_1_steps_) + (d_9_su_)
                    if d_8_soe_:
                        raise _dafny.Break("0")
                    if d_7_soo_:
                        d_10_eg_: _dafny.Seq
                        d_11_ei_: bool
                        d_12_ec_: _dafny.Seq
                        out4_: _dafny.Seq
                        out5_: bool
                        out6_: _dafny.Seq
                        out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                        d_10_eg_ = out4_
                        d_11_ei_ = out5_
                        d_12_ec_ = out6_
                        generated = d_10_eg_
                        insideConstrainedOut = d_11_ei_
                        currentConstrainedOut = d_12_ec_
                    pass
            pass
        d_13_warmup1_: int
        d_13_warmup1_ = 0
        with _dafny.label("1"):
            while (((insideConstrainedOut) and (not(d_2_hasCompletedSpan_))) and ((d_1_steps_) < (maxSteps))) and ((d_13_warmup1_) < (3)):
                with _dafny.c_label("1"):
                    d_14_cp1_: _dafny.Seq
                    d_14_cp1_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                    d_15_next1_: _dafny.Seq
                    out7_: _dafny.Seq
                    out7_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_14_cp1_, currentConstrainedOut, eosToken)
                    d_15_next1_ = out7_
                    d_1_steps_ = (d_1_steps_) + (1)
                    d_13_warmup1_ = (d_13_warmup1_) + (1)
                    if (d_15_next1_) == (eosToken):
                        raise _dafny.Break("1")
                    d_16_ag1_: _dafny.Seq
                    d_17_ai1_: bool
                    d_18_ac1_: _dafny.Seq
                    out8_: _dafny.Seq
                    out9_: bool
                    out10_: _dafny.Seq
                    out8_, out9_, out10_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_15_next1_)
                    d_16_ag1_ = out8_
                    d_17_ai1_ = out9_
                    d_18_ac1_ = out10_
                    generated = d_16_ag1_
                    insideConstrainedOut = d_17_ai1_
                    currentConstrainedOut = d_18_ac1_
                    pass
            pass
        if ((insideConstrainedOut) and (not(d_2_hasCompletedSpan_))) and ((d_1_steps_) < (maxSteps)):
            d_19_bud1_: int
            d_19_bud1_ = 80
            if ((d_1_steps_) + (d_19_bud1_)) > (maxSteps):
                d_19_bud1_ = (maxSteps) - (d_1_steps_)
            if (d_19_bud1_) > (0):
                d_20_wg1_: _dafny.Seq
                d_21_wi1_: bool
                d_22_wc1_: _dafny.Seq
                out11_: _dafny.Seq
                out12_: bool
                out13_: _dafny.Seq
                out11_, out12_, out13_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_19_bud1_)
                d_20_wg1_ = out11_
                d_21_wi1_ = out12_
                d_22_wc1_ = out13_
                generated = d_20_wg1_
                insideConstrainedOut = d_21_wi1_
                currentConstrainedOut = d_22_wc1_
                d_1_steps_ = (d_1_steps_) + (d_19_bud1_)
                if not(insideConstrainedOut):
                    d_2_hasCompletedSpan_ = True
        with _dafny.label("2"):
            while (((d_1_steps_) < (d_4_phase3Limit_)) and (not(insideConstrainedOut))) and (not(d_2_hasCompletedSpan_)):
                with _dafny.c_label("2"):
                    d_23_next3_: _dafny.Seq
                    out14_: _dafny.Seq
                    out14_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                    d_23_next3_ = out14_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if (d_23_next3_) == (eosToken):
                        raise _dafny.Break("2")
                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_23_next3_]))
                    if VerifiedDecoderAgent.default__.RenderedEndsWith(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                        d_24_eg3_: _dafny.Seq
                        d_25_ei3_: bool
                        d_26_ec3_: _dafny.Seq
                        out15_: _dafny.Seq
                        out16_: bool
                        out17_: _dafny.Seq
                        out15_, out16_, out17_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                        d_24_eg3_ = out15_
                        d_25_ei3_ = out16_
                        d_26_ec3_ = out17_
                        generated = d_24_eg3_
                        insideConstrainedOut = d_25_ei3_
                        currentConstrainedOut = d_26_ec3_
                    pass
            pass
        d_27_warmup2_: int
        d_27_warmup2_ = 0
        with _dafny.label("3"):
            while (((insideConstrainedOut) and (not(d_2_hasCompletedSpan_))) and ((d_1_steps_) < (maxSteps))) and ((d_27_warmup2_) < (3)):
                with _dafny.c_label("3"):
                    d_28_cp2_: _dafny.Seq
                    d_28_cp2_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                    d_29_next2_: _dafny.Seq
                    out18_: _dafny.Seq
                    out18_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_28_cp2_, currentConstrainedOut, eosToken)
                    d_29_next2_ = out18_
                    d_1_steps_ = (d_1_steps_) + (1)
                    d_27_warmup2_ = (d_27_warmup2_) + (1)
                    if (d_29_next2_) == (eosToken):
                        raise _dafny.Break("3")
                    d_30_ag2_: _dafny.Seq
                    d_31_ai2_: bool
                    d_32_ac2_: _dafny.Seq
                    out19_: _dafny.Seq
                    out20_: bool
                    out21_: _dafny.Seq
                    out19_, out20_, out21_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_29_next2_)
                    d_30_ag2_ = out19_
                    d_31_ai2_ = out20_
                    d_32_ac2_ = out21_
                    generated = d_30_ag2_
                    insideConstrainedOut = d_31_ai2_
                    currentConstrainedOut = d_32_ac2_
                    pass
            pass
        if ((insideConstrainedOut) and (not(d_2_hasCompletedSpan_))) and ((d_1_steps_) < (maxSteps)):
            d_33_bud2_: int
            d_33_bud2_ = 80
            if ((d_1_steps_) + (d_33_bud2_)) > (maxSteps):
                d_33_bud2_ = (maxSteps) - (d_1_steps_)
            if (d_33_bud2_) > (0):
                d_34_wg2_: _dafny.Seq
                d_35_wi2_: bool
                d_36_wc2_: _dafny.Seq
                out22_: _dafny.Seq
                out23_: bool
                out24_: _dafny.Seq
                out22_, out23_, out24_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_33_bud2_)
                d_34_wg2_ = out22_
                d_35_wi2_ = out23_
                d_36_wc2_ = out24_
                generated = d_34_wg2_
                insideConstrainedOut = d_35_wi2_
                currentConstrainedOut = d_36_wc2_
                d_1_steps_ = (d_1_steps_) + (d_33_bud2_)
                if not(insideConstrainedOut):
                    d_2_hasCompletedSpan_ = True
        if ((not(insideConstrainedOut)) and (not(d_2_hasCompletedSpan_))) and (((d_1_steps_) + (1)) < (maxSteps)):
            d_37_fg_: _dafny.Seq
            d_38_fi_: bool
            d_39_fc_: _dafny.Seq
            out25_: _dafny.Seq
            out26_: bool
            out27_: _dafny.Seq
            out25_, out26_, out27_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_37_fg_ = out25_
            d_38_fi_ = out26_
            d_39_fc_ = out27_
            generated = d_37_fg_
            insideConstrainedOut = d_38_fi_
            currentConstrainedOut = d_39_fc_
            d_1_steps_ = (d_1_steps_) + (1)
            d_40_warmup3_: int
            d_40_warmup3_ = 0
            with _dafny.label("8_0"):
                while ((insideConstrainedOut) and ((d_1_steps_) < (maxSteps))) and ((d_40_warmup3_) < (3)):
                    with _dafny.c_label("8_0"):
                        d_41_cp5_: _dafny.Seq
                        d_41_cp5_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_42_next5_: _dafny.Seq
                        out28_: _dafny.Seq
                        out28_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_41_cp5_, currentConstrainedOut, eosToken)
                        d_42_next5_ = out28_
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_40_warmup3_ = (d_40_warmup3_) + (1)
                        if (d_42_next5_) == (eosToken):
                            raise _dafny.Break("8_0")
                        d_43_ag5_: _dafny.Seq
                        d_44_ai5_: bool
                        d_45_ac5_: _dafny.Seq
                        out29_: _dafny.Seq
                        out30_: bool
                        out31_: _dafny.Seq
                        out29_, out30_, out31_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_42_next5_)
                        d_43_ag5_ = out29_
                        d_44_ai5_ = out30_
                        d_45_ac5_ = out31_
                        generated = d_43_ag5_
                        insideConstrainedOut = d_44_ai5_
                        currentConstrainedOut = d_45_ac5_
                        pass
                pass
            if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
                d_46_bud5_: int
                d_46_bud5_ = (maxSteps) - (d_1_steps_)
                if (d_46_bud5_) > (0):
                    d_47_wg5_: _dafny.Seq
                    d_48_wi5_: bool
                    d_49_wc5_: _dafny.Seq
                    out32_: _dafny.Seq
                    out33_: bool
                    out34_: _dafny.Seq
                    out32_, out33_, out34_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_46_bud5_)
                    d_47_wg5_ = out32_
                    d_48_wi5_ = out33_
                    d_49_wc5_ = out34_
                    generated = d_47_wg5_
                    insideConstrainedOut = d_48_wi5_
                    currentConstrainedOut = d_49_wc5_
                    d_1_steps_ = (d_1_steps_) + (d_46_bud5_)
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_50_bud6_: int
            d_50_bud6_ = (maxSteps) - (d_1_steps_)
            if (d_50_bud6_) > (0):
                d_51_wg6_: _dafny.Seq
                d_52_wi6_: bool
                d_53_wc6_: _dafny.Seq
                out35_: _dafny.Seq
                out36_: bool
                out37_: _dafny.Seq
                out35_, out36_, out37_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_50_bud6_)
                d_51_wg6_ = out35_
                d_52_wi6_ = out36_
                d_53_wc6_ = out37_
                generated = d_51_wg6_
                insideConstrainedOut = d_52_wi6_
                currentConstrainedOut = d_53_wc6_
                d_1_steps_ = (d_1_steps_) + (d_50_bud6_)
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

