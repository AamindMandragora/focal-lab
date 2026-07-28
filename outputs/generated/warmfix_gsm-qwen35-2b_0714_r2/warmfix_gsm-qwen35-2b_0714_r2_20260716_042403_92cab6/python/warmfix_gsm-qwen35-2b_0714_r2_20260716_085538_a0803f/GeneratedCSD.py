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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math problem step by step using plain English and plain arithmetic. Do NOT use LaTeX, dollar signs, backslashes, fractions notation, or ** for formatting. Write all arithmetic in plain text. At the very end, write: The answer is <<EXPR>> where EXPR is a simple arithmetic expression. RULES: (1) Use EXACTLY the variable names as written in the {curly braces} in the problem statement. For example if the problem has {frac_1} write frac_1, if it has {n1} write n1. (2) For multiplication of an integer by a fraction that should yield an integer, wrap in int(), e.g. int(n * frac). (3) Use only: variable names, numbers, +, -, *, /, (, ), int(). (4) Example good answers: <<count*(n1+n2+n3+n4+n5)>>, <<n*int(bill) - (m*p1 + k*p2)>>, <<total + n2 - n1>>, <<int(n*frac_1*frac_2)>>.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_hasCompletedSpan_: bool
        d_2_hasCompletedSpan_ = False
        d_3_minSpanTok_: int
        d_3_minSpanTok_ = 5
        d_4_chunkSz_: int
        d_4_chunkSz_ = 40
        d_5_innerLim_: int
        d_5_innerLim_ = 100
        d_6_phase1Lim_: int
        d_6_phase1Lim_ = _dafny.euclidian_division((maxSteps) * (70), 100)
        if ((d_6_phase1Lim_) == (0)) and ((maxSteps) > (0)):
            d_6_phase1Lim_ = 1
        d_7_phase4Lim_: int
        d_7_phase4Lim_ = _dafny.euclidian_division((maxSteps) * (85), 100)
        if ((d_7_phase4Lim_) == (0)) and ((maxSteps) > (0)):
            d_7_phase4Lim_ = 1
        if (d_7_phase4Lim_) > (maxSteps):
            d_7_phase4Lim_ = maxSteps
        with _dafny.label("0"):
            while (((d_1_steps_) < (d_6_phase1Lim_)) and (not(insideConstrainedOut))) and (not(d_2_hasCompletedSpan_)):
                with _dafny.c_label("0"):
                    d_8_chunk_: int
                    d_8_chunk_ = d_4_chunkSz_
                    if ((d_1_steps_) + (d_8_chunk_)) > (d_6_phase1Lim_):
                        d_8_chunk_ = (d_6_phase1Lim_) - (d_1_steps_)
                    if (d_8_chunk_) == (0):
                        raise _dafny.Break("0")
                    d_9_cg1_: _dafny.Seq
                    d_10_soo1_: bool
                    d_11_soe1_: bool
                    d_12_su1_: int
                    out0_: _dafny.Seq
                    out1_: bool
                    out2_: bool
                    out3_: int
                    out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_8_chunk_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                    d_9_cg1_ = out0_
                    d_10_soo1_ = out1_
                    d_11_soe1_ = out2_
                    d_12_su1_ = out3_
                    generated = d_9_cg1_
                    d_1_steps_ = (d_1_steps_) + (d_12_su1_)
                    if d_11_soe1_:
                        raise _dafny.Break("0")
                    if d_10_soo1_:
                        d_13_eg1_: _dafny.Seq
                        d_14_ei1_: bool
                        d_15_ec1_: _dafny.Seq
                        out4_: _dafny.Seq
                        out5_: bool
                        out6_: _dafny.Seq
                        out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                        d_13_eg1_ = out4_
                        d_14_ei1_ = out5_
                        d_15_ec1_ = out6_
                        generated = d_13_eg1_
                        insideConstrainedOut = d_14_ei1_
                        currentConstrainedOut = d_15_ec1_
                    pass
            pass
        d_16_innerSteps2_: int
        d_16_innerSteps2_ = 0
        with _dafny.label("1"):
            while (((insideConstrainedOut) and (not(d_2_hasCompletedSpan_))) and ((d_1_steps_) < (maxSteps))) and ((d_16_innerSteps2_) < (d_5_innerLim_)):
                with _dafny.c_label("1"):
                    if ((len(currentConstrainedOut)) >= (d_3_minSpanTok_)) and ((parser).IsCompletePrefix(currentConstrainedOut)):
                        d_17_cg2_: _dafny.Seq
                        d_18_ci2_: bool
                        d_19_cc2_: _dafny.Seq
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: _dafny.Seq
                        out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_17_cg2_ = out7_
                        d_18_ci2_ = out8_
                        d_19_cc2_ = out9_
                        d_1_steps_ = (d_1_steps_) + (1)
                        generated = d_17_cg2_
                        insideConstrainedOut = d_18_ci2_
                        currentConstrainedOut = d_19_cc2_
                        d_2_hasCompletedSpan_ = True
                    elif True:
                        d_20_cp2_: _dafny.Seq
                        d_20_cp2_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_21_nxt2_: _dafny.Seq
                        out10_: _dafny.Seq
                        out10_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_20_cp2_, currentConstrainedOut, eosToken)
                        d_21_nxt2_ = out10_
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_16_innerSteps2_ = (d_16_innerSteps2_) + (1)
                        if (d_21_nxt2_) == (eosToken):
                            raise _dafny.Break("1")
                        d_22_ag2_: _dafny.Seq
                        d_23_ai2_: bool
                        d_24_ac2_: _dafny.Seq
                        out11_: _dafny.Seq
                        out12_: bool
                        out13_: _dafny.Seq
                        out11_, out12_, out13_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_21_nxt2_)
                        d_22_ag2_ = out11_
                        d_23_ai2_ = out12_
                        d_24_ac2_ = out13_
                        generated = d_22_ag2_
                        insideConstrainedOut = d_23_ai2_
                        currentConstrainedOut = d_24_ac2_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_25_bud3_: int
            d_25_bud3_ = (maxSteps) - (d_1_steps_)
            d_26_wg3_: _dafny.Seq
            d_27_wi3_: bool
            d_28_wc3_: _dafny.Seq
            out14_: _dafny.Seq
            out15_: bool
            out16_: _dafny.Seq
            out14_, out15_, out16_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_25_bud3_)
            d_26_wg3_ = out14_
            d_27_wi3_ = out15_
            d_28_wc3_ = out16_
            generated = d_26_wg3_
            insideConstrainedOut = d_27_wi3_
            currentConstrainedOut = d_28_wc3_
            d_1_steps_ = (d_1_steps_) + (d_25_bud3_)
            if not(insideConstrainedOut):
                d_2_hasCompletedSpan_ = True
        with _dafny.label("2"):
            while ((not(insideConstrainedOut)) and (not(d_2_hasCompletedSpan_))) and ((d_1_steps_) < (d_7_phase4Lim_)):
                with _dafny.c_label("2"):
                    d_29_nxt4_: _dafny.Seq
                    out17_: _dafny.Seq
                    out17_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                    d_29_nxt4_ = out17_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if (d_29_nxt4_) == (eosToken):
                        raise _dafny.Break("2")
                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_29_nxt4_]))
                    if VerifiedDecoderAgent.default__.RenderedEndsWith(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                        d_30_eg4_: _dafny.Seq
                        d_31_ei4_: bool
                        d_32_ec4_: _dafny.Seq
                        out18_: _dafny.Seq
                        out19_: bool
                        out20_: _dafny.Seq
                        out18_, out19_, out20_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                        d_30_eg4_ = out18_
                        d_31_ei4_ = out19_
                        d_32_ec4_ = out20_
                        generated = d_30_eg4_
                        insideConstrainedOut = d_31_ei4_
                        currentConstrainedOut = d_32_ec4_
                    pass
            pass
        d_33_innerSteps5_: int
        d_33_innerSteps5_ = 0
        with _dafny.label("3"):
            while (((insideConstrainedOut) and (not(d_2_hasCompletedSpan_))) and ((d_1_steps_) < (maxSteps))) and ((d_33_innerSteps5_) < (d_5_innerLim_)):
                with _dafny.c_label("3"):
                    if ((len(currentConstrainedOut)) >= (d_3_minSpanTok_)) and ((parser).IsCompletePrefix(currentConstrainedOut)):
                        d_34_cg5_: _dafny.Seq
                        d_35_ci5_: bool
                        d_36_cc5_: _dafny.Seq
                        out21_: _dafny.Seq
                        out22_: bool
                        out23_: _dafny.Seq
                        out21_, out22_, out23_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_34_cg5_ = out21_
                        d_35_ci5_ = out22_
                        d_36_cc5_ = out23_
                        d_1_steps_ = (d_1_steps_) + (1)
                        generated = d_34_cg5_
                        insideConstrainedOut = d_35_ci5_
                        currentConstrainedOut = d_36_cc5_
                        d_2_hasCompletedSpan_ = True
                    elif True:
                        d_37_cp5_: _dafny.Seq
                        d_37_cp5_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_38_nxt5_: _dafny.Seq
                        out24_: _dafny.Seq
                        out24_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_37_cp5_, currentConstrainedOut, eosToken)
                        d_38_nxt5_ = out24_
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_33_innerSteps5_ = (d_33_innerSteps5_) + (1)
                        if (d_38_nxt5_) == (eosToken):
                            raise _dafny.Break("3")
                        d_39_ag5_: _dafny.Seq
                        d_40_ai5_: bool
                        d_41_ac5_: _dafny.Seq
                        out25_: _dafny.Seq
                        out26_: bool
                        out27_: _dafny.Seq
                        out25_, out26_, out27_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_38_nxt5_)
                        d_39_ag5_ = out25_
                        d_40_ai5_ = out26_
                        d_41_ac5_ = out27_
                        generated = d_39_ag5_
                        insideConstrainedOut = d_40_ai5_
                        currentConstrainedOut = d_41_ac5_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_42_bud6_: int
            d_42_bud6_ = (maxSteps) - (d_1_steps_)
            d_43_wg6_: _dafny.Seq
            d_44_wi6_: bool
            d_45_wc6_: _dafny.Seq
            out28_: _dafny.Seq
            out29_: bool
            out30_: _dafny.Seq
            out28_, out29_, out30_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_42_bud6_)
            d_43_wg6_ = out28_
            d_44_wi6_ = out29_
            d_45_wc6_ = out30_
            generated = d_43_wg6_
            insideConstrainedOut = d_44_wi6_
            currentConstrainedOut = d_45_wc6_
            d_1_steps_ = (d_1_steps_) + (d_42_bud6_)
            if not(insideConstrainedOut):
                d_2_hasCompletedSpan_ = True
        if ((not(insideConstrainedOut)) and (not(d_2_hasCompletedSpan_))) and (((d_1_steps_) + (2)) <= (maxSteps)):
            d_46_fg7_: _dafny.Seq
            d_47_fi7_: bool
            d_48_fc7_: _dafny.Seq
            out31_: _dafny.Seq
            out32_: bool
            out33_: _dafny.Seq
            out31_, out32_, out33_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_46_fg7_ = out31_
            d_47_fi7_ = out32_
            d_48_fc7_ = out33_
            generated = d_46_fg7_
            insideConstrainedOut = d_47_fi7_
            currentConstrainedOut = d_48_fc7_
            d_1_steps_ = (d_1_steps_) + (1)
            d_49_innerSteps7_: int
            d_49_innerSteps7_ = 0
            d_50_innerLim7_: int
            d_50_innerLim7_ = 60
            with _dafny.label("9_0"):
                while (((insideConstrainedOut) and (not(d_2_hasCompletedSpan_))) and ((d_1_steps_) < (maxSteps))) and ((d_49_innerSteps7_) < (d_50_innerLim7_)):
                    with _dafny.c_label("9_0"):
                        if ((len(currentConstrainedOut)) >= (d_3_minSpanTok_)) and ((parser).IsCompletePrefix(currentConstrainedOut)):
                            d_51_cg7_: _dafny.Seq
                            d_52_ci7_: bool
                            d_53_cc7_: _dafny.Seq
                            out34_: _dafny.Seq
                            out35_: bool
                            out36_: _dafny.Seq
                            out34_, out35_, out36_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_51_cg7_ = out34_
                            d_52_ci7_ = out35_
                            d_53_cc7_ = out36_
                            d_1_steps_ = (d_1_steps_) + (1)
                            generated = d_51_cg7_
                            insideConstrainedOut = d_52_ci7_
                            currentConstrainedOut = d_53_cc7_
                            d_2_hasCompletedSpan_ = True
                        elif True:
                            d_54_cp7_: _dafny.Seq
                            d_54_cp7_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_55_nxt7_: _dafny.Seq
                            out37_: _dafny.Seq
                            out37_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_54_cp7_, currentConstrainedOut, eosToken)
                            d_55_nxt7_ = out37_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_49_innerSteps7_ = (d_49_innerSteps7_) + (1)
                            if (d_55_nxt7_) == (eosToken):
                                raise _dafny.Break("9_0")
                            d_56_ag7_: _dafny.Seq
                            d_57_ai7_: bool
                            d_58_ac7_: _dafny.Seq
                            out38_: _dafny.Seq
                            out39_: bool
                            out40_: _dafny.Seq
                            out38_, out39_, out40_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_55_nxt7_)
                            d_56_ag7_ = out38_
                            d_57_ai7_ = out39_
                            d_58_ac7_ = out40_
                            generated = d_56_ag7_
                            insideConstrainedOut = d_57_ai7_
                            currentConstrainedOut = d_58_ac7_
                        pass
                pass
            if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
                d_59_bud7_: int
                d_59_bud7_ = (maxSteps) - (d_1_steps_)
                d_60_wg7_: _dafny.Seq
                d_61_wi7_: bool
                d_62_wc7_: _dafny.Seq
                out41_: _dafny.Seq
                out42_: bool
                out43_: _dafny.Seq
                out41_, out42_, out43_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_59_bud7_)
                d_60_wg7_ = out41_
                d_61_wi7_ = out42_
                d_62_wc7_ = out43_
                generated = d_60_wg7_
                insideConstrainedOut = d_61_wi7_
                currentConstrainedOut = d_62_wc7_
                d_1_steps_ = (d_1_steps_) + (d_59_bud7_)
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_63_bud8_: int
            d_63_bud8_ = (maxSteps) - (d_1_steps_)
            if (d_63_bud8_) > (0):
                d_64_wg8_: _dafny.Seq
                d_65_wi8_: bool
                d_66_wc8_: _dafny.Seq
                out44_: _dafny.Seq
                out45_: bool
                out46_: _dafny.Seq
                out44_, out45_, out46_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_63_bud8_)
                d_64_wg8_ = out44_
                d_65_wi8_ = out45_
                d_66_wc8_ = out46_
                generated = d_64_wg8_
                insideConstrainedOut = d_65_wi8_
                currentConstrainedOut = d_66_wc8_
                d_1_steps_ = (d_1_steps_) + (d_63_bud8_)
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

