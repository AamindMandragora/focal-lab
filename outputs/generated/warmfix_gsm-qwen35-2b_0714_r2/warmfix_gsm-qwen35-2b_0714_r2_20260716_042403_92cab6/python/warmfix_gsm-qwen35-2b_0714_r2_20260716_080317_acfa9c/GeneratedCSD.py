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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step. At the end write the COMPLETE combined arithmetic formula: <<EXPR>> where EXPR uses the problem variable names and operators (+, -, *, /, parentheses). The formula must compute the full final answer in one expression. Do NOT write a single variable or a partial sub-step. Examples of good formulas: <<n1*c1 + n2*c2 + c3>>, <<total - (total*frac + daily*period*7)>>, <<(n*m)//k>>, <<int(n1*frac1 + n2*mult1)>>. Include every variable and every operation needed.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_hasCompletedSpan_: bool
        d_2_hasCompletedSpan_ = False
        d_3_minSpanTokens_: int
        d_3_minSpanTokens_ = 5
        d_4_innerStepLimit_: int
        d_4_innerStepLimit_ = 80
        d_5_chunkSize_: int
        d_5_chunkSize_ = 40
        d_6_freeLimit_: int
        d_6_freeLimit_ = _dafny.euclidian_division((maxSteps) * (7), 10)
        if ((d_6_freeLimit_) == (0)) and ((maxSteps) > (0)):
            d_6_freeLimit_ = 1
        if (d_6_freeLimit_) > (maxSteps):
            d_6_freeLimit_ = maxSteps
        d_7_closeReserve_: int
        d_7_closeReserve_ = 80
        if (d_7_closeReserve_) > (maxSteps):
            d_7_closeReserve_ = maxSteps
        d_8_constLimit_: int
        d_8_constLimit_ = maxSteps
        if (maxSteps) >= (d_7_closeReserve_):
            d_8_constLimit_ = (maxSteps) - (d_7_closeReserve_)
        if (d_8_constLimit_) < (d_6_freeLimit_):
            d_8_constLimit_ = d_6_freeLimit_
        if (d_8_constLimit_) > (maxSteps):
            d_8_constLimit_ = maxSteps
        with _dafny.label("0"):
            while ((d_1_steps_) < (d_6_freeLimit_)) and (not(insideConstrainedOut)):
                with _dafny.c_label("0"):
                    d_9_actualChunk_: int
                    d_9_actualChunk_ = d_5_chunkSize_
                    if ((d_1_steps_) + (d_9_actualChunk_)) > (d_6_freeLimit_):
                        d_9_actualChunk_ = (d_6_freeLimit_) - (d_1_steps_)
                    if (d_9_actualChunk_) == (0):
                        raise _dafny.Break("0")
                    d_10_cg1_: _dafny.Seq
                    d_11_stoppedOnOpen_: bool
                    d_12_stoppedOnEos_: bool
                    d_13_stepsUsed1_: int
                    out0_: _dafny.Seq
                    out1_: bool
                    out2_: bool
                    out3_: int
                    out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_9_actualChunk_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                    d_10_cg1_ = out0_
                    d_11_stoppedOnOpen_ = out1_
                    d_12_stoppedOnEos_ = out2_
                    d_13_stepsUsed1_ = out3_
                    generated = d_10_cg1_
                    d_1_steps_ = (d_1_steps_) + (d_13_stepsUsed1_)
                    if d_12_stoppedOnEos_:
                        raise _dafny.Break("0")
                    if d_11_stoppedOnOpen_:
                        d_14_eg1_: _dafny.Seq
                        d_15_ei1_: bool
                        d_16_ec1_: _dafny.Seq
                        out4_: _dafny.Seq
                        out5_: bool
                        out6_: _dafny.Seq
                        out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                        d_14_eg1_ = out4_
                        d_15_ei1_ = out5_
                        d_16_ec1_ = out6_
                        generated = d_14_eg1_
                        insideConstrainedOut = d_15_ei1_
                        currentConstrainedOut = d_16_ec1_
                    pass
            pass
        d_17_innerSteps_: int
        d_17_innerSteps_ = 0
        with _dafny.label("1"):
            while ((insideConstrainedOut) and ((d_1_steps_) < (d_8_constLimit_))) and ((d_17_innerSteps_) < (d_4_innerStepLimit_)):
                with _dafny.c_label("1"):
                    if ((len(currentConstrainedOut)) >= (d_3_minSpanTokens_)) and ((parser).IsCompletePrefix(currentConstrainedOut)):
                        d_18_cg2_: _dafny.Seq
                        d_19_ci2_: bool
                        d_20_cc2_: _dafny.Seq
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: _dafny.Seq
                        out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_18_cg2_ = out7_
                        d_19_ci2_ = out8_
                        d_20_cc2_ = out9_
                        d_1_steps_ = (d_1_steps_) + (1)
                        generated = d_18_cg2_
                        insideConstrainedOut = d_19_ci2_
                        currentConstrainedOut = d_20_cc2_
                        d_2_hasCompletedSpan_ = True
                    elif True:
                        d_21_cp2_: _dafny.Seq
                        d_21_cp2_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_22_next2_: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                        if (len(currentConstrainedOut)) < (d_3_minSpanTokens_):
                            out10_: _dafny.Seq
                            out10_ = (d_0_helpers_).SafePenalizedConstrainedStep(lm, parser, d_21_cp2_, currentConstrainedOut, _dafny.SeqWithoutIsStrInference([eosToken]), _dafny.BigRational('8e0'), eosToken)
                            d_22_next2_ = out10_
                        elif True:
                            out11_: _dafny.Seq
                            out11_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_21_cp2_, currentConstrainedOut, eosToken)
                            d_22_next2_ = out11_
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_17_innerSteps_ = (d_17_innerSteps_) + (1)
                        if (d_22_next2_) == (eosToken):
                            raise _dafny.Break("1")
                        d_23_ag2_: _dafny.Seq
                        d_24_ai2_: bool
                        d_25_ac2_: _dafny.Seq
                        out12_: _dafny.Seq
                        out13_: bool
                        out14_: _dafny.Seq
                        out12_, out13_, out14_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_22_next2_)
                        d_23_ag2_ = out12_
                        d_24_ai2_ = out13_
                        d_25_ac2_ = out14_
                        generated = d_23_ag2_
                        insideConstrainedOut = d_24_ai2_
                        currentConstrainedOut = d_25_ac2_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_26_rem3_: int
            d_26_rem3_ = (maxSteps) - (d_1_steps_)
            d_27_closeB3_: int
            d_27_closeB3_ = 60
            if (d_27_closeB3_) > (d_26_rem3_):
                d_27_closeB3_ = d_26_rem3_
            if (d_27_closeB3_) > (0):
                d_28_wg3_: _dafny.Seq
                d_29_wi3_: bool
                d_30_wc3_: _dafny.Seq
                out15_: _dafny.Seq
                out16_: bool
                out17_: _dafny.Seq
                out15_, out16_, out17_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_27_closeB3_)
                d_28_wg3_ = out15_
                d_29_wi3_ = out16_
                d_30_wc3_ = out17_
                generated = d_28_wg3_
                insideConstrainedOut = d_29_wi3_
                currentConstrainedOut = d_30_wc3_
                d_1_steps_ = (d_1_steps_) + (d_27_closeB3_)
                if not(insideConstrainedOut):
                    d_2_hasCompletedSpan_ = True
        with _dafny.label("2"):
            while ((not(insideConstrainedOut)) and (not(d_2_hasCompletedSpan_))) and ((d_1_steps_) < (d_6_freeLimit_)):
                with _dafny.c_label("2"):
                    d_31_next4_: _dafny.Seq
                    out18_: _dafny.Seq
                    out18_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                    d_31_next4_ = out18_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if (d_31_next4_) == (eosToken):
                        raise _dafny.Break("2")
                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_31_next4_]))
                    if VerifiedDecoderAgent.default__.RenderedEndsWith(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                        d_32_eg4_: _dafny.Seq
                        d_33_ei4_: bool
                        d_34_ec4_: _dafny.Seq
                        out19_: _dafny.Seq
                        out20_: bool
                        out21_: _dafny.Seq
                        out19_, out20_, out21_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                        d_32_eg4_ = out19_
                        d_33_ei4_ = out20_
                        d_34_ec4_ = out21_
                        generated = d_32_eg4_
                        insideConstrainedOut = d_33_ei4_
                        currentConstrainedOut = d_34_ec4_
                    pass
            pass
        d_35_innerSteps5_: int
        d_35_innerSteps5_ = 0
        with _dafny.label("3"):
            while (((insideConstrainedOut) and (not(d_2_hasCompletedSpan_))) and ((d_1_steps_) < (d_8_constLimit_))) and ((d_35_innerSteps5_) < (d_4_innerStepLimit_)):
                with _dafny.c_label("3"):
                    if ((len(currentConstrainedOut)) >= (d_3_minSpanTokens_)) and ((parser).IsCompletePrefix(currentConstrainedOut)):
                        d_36_cg5_: _dafny.Seq
                        d_37_ci5_: bool
                        d_38_cc5_: _dafny.Seq
                        out22_: _dafny.Seq
                        out23_: bool
                        out24_: _dafny.Seq
                        out22_, out23_, out24_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_36_cg5_ = out22_
                        d_37_ci5_ = out23_
                        d_38_cc5_ = out24_
                        d_1_steps_ = (d_1_steps_) + (1)
                        generated = d_36_cg5_
                        insideConstrainedOut = d_37_ci5_
                        currentConstrainedOut = d_38_cc5_
                        d_2_hasCompletedSpan_ = True
                    elif True:
                        d_39_cp5_: _dafny.Seq
                        d_39_cp5_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_40_next5_: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                        if (len(currentConstrainedOut)) < (d_3_minSpanTokens_):
                            out25_: _dafny.Seq
                            out25_ = (d_0_helpers_).SafePenalizedConstrainedStep(lm, parser, d_39_cp5_, currentConstrainedOut, _dafny.SeqWithoutIsStrInference([eosToken]), _dafny.BigRational('8e0'), eosToken)
                            d_40_next5_ = out25_
                        elif True:
                            out26_: _dafny.Seq
                            out26_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_39_cp5_, currentConstrainedOut, eosToken)
                            d_40_next5_ = out26_
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_35_innerSteps5_ = (d_35_innerSteps5_) + (1)
                        if (d_40_next5_) == (eosToken):
                            raise _dafny.Break("3")
                        d_41_ag5_: _dafny.Seq
                        d_42_ai5_: bool
                        d_43_ac5_: _dafny.Seq
                        out27_: _dafny.Seq
                        out28_: bool
                        out29_: _dafny.Seq
                        out27_, out28_, out29_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_40_next5_)
                        d_41_ag5_ = out27_
                        d_42_ai5_ = out28_
                        d_43_ac5_ = out29_
                        generated = d_41_ag5_
                        insideConstrainedOut = d_42_ai5_
                        currentConstrainedOut = d_43_ac5_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_44_rem6_: int
            d_44_rem6_ = (maxSteps) - (d_1_steps_)
            d_45_closeB6_: int
            d_45_closeB6_ = 60
            if (d_45_closeB6_) > (d_44_rem6_):
                d_45_closeB6_ = d_44_rem6_
            if (d_45_closeB6_) > (0):
                d_46_wg6_: _dafny.Seq
                d_47_wi6_: bool
                d_48_wc6_: _dafny.Seq
                out30_: _dafny.Seq
                out31_: bool
                out32_: _dafny.Seq
                out30_, out31_, out32_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_45_closeB6_)
                d_46_wg6_ = out30_
                d_47_wi6_ = out31_
                d_48_wc6_ = out32_
                generated = d_46_wg6_
                insideConstrainedOut = d_47_wi6_
                currentConstrainedOut = d_48_wc6_
                d_1_steps_ = (d_1_steps_) + (d_45_closeB6_)
                if not(insideConstrainedOut):
                    d_2_hasCompletedSpan_ = True
        if ((not(insideConstrainedOut)) and (not(d_2_hasCompletedSpan_))) and (((d_1_steps_) + (2)) <= (maxSteps)):
            d_49_fg7_: _dafny.Seq
            d_50_fi7_: bool
            d_51_fc7_: _dafny.Seq
            out33_: _dafny.Seq
            out34_: bool
            out35_: _dafny.Seq
            out33_, out34_, out35_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_49_fg7_ = out33_
            d_50_fi7_ = out34_
            d_51_fc7_ = out35_
            generated = d_49_fg7_
            insideConstrainedOut = d_50_fi7_
            currentConstrainedOut = d_51_fc7_
            d_1_steps_ = (d_1_steps_) + (1)
            d_52_innerSteps7_: int
            d_52_innerSteps7_ = 0
            d_53_innerLimit7_: int
            d_53_innerLimit7_ = 40
            with _dafny.label("12_0"):
                while (((insideConstrainedOut) and (not(d_2_hasCompletedSpan_))) and ((d_1_steps_) < (maxSteps))) and ((d_52_innerSteps7_) < (d_53_innerLimit7_)):
                    with _dafny.c_label("12_0"):
                        if ((len(currentConstrainedOut)) >= (d_3_minSpanTokens_)) and ((parser).IsCompletePrefix(currentConstrainedOut)):
                            d_54_cg7_: _dafny.Seq
                            d_55_ci7_: bool
                            d_56_cc7_: _dafny.Seq
                            out36_: _dafny.Seq
                            out37_: bool
                            out38_: _dafny.Seq
                            out36_, out37_, out38_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_54_cg7_ = out36_
                            d_55_ci7_ = out37_
                            d_56_cc7_ = out38_
                            d_1_steps_ = (d_1_steps_) + (1)
                            generated = d_54_cg7_
                            insideConstrainedOut = d_55_ci7_
                            currentConstrainedOut = d_56_cc7_
                            d_2_hasCompletedSpan_ = True
                        elif True:
                            d_57_cp7_: _dafny.Seq
                            d_57_cp7_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_58_next7_: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                            if (len(currentConstrainedOut)) < (d_3_minSpanTokens_):
                                out39_: _dafny.Seq
                                out39_ = (d_0_helpers_).SafePenalizedConstrainedStep(lm, parser, d_57_cp7_, currentConstrainedOut, _dafny.SeqWithoutIsStrInference([eosToken]), _dafny.BigRational('8e0'), eosToken)
                                d_58_next7_ = out39_
                            elif True:
                                out40_: _dafny.Seq
                                out40_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_57_cp7_, currentConstrainedOut, eosToken)
                                d_58_next7_ = out40_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_52_innerSteps7_ = (d_52_innerSteps7_) + (1)
                            if (d_58_next7_) == (eosToken):
                                raise _dafny.Break("12_0")
                            d_59_ag7_: _dafny.Seq
                            d_60_ai7_: bool
                            d_61_ac7_: _dafny.Seq
                            out41_: _dafny.Seq
                            out42_: bool
                            out43_: _dafny.Seq
                            out41_, out42_, out43_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_58_next7_)
                            d_59_ag7_ = out41_
                            d_60_ai7_ = out42_
                            d_61_ac7_ = out43_
                            generated = d_59_ag7_
                            insideConstrainedOut = d_60_ai7_
                            currentConstrainedOut = d_61_ac7_
                        pass
                pass
            if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
                d_62_rem7_: int
                d_62_rem7_ = (maxSteps) - (d_1_steps_)
                d_63_closeB7_: int
                d_63_closeB7_ = 40
                if (d_63_closeB7_) > (d_62_rem7_):
                    d_63_closeB7_ = d_62_rem7_
                if (d_63_closeB7_) > (0):
                    d_64_wg7_: _dafny.Seq
                    d_65_wi7_: bool
                    d_66_wc7_: _dafny.Seq
                    out44_: _dafny.Seq
                    out45_: bool
                    out46_: _dafny.Seq
                    out44_, out45_, out46_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_63_closeB7_)
                    d_64_wg7_ = out44_
                    d_65_wi7_ = out45_
                    d_66_wc7_ = out46_
                    generated = d_64_wg7_
                    insideConstrainedOut = d_65_wi7_
                    currentConstrainedOut = d_66_wc7_
                    d_1_steps_ = (d_1_steps_) + (d_63_closeB7_)
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_67_finalBudget_: int
            d_67_finalBudget_ = (maxSteps) - (d_1_steps_)
            if (d_67_finalBudget_) > (0):
                d_68_wg8_: _dafny.Seq
                d_69_wi8_: bool
                d_70_wc8_: _dafny.Seq
                out47_: _dafny.Seq
                out48_: bool
                out49_: _dafny.Seq
                out47_, out48_, out49_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_67_finalBudget_)
                d_68_wg8_ = out47_
                d_69_wi8_ = out48_
                d_70_wc8_ = out49_
                generated = d_68_wg8_
                insideConstrainedOut = d_69_wi8_
                currentConstrainedOut = d_70_wc8_
                d_1_steps_ = (d_1_steps_) + (d_67_finalBudget_)
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

