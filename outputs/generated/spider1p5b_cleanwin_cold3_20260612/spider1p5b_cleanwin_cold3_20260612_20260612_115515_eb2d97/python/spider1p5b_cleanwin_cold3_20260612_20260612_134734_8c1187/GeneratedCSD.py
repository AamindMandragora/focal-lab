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
        d_0_helpers_: VerifiedDecoderAgent.CSDHelpers
        nw0_ = VerifiedDecoderAgent.CSDHelpers()
        nw0_.ctor__()
        d_0_helpers_ = nw0_
        generated = generatedPrefix
        insideConstrainedOut = insideConstrained
        currentConstrainedOut = currentConstrained
        cost = 0
        d_1_guidance_: _dafny.Seq
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Output exactly: SQL: <<YOUR SQL QUERY HERE>> using the provided schema. The query must be a valid SELECT statement. No semicolons inside the span. No extra text outside the span."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_semicolonTokens_: _dafny.Seq
        d_3_semicolonTokens_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ";")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "`;`")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "`;"))])
        d_4_chunkBudget_: int
        d_4_chunkBudget_ = 30
        if (((d_2_steps_) + (d_4_chunkBudget_)) <= (maxSteps)) and (not(insideConstrainedOut)):
            d_5_chunkGenerated_: _dafny.Seq
            d_6_stoppedOnOpenSpan_: bool
            d_7_stoppedOnEos_: bool
            d_8_stepsUsed_: int
            out0_: _dafny.Seq
            out1_: bool
            out2_: bool
            out3_: int
            out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_4_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
            d_5_chunkGenerated_ = out0_
            d_6_stoppedOnOpenSpan_ = out1_
            d_7_stoppedOnEos_ = out2_
            d_8_stepsUsed_ = out3_
            generated = d_5_chunkGenerated_
            d_2_steps_ = (d_2_steps_) + (d_8_stepsUsed_)
            if d_6_stoppedOnOpenSpan_:
                d_9_cg_: _dafny.Seq
                d_10_ci_: bool
                d_11_cc_: _dafny.Seq
                out4_: _dafny.Seq
                out5_: bool
                out6_: _dafny.Seq
                out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                d_9_cg_ = out4_
                d_10_ci_ = out5_
                d_11_cc_ = out6_
                generated = d_9_cg_
                insideConstrainedOut = d_10_ci_
                currentConstrainedOut = d_11_cc_
            elif d_7_stoppedOnEos_:
                cost = d_2_steps_
                return generated, insideConstrainedOut, currentConstrainedOut, cost
        if (not(insideConstrainedOut)) and ((d_2_steps_) < (maxSteps)):
            d_12_cg_: _dafny.Seq
            d_13_ci_: bool
            d_14_cc_: _dafny.Seq
            out7_: _dafny.Seq
            out8_: bool
            out9_: _dafny.Seq
            out7_, out8_, out9_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_12_cg_ = out7_
            d_13_ci_ = out8_
            d_14_cc_ = out9_
            generated = d_12_cg_
            insideConstrainedOut = d_13_ci_
            currentConstrainedOut = d_14_cc_
            d_2_steps_ = (d_2_steps_) + (1)
        d_15_closeReserve_: int
        d_15_closeReserve_ = 5
        with _dafny.label("0"):
            while (((d_2_steps_) + (d_15_closeReserve_)) < (maxSteps)) and (insideConstrainedOut):
                with _dafny.c_label("0"):
                    if (parser).IsCompletePrefix(currentConstrainedOut):
                        d_16_narrow_: bool
                        out10_: bool
                        out10_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 2)
                        d_16_narrow_ = out10_
                        if d_16_narrow_:
                            d_17_cg_: _dafny.Seq
                            d_18_ci_: bool
                            d_19_cc_: _dafny.Seq
                            out11_: _dafny.Seq
                            out12_: bool
                            out13_: _dafny.Seq
                            out11_, out12_, out13_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_17_cg_ = out11_
                            d_18_ci_ = out12_
                            d_19_cc_ = out13_
                            generated = d_17_cg_
                            insideConstrainedOut = d_18_ci_
                            currentConstrainedOut = d_19_cc_
                            d_2_steps_ = (d_2_steps_) + (1)
                            raise _dafny.Break("0")
                        d_20_cg2_: _dafny.Seq
                        d_21_ci2_: bool
                        d_22_cc2_: _dafny.Seq
                        d_23_closed_: bool
                        out14_: _dafny.Seq
                        out15_: bool
                        out16_: _dafny.Seq
                        out17_: bool
                        out14_, out15_, out16_, out17_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                        d_20_cg2_ = out14_
                        d_21_ci2_ = out15_
                        d_22_cc2_ = out16_
                        d_23_closed_ = out17_
                        d_2_steps_ = (d_2_steps_) + (1)
                        if d_23_closed_:
                            generated = d_20_cg2_
                            insideConstrainedOut = d_21_ci2_
                            currentConstrainedOut = d_22_cc2_
                            raise _dafny.Break("0")
                        d_24_constrainedPrompt2_: _dafny.Seq
                        d_24_constrainedPrompt2_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_25_next2_: _dafny.Seq
                        out18_: _dafny.Seq
                        out18_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_24_constrainedPrompt2_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_3_semicolonTokens_, _dafny.BigRational('6e0'), 15, eosToken)
                        d_25_next2_ = out18_
                        d_2_steps_ = (d_2_steps_) + (1)
                        if (d_25_next2_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_26_ag_: _dafny.Seq
                            d_27_ai_: bool
                            d_28_ac_: _dafny.Seq
                            out19_: _dafny.Seq
                            out20_: bool
                            out21_: _dafny.Seq
                            out19_, out20_, out21_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_25_next2_)
                            d_26_ag_ = out19_
                            d_27_ai_ = out20_
                            d_28_ac_ = out21_
                            generated = d_26_ag_
                            insideConstrainedOut = d_27_ai_
                            currentConstrainedOut = d_28_ac_
                    elif True:
                        d_29_constrainedPrompt_: _dafny.Seq
                        d_29_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_30_validCount_: int
                        out22_: int
                        out22_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                        d_30_validCount_ = out22_
                        if (d_30_validCount_) == (0):
                            d_31_rg_: _dafny.Seq
                            d_32_rc_: _dafny.Seq
                            out23_: _dafny.Seq
                            out24_: _dafny.Seq
                            out23_, out24_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                            d_31_rg_ = out23_
                            d_32_rc_ = out24_
                            generated = d_31_rg_
                            currentConstrainedOut = d_32_rc_
                            if (parser).IsCompletePrefix(currentConstrainedOut):
                                d_33_cg_: _dafny.Seq
                                d_34_ci_: bool
                                d_35_cc_: _dafny.Seq
                                out25_: _dafny.Seq
                                out26_: bool
                                out27_: _dafny.Seq
                                out25_, out26_, out27_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_33_cg_ = out25_
                                d_34_ci_ = out26_
                                d_35_cc_ = out27_
                                generated = d_33_cg_
                                insideConstrainedOut = d_34_ci_
                                currentConstrainedOut = d_35_cc_
                                d_2_steps_ = (d_2_steps_) + (1)
                            elif True:
                                d_36_cg3_: _dafny.Seq
                                d_37_ci3_: bool
                                d_38_cc3_: _dafny.Seq
                                d_39_closed3_: bool
                                out28_: _dafny.Seq
                                out29_: bool
                                out30_: _dafny.Seq
                                out31_: bool
                                out28_, out29_, out30_, out31_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                                d_36_cg3_ = out28_
                                d_37_ci3_ = out29_
                                d_38_cc3_ = out30_
                                d_39_closed3_ = out31_
                                d_2_steps_ = (d_2_steps_) + (1)
                                if d_39_closed3_:
                                    generated = d_36_cg3_
                                    insideConstrainedOut = d_37_ci3_
                                    currentConstrainedOut = d_38_cc3_
                            raise _dafny.Break("0")
                        d_40_next_: _dafny.Seq
                        out32_: _dafny.Seq
                        out32_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_29_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_3_semicolonTokens_, _dafny.BigRational('6e0'), 15, eosToken)
                        d_40_next_ = out32_
                        d_2_steps_ = (d_2_steps_) + (1)
                        if (d_40_next_) == (eosToken):
                            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_2_steps_) < (maxSteps)):
                                d_41_cg_: _dafny.Seq
                                d_42_ci_: bool
                                d_43_cc_: _dafny.Seq
                                out33_: _dafny.Seq
                                out34_: bool
                                out35_: _dafny.Seq
                                out33_, out34_, out35_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_41_cg_ = out33_
                                d_42_ci_ = out34_
                                d_43_cc_ = out35_
                                generated = d_41_cg_
                                insideConstrainedOut = d_42_ci_
                                currentConstrainedOut = d_43_cc_
                                d_2_steps_ = (d_2_steps_) + (1)
                            raise _dafny.Break("0")
                        elif True:
                            d_44_ag_: _dafny.Seq
                            d_45_ai_: bool
                            d_46_ac_: _dafny.Seq
                            out36_: _dafny.Seq
                            out37_: bool
                            out38_: _dafny.Seq
                            out36_, out37_, out38_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_40_next_)
                            d_44_ag_ = out36_
                            d_45_ai_ = out37_
                            d_46_ac_ = out38_
                            generated = d_44_ag_
                            insideConstrainedOut = d_45_ai_
                            currentConstrainedOut = d_46_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            if (parser).IsCompletePrefix(currentConstrainedOut):
                d_47_cg_: _dafny.Seq
                d_48_ci_: bool
                d_49_cc_: _dafny.Seq
                out39_: _dafny.Seq
                out40_: bool
                out41_: _dafny.Seq
                out39_, out40_, out41_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_47_cg_ = out39_
                d_48_ci_ = out40_
                d_49_cc_ = out41_
                generated = d_47_cg_
                insideConstrainedOut = d_48_ci_
                currentConstrainedOut = d_49_cc_
                d_2_steps_ = (d_2_steps_) + (1)
            elif True:
                d_50_rg_: _dafny.Seq
                d_51_rc_: _dafny.Seq
                out42_: _dafny.Seq
                out43_: _dafny.Seq
                out42_, out43_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                d_50_rg_ = out42_
                d_51_rc_ = out43_
                generated = d_50_rg_
                currentConstrainedOut = d_51_rc_
                if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_2_steps_) < (maxSteps)):
                    d_52_cg_: _dafny.Seq
                    d_53_ci_: bool
                    d_54_cc_: _dafny.Seq
                    out44_: _dafny.Seq
                    out45_: bool
                    out46_: _dafny.Seq
                    out44_, out45_, out46_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                    d_52_cg_ = out44_
                    d_53_ci_ = out45_
                    d_54_cc_ = out46_
                    generated = d_52_cg_
                    insideConstrainedOut = d_53_ci_
                    currentConstrainedOut = d_54_cc_
                    d_2_steps_ = (d_2_steps_) + (1)
                elif (d_2_steps_) < (maxSteps):
                    d_55_cg4_: _dafny.Seq
                    d_56_ci4_: bool
                    d_57_cc4_: _dafny.Seq
                    d_58_closed4_: bool
                    out47_: _dafny.Seq
                    out48_: bool
                    out49_: _dafny.Seq
                    out50_: bool
                    out47_, out48_, out49_, out50_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                    d_55_cg4_ = out47_
                    d_56_ci4_ = out48_
                    d_57_cc4_ = out49_
                    d_58_closed4_ = out50_
                    d_2_steps_ = (d_2_steps_) + (1)
                    if d_58_closed4_:
                        generated = d_55_cg4_
                        insideConstrainedOut = d_56_ci4_
                        currentConstrainedOut = d_57_cc4_
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

