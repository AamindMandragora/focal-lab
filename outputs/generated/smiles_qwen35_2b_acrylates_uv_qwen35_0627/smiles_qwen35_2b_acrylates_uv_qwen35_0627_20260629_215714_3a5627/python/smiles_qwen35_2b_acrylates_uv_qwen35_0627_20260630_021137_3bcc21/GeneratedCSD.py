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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "You are generating a SMILES string for a novel acrylate molecule. Acrylates have core structure C=CC(=O)O or CC(=C)C(=O)O (methyl acrylate variant). Generate diverse ester/functional group substitution. Output only the SMILES.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_minConstrainedTokens_: int
        d_2_minConstrainedTokens_ = 30
        if (not(insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
            d_3_og_: _dafny.Seq
            d_4_oi_: bool
            d_5_oc_: _dafny.Seq
            out0_: _dafny.Seq
            out1_: bool
            out2_: _dafny.Seq
            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_3_og_ = out0_
            d_4_oi_ = out1_
            d_5_oc_ = out2_
            generated = d_3_og_
            insideConstrainedOut = d_4_oi_
            currentConstrainedOut = d_5_oc_
            d_1_steps_ = (d_1_steps_) + (1)
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        raise _dafny.Break("0")
                    d_6_remaining_: int
                    d_6_remaining_ = (maxSteps) - (d_1_steps_)
                    if (d_6_remaining_) <= (25):
                        d_7_csg_: _dafny.Seq
                        d_8_csi_: bool
                        d_9_csc_: _dafny.Seq
                        out3_: _dafny.Seq
                        out4_: bool
                        out5_: _dafny.Seq
                        out3_, out4_, out5_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_6_remaining_)
                        d_7_csg_ = out3_
                        d_8_csi_ = out4_
                        d_9_csc_ = out5_
                        generated = d_7_csg_
                        insideConstrainedOut = d_8_csi_
                        currentConstrainedOut = d_9_csc_
                        d_1_steps_ = maxSteps
                        raise _dafny.Break("0")
                    if (len(currentConstrainedOut)) >= (d_2_minConstrainedTokens_):
                        d_10_cg_: _dafny.Seq
                        d_11_ci_: bool
                        d_12_cc_: _dafny.Seq
                        d_13_closed_: bool
                        out6_: _dafny.Seq
                        out7_: bool
                        out8_: _dafny.Seq
                        out9_: bool
                        out6_, out7_, out8_, out9_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                        d_10_cg_ = out6_
                        d_11_ci_ = out7_
                        d_12_cc_ = out8_
                        d_13_closed_ = out9_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if d_13_closed_:
                            generated = d_10_cg_
                            insideConstrainedOut = d_11_ci_
                            currentConstrainedOut = d_12_cc_
                            raise _dafny.Break("0")
                        d_14_narrow_: bool
                        out10_: bool
                        out10_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 2)
                        d_14_narrow_ = out10_
                        if d_14_narrow_:
                            d_15_rg_: _dafny.Seq
                            d_16_rc_: _dafny.Seq
                            out11_: _dafny.Seq
                            out12_: _dafny.Seq
                            out11_, out12_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                            d_15_rg_ = out11_
                            d_16_rc_ = out12_
                            if (parser).IsCompletePrefix(d_16_rc_):
                                generated = d_15_rg_
                                currentConstrainedOut = d_16_rc_
                                if (d_1_steps_) < (maxSteps):
                                    d_17_cg2_: _dafny.Seq
                                    d_18_ci2_: bool
                                    d_19_cc2_: _dafny.Seq
                                    out13_: _dafny.Seq
                                    out14_: bool
                                    out15_: _dafny.Seq
                                    out13_, out14_, out15_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_17_cg2_ = out13_
                                    d_18_ci2_ = out14_
                                    d_19_cc2_ = out15_
                                    generated = d_17_cg2_
                                    insideConstrainedOut = d_18_ci2_
                                    currentConstrainedOut = d_19_cc2_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                raise _dafny.Break("0")
                        if not((parser).IsCompletePrefix(currentConstrainedOut)):
                            d_20_constrainedPrompt_: _dafny.Seq
                            d_20_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_21_penTokens_: _dafny.Seq
                            d_21_penTokens_ = _dafny.SeqWithoutIsStrInference([])
                            d_22_next_: _dafny.Seq
                            out16_: _dafny.Seq
                            out16_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_20_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('3e0'), d_21_penTokens_, _dafny.BigRational('2e0'), 4, eosToken)
                            d_22_next_ = out16_
                            if (d_22_next_) == (eosToken):
                                if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                    d_23_cg3_: _dafny.Seq
                                    d_24_ci3_: bool
                                    d_25_cc3_: _dafny.Seq
                                    out17_: _dafny.Seq
                                    out18_: bool
                                    out19_: _dafny.Seq
                                    out17_, out18_, out19_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_23_cg3_ = out17_
                                    d_24_ci3_ = out18_
                                    d_25_cc3_ = out19_
                                    generated = d_23_cg3_
                                    insideConstrainedOut = d_24_ci3_
                                    currentConstrainedOut = d_25_cc3_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                raise _dafny.Break("0")
                            elif True:
                                if not((parser).IsCompletePrefix(currentConstrainedOut)):
                                    d_26_ag_: _dafny.Seq
                                    d_27_ai_: bool
                                    d_28_ac_: _dafny.Seq
                                    out20_: _dafny.Seq
                                    out21_: bool
                                    out22_: _dafny.Seq
                                    out20_, out21_, out22_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_22_next_)
                                    d_26_ag_ = out20_
                                    d_27_ai_ = out21_
                                    d_28_ac_ = out22_
                                    generated = d_26_ag_
                                    insideConstrainedOut = d_27_ai_
                                    currentConstrainedOut = d_28_ac_
                        elif True:
                            d_29_cg4_: _dafny.Seq
                            d_30_ci4_: bool
                            d_31_cc4_: _dafny.Seq
                            out23_: _dafny.Seq
                            out24_: bool
                            out25_: _dafny.Seq
                            out23_, out24_, out25_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_29_cg4_ = out23_
                            d_30_ci4_ = out24_
                            d_31_cc4_ = out25_
                            generated = d_29_cg4_
                            insideConstrainedOut = d_30_ci4_
                            currentConstrainedOut = d_31_cc4_
                            d_1_steps_ = (d_1_steps_) + (1)
                            raise _dafny.Break("0")
                    elif True:
                        d_32_isComplete_: bool
                        d_32_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if (d_32_isComplete_) and ((len(currentConstrainedOut)) >= (8)):
                            d_33_constrainedPrompt_: _dafny.Seq
                            d_33_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_34_next_: _dafny.Seq
                            out26_: _dafny.Seq
                            out26_ = (d_0_helpers_).SafeTemperatureConstrainedStep(lm, parser, d_33_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('9e-1'), eosToken)
                            d_34_next_ = out26_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_34_next_) == (eosToken):
                                if (d_1_steps_) < (maxSteps):
                                    d_35_cg5_: _dafny.Seq
                                    d_36_ci5_: bool
                                    d_37_cc5_: _dafny.Seq
                                    out27_: _dafny.Seq
                                    out28_: bool
                                    out29_: _dafny.Seq
                                    out27_, out28_, out29_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_35_cg5_ = out27_
                                    d_36_ci5_ = out28_
                                    d_37_cc5_ = out29_
                                    generated = d_35_cg5_
                                    insideConstrainedOut = d_36_ci5_
                                    currentConstrainedOut = d_37_cc5_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                raise _dafny.Break("0")
                            elif True:
                                if not((parser).IsCompletePrefix(currentConstrainedOut)):
                                    d_38_ag2_: _dafny.Seq
                                    d_39_ai2_: bool
                                    d_40_ac2_: _dafny.Seq
                                    out30_: _dafny.Seq
                                    out31_: bool
                                    out32_: _dafny.Seq
                                    out30_, out31_, out32_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_34_next_)
                                    d_38_ag2_ = out30_
                                    d_39_ai2_ = out31_
                                    d_40_ac2_ = out32_
                                    generated = d_38_ag2_
                                    insideConstrainedOut = d_39_ai2_
                                    currentConstrainedOut = d_40_ac2_
                        elif not(d_32_isComplete_):
                            d_41_constrainedPrompt_: _dafny.Seq
                            d_41_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_42_next_: _dafny.Seq
                            out33_: _dafny.Seq
                            out33_ = (d_0_helpers_).SafeTemperatureConstrainedStep(lm, parser, d_41_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('8e-1'), eosToken)
                            d_42_next_ = out33_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_42_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                if not((parser).IsCompletePrefix(currentConstrainedOut)):
                                    d_43_ag3_: _dafny.Seq
                                    d_44_ai3_: bool
                                    d_45_ac3_: _dafny.Seq
                                    out34_: _dafny.Seq
                                    out35_: bool
                                    out36_: _dafny.Seq
                                    out34_, out35_, out36_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_42_next_)
                                    d_43_ag3_ = out34_
                                    d_44_ai3_ = out35_
                                    d_45_ac3_ = out36_
                                    generated = d_43_ag3_
                                    insideConstrainedOut = d_44_ai3_
                                    currentConstrainedOut = d_45_ac3_
                        elif True:
                            d_46_constrainedPrompt_: _dafny.Seq
                            d_46_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_47_next_: _dafny.Seq
                            out37_: _dafny.Seq
                            out37_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_46_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_47_next_ = out37_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_47_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                if not((parser).IsCompletePrefix(currentConstrainedOut)):
                                    d_48_ag4_: _dafny.Seq
                                    d_49_ai4_: bool
                                    d_50_ac4_: _dafny.Seq
                                    out38_: _dafny.Seq
                                    out39_: bool
                                    out40_: _dafny.Seq
                                    out38_, out39_, out40_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_47_next_)
                                    d_48_ag4_ = out38_
                                    d_49_ai4_ = out39_
                                    d_50_ac4_ = out40_
                                    generated = d_48_ag4_
                                    insideConstrainedOut = d_49_ai4_
                                    currentConstrainedOut = d_50_ac4_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_51_remaining2_: int
            d_51_remaining2_ = (maxSteps) - (d_1_steps_)
            d_52_csg2_: _dafny.Seq
            d_53_csi2_: bool
            d_54_csc2_: _dafny.Seq
            out41_: _dafny.Seq
            out42_: bool
            out43_: _dafny.Seq
            out41_, out42_, out43_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_51_remaining2_)
            d_52_csg2_ = out41_
            d_53_csi2_ = out42_
            d_54_csc2_ = out43_
            generated = d_52_csg2_
            insideConstrainedOut = d_53_csi2_
            currentConstrainedOut = d_54_csc2_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

