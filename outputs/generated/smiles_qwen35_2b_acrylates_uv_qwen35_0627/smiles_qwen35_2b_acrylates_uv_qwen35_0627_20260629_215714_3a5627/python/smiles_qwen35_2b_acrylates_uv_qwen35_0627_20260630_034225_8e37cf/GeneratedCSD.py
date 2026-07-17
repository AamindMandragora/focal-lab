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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Task: output exactly one SMILES string for an acrylate ester. An acrylate ester MUST begin with C=CC(=O)O followed by a carbon chain R. Valid acrylates: C=CC(=O)OCC (ethyl acrylate), C=CC(=O)OCCC (propyl acrylate), C=CC(=O)OC(C)C (isopropyl acrylate), C=CC(=O)OCCCC (butyl acrylate), C=CC(=O)OC(C)(C)C (tert-butyl acrylate), C=CC(=O)OCCCCC (pentyl acrylate), C=CC(=O)OCC(C)C (isobutyl acrylate), C=CC(=O)OCCO (2-hydroxyethyl acrylate). Output ONLY the SMILES string. NO other text.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_minConstrainedTokens_: int
        d_2_minConstrainedTokens_ = 20
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
                    if (d_6_remaining_) <= (35):
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
                    d_10_constrainedLen_: int
                    d_10_constrainedLen_ = len(currentConstrainedOut)
                    d_11_constrainedPrompt_: _dafny.Seq
                    d_11_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (d_10_constrainedLen_):]))
                    if (d_10_constrainedLen_) >= (d_2_minConstrainedTokens_):
                        d_12_cg_: _dafny.Seq
                        d_13_ci_: bool
                        d_14_cc_: _dafny.Seq
                        d_15_closed_: bool
                        out6_: _dafny.Seq
                        out7_: bool
                        out8_: _dafny.Seq
                        out9_: bool
                        out6_, out7_, out8_, out9_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                        d_12_cg_ = out6_
                        d_13_ci_ = out7_
                        d_14_cc_ = out8_
                        d_15_closed_ = out9_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if d_15_closed_:
                            generated = d_12_cg_
                            insideConstrainedOut = d_13_ci_
                            currentConstrainedOut = d_14_cc_
                            raise _dafny.Break("0")
                        d_16_narrow_: bool
                        out10_: bool
                        out10_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 2)
                        d_16_narrow_ = out10_
                        if d_16_narrow_:
                            if (parser).IsCompletePrefix(currentConstrainedOut):
                                if (d_1_steps_) < (maxSteps):
                                    d_17_cg2_: _dafny.Seq
                                    d_18_ci2_: bool
                                    d_19_cc2_: _dafny.Seq
                                    out11_: _dafny.Seq
                                    out12_: bool
                                    out13_: _dafny.Seq
                                    out11_, out12_, out13_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_17_cg2_ = out11_
                                    d_18_ci2_ = out12_
                                    d_19_cc2_ = out13_
                                    generated = d_17_cg2_
                                    insideConstrainedOut = d_18_ci2_
                                    currentConstrainedOut = d_19_cc2_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                raise _dafny.Break("0")
                            elif True:
                                d_20_rg_: _dafny.Seq
                                d_21_rc_: _dafny.Seq
                                out14_: _dafny.Seq
                                out15_: _dafny.Seq
                                out14_, out15_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                                d_20_rg_ = out14_
                                d_21_rc_ = out15_
                                if ((parser).IsCompletePrefix(d_21_rc_)) and ((len(d_21_rc_)) >= (10)):
                                    generated = d_20_rg_
                                    currentConstrainedOut = d_21_rc_
                                    if (d_1_steps_) < (maxSteps):
                                        d_22_cg3_: _dafny.Seq
                                        d_23_ci3_: bool
                                        d_24_cc3_: _dafny.Seq
                                        out16_: _dafny.Seq
                                        out17_: bool
                                        out18_: _dafny.Seq
                                        out16_, out17_, out18_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                        d_22_cg3_ = out16_
                                        d_23_ci3_ = out17_
                                        d_24_cc3_ = out18_
                                        generated = d_22_cg3_
                                        insideConstrainedOut = d_23_ci3_
                                        currentConstrainedOut = d_24_cc3_
                                        d_1_steps_ = (d_1_steps_) + (1)
                                    raise _dafny.Break("0")
                                d_25_remaining2_: int
                                d_25_remaining2_ = (maxSteps) - (d_1_steps_)
                                d_26_csg2_: _dafny.Seq
                                d_27_csi2_: bool
                                d_28_csc2_: _dafny.Seq
                                out19_: _dafny.Seq
                                out20_: bool
                                out21_: _dafny.Seq
                                out19_, out20_, out21_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_25_remaining2_)
                                d_26_csg2_ = out19_
                                d_27_csi2_ = out20_
                                d_28_csc2_ = out21_
                                generated = d_26_csg2_
                                insideConstrainedOut = d_27_csi2_
                                currentConstrainedOut = d_28_csc2_
                                d_1_steps_ = maxSteps
                                raise _dafny.Break("0")
                        if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((d_1_steps_) < (maxSteps)):
                            d_29_penTokens_: _dafny.Seq
                            d_29_penTokens_ = _dafny.SeqWithoutIsStrInference([])
                            d_30_next_: _dafny.Seq
                            out22_: _dafny.Seq
                            out22_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_11_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('3e0'), d_29_penTokens_, _dafny.BigRational('2e0'), 4, eosToken)
                            d_30_next_ = out22_
                            if (d_30_next_) == (eosToken):
                                if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                    d_31_cg4_: _dafny.Seq
                                    d_32_ci4_: bool
                                    d_33_cc4_: _dafny.Seq
                                    out23_: _dafny.Seq
                                    out24_: bool
                                    out25_: _dafny.Seq
                                    out23_, out24_, out25_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_31_cg4_ = out23_
                                    d_32_ci4_ = out24_
                                    d_33_cc4_ = out25_
                                    generated = d_31_cg4_
                                    insideConstrainedOut = d_32_ci4_
                                    currentConstrainedOut = d_33_cc4_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                raise _dafny.Break("0")
                            elif True:
                                if not((parser).IsCompletePrefix(currentConstrainedOut)):
                                    d_34_valid_: bool
                                    out26_: bool
                                    out26_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_30_next_)
                                    d_34_valid_ = out26_
                                    if d_34_valid_:
                                        d_35_ag_: _dafny.Seq
                                        d_36_ai_: bool
                                        d_37_ac_: _dafny.Seq
                                        out27_: _dafny.Seq
                                        out28_: bool
                                        out29_: _dafny.Seq
                                        out27_, out28_, out29_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_30_next_)
                                        d_35_ag_ = out27_
                                        d_36_ai_ = out28_
                                        d_37_ac_ = out29_
                                        generated = d_35_ag_
                                        insideConstrainedOut = d_36_ai_
                                        currentConstrainedOut = d_37_ac_
                                elif True:
                                    if (d_1_steps_) < (maxSteps):
                                        d_38_cg5_: _dafny.Seq
                                        d_39_ci5_: bool
                                        d_40_cc5_: _dafny.Seq
                                        out30_: _dafny.Seq
                                        out31_: bool
                                        out32_: _dafny.Seq
                                        out30_, out31_, out32_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                        d_38_cg5_ = out30_
                                        d_39_ci5_ = out31_
                                        d_40_cc5_ = out32_
                                        generated = d_38_cg5_
                                        insideConstrainedOut = d_39_ci5_
                                        currentConstrainedOut = d_40_cc5_
                                        d_1_steps_ = (d_1_steps_) + (1)
                                    raise _dafny.Break("0")
                        elif ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                            d_41_cg6_: _dafny.Seq
                            d_42_ci6_: bool
                            d_43_cc6_: _dafny.Seq
                            out33_: _dafny.Seq
                            out34_: bool
                            out35_: _dafny.Seq
                            out33_, out34_, out35_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_41_cg6_ = out33_
                            d_42_ci6_ = out34_
                            d_43_cc6_ = out35_
                            generated = d_41_cg6_
                            insideConstrainedOut = d_42_ci6_
                            currentConstrainedOut = d_43_cc6_
                            d_1_steps_ = (d_1_steps_) + (1)
                            raise _dafny.Break("0")
                    elif True:
                        if not((parser).IsCompletePrefix(currentConstrainedOut)):
                            d_44_next_: _dafny.Seq
                            out36_: _dafny.Seq
                            out36_ = (d_0_helpers_).SafeTemperatureConstrainedStep(lm, parser, d_11_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('7e-1'), eosToken)
                            d_44_next_ = out36_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_44_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                if not((parser).IsCompletePrefix(currentConstrainedOut)):
                                    d_45_valid_: bool
                                    out37_: bool
                                    out37_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_44_next_)
                                    d_45_valid_ = out37_
                                    if d_45_valid_:
                                        d_46_ag_: _dafny.Seq
                                        d_47_ai_: bool
                                        d_48_ac_: _dafny.Seq
                                        out38_: _dafny.Seq
                                        out39_: bool
                                        out40_: _dafny.Seq
                                        out38_, out39_, out40_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_44_next_)
                                        d_46_ag_ = out38_
                                        d_47_ai_ = out39_
                                        d_48_ac_ = out40_
                                        generated = d_46_ag_
                                        insideConstrainedOut = d_47_ai_
                                        currentConstrainedOut = d_48_ac_
                        elif True:
                            d_49_next_: _dafny.Seq
                            out41_: _dafny.Seq
                            out41_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_11_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_49_next_ = out41_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_49_next_) == (eosToken):
                                if ((len(currentConstrainedOut)) >= (10)) and ((d_1_steps_) < (maxSteps)):
                                    d_50_cg7_: _dafny.Seq
                                    d_51_ci7_: bool
                                    d_52_cc7_: _dafny.Seq
                                    out42_: _dafny.Seq
                                    out43_: bool
                                    out44_: _dafny.Seq
                                    out42_, out43_, out44_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_50_cg7_ = out42_
                                    d_51_ci7_ = out43_
                                    d_52_cc7_ = out44_
                                    generated = d_50_cg7_
                                    insideConstrainedOut = d_51_ci7_
                                    currentConstrainedOut = d_52_cc7_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                raise _dafny.Break("0")
                            elif True:
                                if not((parser).IsCompletePrefix(currentConstrainedOut)):
                                    d_53_valid_: bool
                                    out45_: bool
                                    out45_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_49_next_)
                                    d_53_valid_ = out45_
                                    if d_53_valid_:
                                        d_54_ag_: _dafny.Seq
                                        d_55_ai_: bool
                                        d_56_ac_: _dafny.Seq
                                        out46_: _dafny.Seq
                                        out47_: bool
                                        out48_: _dafny.Seq
                                        out46_, out47_, out48_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_49_next_)
                                        d_54_ag_ = out46_
                                        d_55_ai_ = out47_
                                        d_56_ac_ = out48_
                                        generated = d_54_ag_
                                        insideConstrainedOut = d_55_ai_
                                        currentConstrainedOut = d_56_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_57_remaining3_: int
            d_57_remaining3_ = (maxSteps) - (d_1_steps_)
            d_58_csg3_: _dafny.Seq
            d_59_csi3_: bool
            d_60_csc3_: _dafny.Seq
            out49_: _dafny.Seq
            out50_: bool
            out51_: _dafny.Seq
            out49_, out50_, out51_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_57_remaining3_)
            d_58_csg3_ = out49_
            d_59_csi3_ = out50_
            d_60_csc3_ = out51_
            generated = d_58_csg3_
            insideConstrainedOut = d_59_csi3_
            currentConstrainedOut = d_60_csc3_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

