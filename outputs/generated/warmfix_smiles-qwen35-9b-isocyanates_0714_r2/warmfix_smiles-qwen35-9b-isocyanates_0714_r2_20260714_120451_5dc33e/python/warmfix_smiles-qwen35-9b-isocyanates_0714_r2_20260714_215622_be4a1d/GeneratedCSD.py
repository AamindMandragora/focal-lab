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
        d_1_steps_: int
        d_1_steps_ = 0
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate exactly one valid SMILES for an isocyanate (must contain N=C=O). Use diverse structures: O=C=NCC(F)(F)F, O=C=NCCCl, O=C=NC1CCCCC1, O=C=NCc1ccccc1, O=C=Nc1ccc(Cl)cc1, O=C=NC(C)(C)C, O=C=NCC=C, O=C=NCC#C, O=C=Nc1ccccn1, O=C=NCCOC, O=C=NCCSc1ccccc1, O=C=NC1CCCC1, O=C=NCC(C)C, O=C=NCCc1ccccc1, O=C=NCC(F)F. Output ONLY the SMILES string.")))
        if (not(insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
            d_2_og_: _dafny.Seq
            d_3_oi_: bool
            d_4_oc_: _dafny.Seq
            out0_: _dafny.Seq
            out1_: bool
            out2_: _dafny.Seq
            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_2_og_ = out0_
            d_3_oi_ = out1_
            d_4_oc_ = out2_
            generated = d_2_og_
            insideConstrainedOut = d_3_oi_
            currentConstrainedOut = d_4_oc_
            d_1_steps_ = (d_1_steps_) + (1)
        d_5_minSpanLength_: int
        d_5_minSpanLength_ = 5
        d_6_tokenCount_: int
        d_6_tokenCount_ = 0
        d_7_penalizeTokens_: _dafny.Seq
        d_7_penalizeTokens_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "C")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "CC")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "CCC"))])
        with _dafny.label("0"):
            while ((d_1_steps_) < (maxSteps)) and (insideConstrainedOut):
                with _dafny.c_label("0"):
                    if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((len(currentConstrainedOut)) >= (d_5_minSpanLength_)):
                        d_8_smilesStr_: _dafny.Seq
                        d_8_smilesStr_ = VerifiedDecoderAgent.CSDHelpers.PrefixToString(currentConstrainedOut)
                        d_9_hasNCO_: int
                        d_9_hasNCO_ = VerifiedDecoderAgent.CSDHelpers.CountSubstring(d_8_smilesStr_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "N=C=O")))
                        if (d_9_hasNCO_) > (0):
                            if (d_1_steps_) < (maxSteps):
                                d_10_cg_: _dafny.Seq
                                d_11_ci_: bool
                                d_12_cc_: _dafny.Seq
                                out3_: _dafny.Seq
                                out4_: bool
                                out5_: _dafny.Seq
                                out3_, out4_, out5_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_10_cg_ = out3_
                                d_11_ci_ = out4_
                                d_12_cc_ = out5_
                                generated = d_10_cg_
                                insideConstrainedOut = d_11_ci_
                                currentConstrainedOut = d_12_cc_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                raise _dafny.Break("0")
                        elif True:
                            d_13_constrainedPrompt_: _dafny.Seq
                            d_13_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_14_next_: _dafny.Seq
                            out6_: _dafny.Seq
                            out6_ = (d_0_helpers_).SafeTemperatureConstrainedStep(lm, parser, d_13_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('16e-1'), eosToken)
                            d_14_next_ = out6_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_14_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_15_ag_: _dafny.Seq
                                d_16_ai_: bool
                                d_17_ac_: _dafny.Seq
                                out7_: _dafny.Seq
                                out8_: bool
                                out9_: _dafny.Seq
                                out7_, out8_, out9_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_14_next_)
                                d_15_ag_ = out7_
                                d_16_ai_ = out8_
                                d_17_ac_ = out9_
                                generated = d_15_ag_
                                insideConstrainedOut = d_16_ai_
                                currentConstrainedOut = d_17_ac_
                                d_6_tokenCount_ = (d_6_tokenCount_) + (1)
                    elif True:
                        d_18_constrainedPrompt_: _dafny.Seq
                        d_18_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_19_next_: _dafny.Seq
                        d_19_next_ = eosToken
                        if (d_6_tokenCount_) < (6):
                            out10_: _dafny.Seq
                            out10_ = (d_0_helpers_).SafePenalizedConstrainedStep(lm, parser, d_18_constrainedPrompt_, currentConstrainedOut, d_7_penalizeTokens_, _dafny.BigRational('5e0'), eosToken)
                            d_19_next_ = out10_
                        elif (d_6_tokenCount_) < (20):
                            out11_: _dafny.Seq
                            out11_ = (d_0_helpers_).SafeTemperatureConstrainedStep(lm, parser, d_18_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('16e-1'), eosToken)
                            d_19_next_ = out11_
                        elif True:
                            out12_: _dafny.Seq
                            out12_ = (d_0_helpers_).SafeTemperatureConstrainedStep(lm, parser, d_18_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('12e-1'), eosToken)
                            d_19_next_ = out12_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_19_next_) == (eosToken):
                            if (((parser).IsCompletePrefix(currentConstrainedOut)) and ((len(currentConstrainedOut)) >= (d_5_minSpanLength_))) and ((d_1_steps_) < (maxSteps)):
                                d_20_smilesStr_: _dafny.Seq
                                d_20_smilesStr_ = VerifiedDecoderAgent.CSDHelpers.PrefixToString(currentConstrainedOut)
                                d_21_hasNCO_: int
                                d_21_hasNCO_ = VerifiedDecoderAgent.CSDHelpers.CountSubstring(d_20_smilesStr_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "N=C=O")))
                                if (d_21_hasNCO_) > (0):
                                    d_22_cg_: _dafny.Seq
                                    d_23_ci_: bool
                                    d_24_cc_: _dafny.Seq
                                    out13_: _dafny.Seq
                                    out14_: bool
                                    out15_: _dafny.Seq
                                    out13_, out14_, out15_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_22_cg_ = out13_
                                    d_23_ci_ = out14_
                                    d_24_cc_ = out15_
                                    generated = d_22_cg_
                                    insideConstrainedOut = d_23_ci_
                                    currentConstrainedOut = d_24_cc_
                                    d_1_steps_ = (d_1_steps_) + (1)
                            raise _dafny.Break("0")
                        elif True:
                            d_25_ag_: _dafny.Seq
                            d_26_ai_: bool
                            d_27_ac_: _dafny.Seq
                            out16_: _dafny.Seq
                            out17_: bool
                            out18_: _dafny.Seq
                            out16_, out17_, out18_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_19_next_)
                            d_25_ag_ = out16_
                            d_26_ai_ = out17_
                            d_27_ac_ = out18_
                            generated = d_25_ag_
                            insideConstrainedOut = d_26_ai_
                            currentConstrainedOut = d_27_ac_
                            d_6_tokenCount_ = (d_6_tokenCount_) + (1)
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_28_smilesStr_: _dafny.Seq
            d_28_smilesStr_ = VerifiedDecoderAgent.CSDHelpers.PrefixToString(currentConstrainedOut)
            d_29_hasNCO_: int
            d_29_hasNCO_ = VerifiedDecoderAgent.CSDHelpers.CountSubstring(d_28_smilesStr_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "N=C=O")))
            if (d_29_hasNCO_) > (0):
                d_30_closeBudget_: int
                d_30_closeBudget_ = (maxSteps) - (d_1_steps_)
                d_31_cg_: _dafny.Seq
                d_32_ci_: bool
                d_33_cc_: _dafny.Seq
                out19_: _dafny.Seq
                out20_: bool
                out21_: _dafny.Seq
                out19_, out20_, out21_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_30_closeBudget_)
                d_31_cg_ = out19_
                d_32_ci_ = out20_
                d_33_cc_ = out21_
                generated = d_31_cg_
                insideConstrainedOut = d_32_ci_
                currentConstrainedOut = d_33_cc_
                d_1_steps_ = maxSteps
            elif True:
                with _dafny.label("2_1_0"):
                    while ((d_1_steps_) < (maxSteps)) and (insideConstrainedOut):
                        with _dafny.c_label("2_1_0"):
                            d_34_constrainedPrompt_: _dafny.Seq
                            d_34_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_35_next_: _dafny.Seq
                            out22_: _dafny.Seq
                            out22_ = (d_0_helpers_).SafeTemperatureConstrainedStep(lm, parser, d_34_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('14e-1'), eosToken)
                            d_35_next_ = out22_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_35_next_) == (eosToken):
                                if (((parser).IsCompletePrefix(currentConstrainedOut)) and ((len(currentConstrainedOut)) >= (d_5_minSpanLength_))) and ((d_1_steps_) < (maxSteps)):
                                    d_36_str2_: _dafny.Seq
                                    d_36_str2_ = VerifiedDecoderAgent.CSDHelpers.PrefixToString(currentConstrainedOut)
                                    d_37_nco2_: int
                                    d_37_nco2_ = VerifiedDecoderAgent.CSDHelpers.CountSubstring(d_36_str2_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "N=C=O")))
                                    if (d_37_nco2_) > (0):
                                        d_38_cg_: _dafny.Seq
                                        d_39_ci_: bool
                                        d_40_cc_: _dafny.Seq
                                        out23_: _dafny.Seq
                                        out24_: bool
                                        out25_: _dafny.Seq
                                        out23_, out24_, out25_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                        d_38_cg_ = out23_
                                        d_39_ci_ = out24_
                                        d_40_cc_ = out25_
                                        generated = d_38_cg_
                                        insideConstrainedOut = d_39_ci_
                                        currentConstrainedOut = d_40_cc_
                                        d_1_steps_ = (d_1_steps_) + (1)
                                raise _dafny.Break("2_1_0")
                            elif True:
                                d_41_ag_: _dafny.Seq
                                d_42_ai_: bool
                                d_43_ac_: _dafny.Seq
                                out26_: _dafny.Seq
                                out27_: bool
                                out28_: _dafny.Seq
                                out26_, out27_, out28_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_35_next_)
                                d_41_ag_ = out26_
                                d_42_ai_ = out27_
                                d_43_ac_ = out28_
                                generated = d_41_ag_
                                insideConstrainedOut = d_42_ai_
                                currentConstrainedOut = d_43_ac_
                                if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((len(currentConstrainedOut)) >= (d_5_minSpanLength_)):
                                    d_44_str3_: _dafny.Seq
                                    d_44_str3_ = VerifiedDecoderAgent.CSDHelpers.PrefixToString(currentConstrainedOut)
                                    d_45_nco3_: int
                                    d_45_nco3_ = VerifiedDecoderAgent.CSDHelpers.CountSubstring(d_44_str3_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "N=C=O")))
                                    if ((d_45_nco3_) > (0)) and ((d_1_steps_) < (maxSteps)):
                                        d_46_cg_: _dafny.Seq
                                        d_47_ci_: bool
                                        d_48_cc_: _dafny.Seq
                                        out29_: _dafny.Seq
                                        out30_: bool
                                        out31_: _dafny.Seq
                                        out29_, out30_, out31_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                        d_46_cg_ = out29_
                                        d_47_ci_ = out30_
                                        d_48_cc_ = out31_
                                        generated = d_46_cg_
                                        insideConstrainedOut = d_47_ci_
                                        currentConstrainedOut = d_48_cc_
                                        d_1_steps_ = (d_1_steps_) + (1)
                                        raise _dafny.Break("2_1_0")
                            pass
                    pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

