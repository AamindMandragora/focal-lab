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
        d_1_guidance_: _dafny.Seq
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate a valid SMILES for an ACRYLATE molecule. Acrylate core: C=CC(=O)O[R] or C=C(C)C(=O)O[R]. Examples of DIVERSE R groups: methyl C=CC(=O)OC, ethyl C=CC(=O)OCC, propyl C=CC(=O)OCCC, butyl C=CC(=O)OCCCC, pentyl C=CC(=O)OCCCCC, hexyl C=CC(=O)OCCCCCC, heptyl C=CC(=O)OCCCCCCC, octyl C=CC(=O)OCCCCCCCC, nonyl C=CC(=O)OCCCCCCCCC, decyl C=CC(=O)OCCCCCCCCCC, 2-hydroxyethyl C=CC(=O)OCCO, 3-hydroxypropyl C=CC(=O)OCCCOH, isobutyl C=CC(=O)OCC(C)C, isopropyl C=CC(=O)OC(C)C, tert-butyl C=CC(=O)OC(C)(C)C, neopentyl C=CC(=O)OCC(C)(C)C, cyclohexyl C=CC(=O)OC1CCCCC1, cyclopentyl C=CC(=O)OC1CCCC1, 2-ethylhexyl C=CC(=O)OCC(CC)CCCC, ethyl methacrylate C=C(C)C(=O)OCC, butyl methacrylate C=C(C)C(=O)OCCCC, benzyl C=CC(=O)OCc1ccccc1, 2-ethoxyethyl C=CC(=O)OCCOCC, glycidyl C=CC(=O)OCC1CO1, allyl C=CC(=O)OCC=C, furfuryl C=CC(=O)OCc1ccco1. Output ONLY the SMILES."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_minLength_: int
        d_3_minLength_ = 12
        if (not(insideConstrainedOut)) and ((d_2_steps_) < (maxSteps)):
            d_4_og_: _dafny.Seq
            d_5_oi_: bool
            d_6_oc_: _dafny.Seq
            out0_: _dafny.Seq
            out1_: bool
            out2_: _dafny.Seq
            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_4_og_ = out0_
            d_5_oi_ = out1_
            d_6_oc_ = out2_
            generated = d_4_og_
            insideConstrainedOut = d_5_oi_
            currentConstrainedOut = d_6_oc_
            d_2_steps_ = (d_2_steps_) + (1)
        with _dafny.label("0"):
            while (d_2_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        raise _dafny.Break("0")
                    elif ((parser).IsCompletePrefix(currentConstrainedOut)) and ((len(currentConstrainedOut)) >= (d_3_minLength_)):
                        d_7_occ1_: int
                        out3_: int
                        out3_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "1")))
                        d_7_occ1_ = out3_
                        d_8_occ2_: int
                        out4_: int
                        out4_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "2")))
                        d_8_occ2_ = out4_
                        d_9_occ3_: int
                        out5_: int
                        out5_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "3")))
                        d_9_occ3_ = out5_
                        d_10_openParen_: int
                        out6_: int
                        out6_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "(")))
                        d_10_openParen_ = out6_
                        d_11_closeParen_: int
                        out7_: int
                        out7_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ")")))
                        d_11_closeParen_ = out7_
                        d_12_ringsBalanced_: bool
                        d_12_ringsBalanced_ = ((((_dafny.euclidian_modulus(d_7_occ1_, 2)) == (0)) and ((_dafny.euclidian_modulus(d_8_occ2_, 2)) == (0))) and ((_dafny.euclidian_modulus(d_9_occ3_, 2)) == (0))) and ((d_10_openParen_) == (d_11_closeParen_))
                        if d_12_ringsBalanced_:
                            d_13_cg_: _dafny.Seq
                            d_14_ci_: bool
                            d_15_cc_: _dafny.Seq
                            out8_: _dafny.Seq
                            out9_: bool
                            out10_: _dafny.Seq
                            out8_, out9_, out10_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_13_cg_ = out8_
                            d_14_ci_ = out9_
                            d_15_cc_ = out10_
                            generated = d_13_cg_
                            insideConstrainedOut = d_14_ci_
                            currentConstrainedOut = d_15_cc_
                            d_2_steps_ = (d_2_steps_) + (1)
                            raise _dafny.Break("0")
                        elif True:
                            d_16_constrainedPrompt_: _dafny.Seq
                            d_16_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_17_next_: _dafny.Seq
                            out11_: _dafny.Seq
                            out11_ = (d_0_helpers_).SafeTemperatureConstrainedStep(lm, parser, d_16_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('13e-1'), eosToken)
                            d_17_next_ = out11_
                            d_2_steps_ = (d_2_steps_) + (1)
                            if (d_17_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_18_ag_: _dafny.Seq
                                d_19_ai_: bool
                                d_20_ac_: _dafny.Seq
                                out12_: _dafny.Seq
                                out13_: bool
                                out14_: _dafny.Seq
                                out12_, out13_, out14_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_17_next_)
                                d_18_ag_ = out12_
                                d_19_ai_ = out13_
                                d_20_ac_ = out14_
                                generated = d_18_ag_
                                insideConstrainedOut = d_19_ai_
                                currentConstrainedOut = d_20_ac_
                    elif True:
                        d_21_constrainedPrompt_: _dafny.Seq
                        d_21_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_22_spanLen_: int
                        d_22_spanLen_ = len(currentConstrainedOut)
                        d_23_next_: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                        if (d_22_spanLen_) < (5):
                            d_24_nextSoft_: _dafny.Seq
                            d_25_softOk_: bool
                            out15_: _dafny.Seq
                            out16_: bool
                            out15_, out16_ = (d_0_helpers_).SafeSoftConstrainedStep(lm, parser, d_21_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('8e0'), eosToken)
                            d_24_nextSoft_ = out15_
                            d_25_softOk_ = out16_
                            d_23_next_ = d_24_nextSoft_
                        elif True:
                            d_26_ringDigitValid_: bool
                            out17_: bool
                            out17_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "1")))
                            d_26_ringDigitValid_ = out17_
                            if d_26_ringDigitValid_:
                                out18_: _dafny.Seq
                                out18_ = (d_0_helpers_).SafeTemperatureConstrainedStep(lm, parser, d_21_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('15e-1'), eosToken)
                                d_23_next_ = out18_
                            elif True:
                                out19_: _dafny.Seq
                                out19_ = (d_0_helpers_).SafeTemperatureConstrainedStep(lm, parser, d_21_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('18e-1'), eosToken)
                                d_23_next_ = out19_
                        d_2_steps_ = (d_2_steps_) + (1)
                        if (d_23_next_) == (eosToken):
                            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((len(currentConstrainedOut)) >= (d_3_minLength_)):
                                d_27_occ1_: int
                                out20_: int
                                out20_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "1")))
                                d_27_occ1_ = out20_
                                d_28_occ2_: int
                                out21_: int
                                out21_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "2")))
                                d_28_occ2_ = out21_
                                d_29_occ3_: int
                                out22_: int
                                out22_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "3")))
                                d_29_occ3_ = out22_
                                d_30_openParen_: int
                                out23_: int
                                out23_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "(")))
                                d_30_openParen_ = out23_
                                d_31_closeParen_: int
                                out24_: int
                                out24_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ")")))
                                d_31_closeParen_ = out24_
                                d_32_ringsBalanced_: bool
                                d_32_ringsBalanced_ = ((((_dafny.euclidian_modulus(d_27_occ1_, 2)) == (0)) and ((_dafny.euclidian_modulus(d_28_occ2_, 2)) == (0))) and ((_dafny.euclidian_modulus(d_29_occ3_, 2)) == (0))) and ((d_30_openParen_) == (d_31_closeParen_))
                                if (d_32_ringsBalanced_) and ((d_2_steps_) < (maxSteps)):
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
                            raise _dafny.Break("0")
                        elif True:
                            d_36_ag_: _dafny.Seq
                            d_37_ai_: bool
                            d_38_ac_: _dafny.Seq
                            out28_: _dafny.Seq
                            out29_: bool
                            out30_: _dafny.Seq
                            out28_, out29_, out30_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_23_next_)
                            d_36_ag_ = out28_
                            d_37_ai_ = out29_
                            d_38_ac_ = out30_
                            generated = d_36_ag_
                            insideConstrainedOut = d_37_ai_
                            currentConstrainedOut = d_38_ac_
                    pass
            pass
        if (((insideConstrainedOut) and ((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) >= (d_3_minLength_))) and ((d_2_steps_) < (maxSteps)):
            d_39_occ1f_: int
            out31_: int
            out31_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "1")))
            d_39_occ1f_ = out31_
            d_40_occ2f_: int
            out32_: int
            out32_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "2")))
            d_40_occ2f_ = out32_
            d_41_occ3f_: int
            out33_: int
            out33_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "3")))
            d_41_occ3f_ = out33_
            d_42_openParenf_: int
            out34_: int
            out34_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "(")))
            d_42_openParenf_ = out34_
            d_43_closeParenf_: int
            out35_: int
            out35_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ")")))
            d_43_closeParenf_ = out35_
            d_44_ringsBalancedf_: bool
            d_44_ringsBalancedf_ = ((((_dafny.euclidian_modulus(d_39_occ1f_, 2)) == (0)) and ((_dafny.euclidian_modulus(d_40_occ2f_, 2)) == (0))) and ((_dafny.euclidian_modulus(d_41_occ3f_, 2)) == (0))) and ((d_42_openParenf_) == (d_43_closeParenf_))
            if d_44_ringsBalancedf_:
                d_45_cg_: _dafny.Seq
                d_46_ci_: bool
                d_47_cc_: _dafny.Seq
                out36_: _dafny.Seq
                out37_: bool
                out38_: _dafny.Seq
                out36_, out37_, out38_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_45_cg_ = out36_
                d_46_ci_ = out37_
                d_47_cc_ = out38_
                generated = d_45_cg_
                insideConstrainedOut = d_46_ci_
                currentConstrainedOut = d_47_cc_
                d_2_steps_ = (d_2_steps_) + (1)
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

