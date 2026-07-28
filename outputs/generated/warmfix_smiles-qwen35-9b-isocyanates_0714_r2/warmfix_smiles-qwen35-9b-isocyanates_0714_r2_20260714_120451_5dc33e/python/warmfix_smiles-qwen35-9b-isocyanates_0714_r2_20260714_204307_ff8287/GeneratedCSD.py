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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate exactly one valid SMILES for an isocyanate (R-N=C=O). The SMILES must contain the N=C=O group. Generate a UNIQUE, NOVEL molecule NOT from any examples. Use DIVERSE structural classes: methyl O=C=NC, ethyl O=C=NCC, propyl O=C=NCCC, butyl O=C=NCCCC, hexyl O=C=NCCCCCC, heptyl O=C=NCCCCCCC, isopropyl O=C=NC(C)C, isobutyl O=C=NCC(C)C, sec-butyl O=C=NC(C)CC, tert-butyl O=C=NC(C)(C)C, tert-amyl O=C=NC(C)(C)CC, neopentyl O=C=NCC(C)(C)C, cyclopropyl O=C=NC1CC1, cyclobutyl O=C=NC1CCC1, cyclopentyl O=C=NC1CCCC1, cycloheptyl O=C=NC1CCCCCC1, cyclooctyl O=C=NC1CCCCCCC1, 2-ethylhexyl O=C=NCC(CC)CCCC, 3-methylbutyl O=C=NCCC(C)C, allyl O=C=NCC=C, propargyl O=C=NCC#C, 2-chloroethyl O=C=NCCCl, trifluoromethyl O=C=NC(F)(F)F, 2,2,2-trifluoroethyl O=C=NCC(F)(F)F, benzyl O=C=NCc1ccccc1, phenethyl O=C=NCCc1ccccc1, 3-chloropropyl O=C=NCCCCl, adamantyl O=C=NC12CC3CC(CC(C3)C1)C2. Output ONLY the SMILES, nothing else.")))
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
        d_5_minSpanLength_ = 4
        d_6_exemplarSkips_: int
        d_6_exemplarSkips_ = 0
        with _dafny.label("0"):
            while ((d_1_steps_) < (maxSteps)) and (insideConstrainedOut):
                with _dafny.c_label("0"):
                    if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((len(currentConstrainedOut)) >= (d_5_minSpanLength_)):
                        d_7_smilesStr_: _dafny.Seq
                        d_7_smilesStr_ = VerifiedDecoderAgent.CSDHelpers.PrefixToString(currentConstrainedOut)
                        d_8_hasNCO_: int
                        d_8_hasNCO_ = VerifiedDecoderAgent.CSDHelpers.CountSubstring(d_7_smilesStr_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "N=C=O")))
                        if (d_8_hasNCO_) > (0):
                            d_9_seenInPrompt_: bool
                            out3_: bool
                            out3_ = (d_0_helpers_).PrefixAppearsInPrompt(lm, currentConstrainedOut)
                            d_9_seenInPrompt_ = out3_
                            if ((d_9_seenInPrompt_) and ((d_6_exemplarSkips_) < (5))) and (((d_1_steps_) + (2)) <= (maxSteps)):
                                d_6_exemplarSkips_ = (d_6_exemplarSkips_) + (1)
                                d_10_constrainedPrompt_: _dafny.Seq
                                d_10_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_11_next_: _dafny.Seq
                                out4_: _dafny.Seq
                                out4_ = (d_0_helpers_).SafeTemperatureConstrainedStep(lm, parser, d_10_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('16e-1'), eosToken)
                                d_11_next_ = out4_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_11_next_) == (eosToken):
                                    if (d_1_steps_) < (maxSteps):
                                        d_12_cg_: _dafny.Seq
                                        d_13_ci_: bool
                                        d_14_cc_: _dafny.Seq
                                        out5_: _dafny.Seq
                                        out6_: bool
                                        out7_: _dafny.Seq
                                        out5_, out6_, out7_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                        d_12_cg_ = out5_
                                        d_13_ci_ = out6_
                                        d_14_cc_ = out7_
                                        generated = d_12_cg_
                                        insideConstrainedOut = d_13_ci_
                                        currentConstrainedOut = d_14_cc_
                                        d_1_steps_ = (d_1_steps_) + (1)
                                    raise _dafny.Break("0")
                                elif True:
                                    d_15_ag_: _dafny.Seq
                                    d_16_ai_: bool
                                    d_17_ac_: _dafny.Seq
                                    out8_: _dafny.Seq
                                    out9_: bool
                                    out10_: _dafny.Seq
                                    out8_, out9_, out10_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_11_next_)
                                    d_15_ag_ = out8_
                                    d_16_ai_ = out9_
                                    d_17_ac_ = out10_
                                    generated = d_15_ag_
                                    insideConstrainedOut = d_16_ai_
                                    currentConstrainedOut = d_17_ac_
                            elif True:
                                if (d_1_steps_) < (maxSteps):
                                    d_18_cg_: _dafny.Seq
                                    d_19_ci_: bool
                                    d_20_cc_: _dafny.Seq
                                    out11_: _dafny.Seq
                                    out12_: bool
                                    out13_: _dafny.Seq
                                    out11_, out12_, out13_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_18_cg_ = out11_
                                    d_19_ci_ = out12_
                                    d_20_cc_ = out13_
                                    generated = d_18_cg_
                                    insideConstrainedOut = d_19_ci_
                                    currentConstrainedOut = d_20_cc_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                elif True:
                                    raise _dafny.Break("0")
                        elif True:
                            d_21_constrainedPrompt_: _dafny.Seq
                            d_21_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_22_next_: _dafny.Seq
                            out14_: _dafny.Seq
                            out14_ = (d_0_helpers_).SafeTemperatureConstrainedStep(lm, parser, d_21_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('16e-1'), eosToken)
                            d_22_next_ = out14_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_22_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_23_ag_: _dafny.Seq
                                d_24_ai_: bool
                                d_25_ac_: _dafny.Seq
                                out15_: _dafny.Seq
                                out16_: bool
                                out17_: _dafny.Seq
                                out15_, out16_, out17_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_22_next_)
                                d_23_ag_ = out15_
                                d_24_ai_ = out16_
                                d_25_ac_ = out17_
                                generated = d_23_ag_
                                insideConstrainedOut = d_24_ai_
                                currentConstrainedOut = d_25_ac_
                    elif True:
                        d_26_constrainedPrompt_: _dafny.Seq
                        d_26_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_27_next_: _dafny.Seq
                        out18_: _dafny.Seq
                        out18_ = (d_0_helpers_).SafeTemperatureConstrainedStep(lm, parser, d_26_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('16e-1'), eosToken)
                        d_27_next_ = out18_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_27_next_) == (eosToken):
                            if (((parser).IsCompletePrefix(currentConstrainedOut)) and ((len(currentConstrainedOut)) >= (d_5_minSpanLength_))) and ((d_1_steps_) < (maxSteps)):
                                d_28_smilesStr_: _dafny.Seq
                                d_28_smilesStr_ = VerifiedDecoderAgent.CSDHelpers.PrefixToString(currentConstrainedOut)
                                d_29_hasNCO_: int
                                d_29_hasNCO_ = VerifiedDecoderAgent.CSDHelpers.CountSubstring(d_28_smilesStr_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "N=C=O")))
                                if (d_29_hasNCO_) > (0):
                                    d_30_cg_: _dafny.Seq
                                    d_31_ci_: bool
                                    d_32_cc_: _dafny.Seq
                                    out19_: _dafny.Seq
                                    out20_: bool
                                    out21_: _dafny.Seq
                                    out19_, out20_, out21_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_30_cg_ = out19_
                                    d_31_ci_ = out20_
                                    d_32_cc_ = out21_
                                    generated = d_30_cg_
                                    insideConstrainedOut = d_31_ci_
                                    currentConstrainedOut = d_32_cc_
                                    d_1_steps_ = (d_1_steps_) + (1)
                            raise _dafny.Break("0")
                        elif True:
                            d_33_ag_: _dafny.Seq
                            d_34_ai_: bool
                            d_35_ac_: _dafny.Seq
                            out22_: _dafny.Seq
                            out23_: bool
                            out24_: _dafny.Seq
                            out22_, out23_, out24_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_27_next_)
                            d_33_ag_ = out22_
                            d_34_ai_ = out23_
                            d_35_ac_ = out24_
                            generated = d_33_ag_
                            insideConstrainedOut = d_34_ai_
                            currentConstrainedOut = d_35_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_36_closeBudget_: int
            d_36_closeBudget_ = (maxSteps) - (d_1_steps_)
            d_37_cg_: _dafny.Seq
            d_38_ci_: bool
            d_39_cc_: _dafny.Seq
            out25_: _dafny.Seq
            out26_: bool
            out27_: _dafny.Seq
            out25_, out26_, out27_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_36_closeBudget_)
            d_37_cg_ = out25_
            d_38_ci_ = out26_
            d_39_cc_ = out27_
            generated = d_37_cg_
            insideConstrainedOut = d_38_ci_
            currentConstrainedOut = d_39_cc_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

