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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate ONE valid SMILES for an isocyanate (R-N=C=O). Required: contains N=C=O. Be maximally diverse - pick a DIFFERENT structural class each time from: methyl=O=C=NC, ethyl=O=C=NCC, propyl=O=C=NCCC, isopropyl=O=C=NC(C)C, butyl=O=C=NCCCC, isobutyl=O=C=NCC(C)C, tert-butyl=O=C=NC(C)(C)C, pentyl=O=C=NCCCCC, hexyl=O=C=NCCCCCC, cyclopropyl=O=C=NC1CC1, cyclobutyl=O=C=NC1CCC1, cyclopentyl=O=C=NC1CCCC1, cyclohexyl=O=C=NC1CCCCC1, phenyl=O=C=Nc1ccccc1, benzyl=O=C=NCc1ccccc1, 4-methylphenyl=O=C=Nc1ccc(C)cc1, 4-chlorophenyl=O=C=Nc1ccc(Cl)cc1, 4-fluorophenyl=O=C=Nc1ccc(F)cc1, 4-methoxyphenyl=O=C=Nc1ccc(OC)cc1, naphthyl=O=C=Nc1ccc2ccccc2c1, allyl=O=C=NCC=C, propargyl=O=C=NCC#C, 2-chloroethyl=O=C=NCCCl, 2-fluoroethyl=O=C=NCCF, trifluoromethyl=O=C=NCC(F)(F)F, pyridyl=O=C=Nc1ccccn1, furanyl=O=C=Nc1ccco1, thienyl=O=C=Nc1cccs1, adamantyl=O=C=NC12CC(CC(C1)C2), diisocyanate=O=C=NCCN=C=O. Output ONLY the SMILES.")))
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
        d_6_minSmilesChars_: int
        d_6_minSmilesChars_ = 7
        d_7_tokenCount_: int
        d_7_tokenCount_ = 0
        d_8_earlyPenaltyTokens_: _dafny.Seq
        d_8_earlyPenaltyTokens_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "C")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "O")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "CC"))])
        with _dafny.label("0"):
            while ((d_1_steps_) < (maxSteps)) and (insideConstrainedOut):
                with _dafny.c_label("0"):
                    if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((len(currentConstrainedOut)) >= (d_5_minSpanLength_)):
                        d_9_smilesStr_: _dafny.Seq
                        d_9_smilesStr_ = VerifiedDecoderAgent.CSDHelpers.PrefixToString(currentConstrainedOut)
                        d_10_hasNCO_: int
                        d_10_hasNCO_ = VerifiedDecoderAgent.CSDHelpers.CountSubstring(d_9_smilesStr_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "N=C=O")))
                        d_11_smilesLen_: int
                        d_11_smilesLen_ = len(d_9_smilesStr_)
                        if ((d_10_hasNCO_) > (0)) and ((d_11_smilesLen_) >= (d_6_minSmilesChars_)):
                            if (d_1_steps_) < (maxSteps):
                                d_12_cg_: _dafny.Seq
                                d_13_ci_: bool
                                d_14_cc_: _dafny.Seq
                                out3_: _dafny.Seq
                                out4_: bool
                                out5_: _dafny.Seq
                                out3_, out4_, out5_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_12_cg_ = out3_
                                d_13_ci_ = out4_
                                d_14_cc_ = out5_
                                generated = d_12_cg_
                                insideConstrainedOut = d_13_ci_
                                currentConstrainedOut = d_14_cc_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                raise _dafny.Break("0")
                        elif True:
                            d_15_constrainedPrompt_: _dafny.Seq
                            d_15_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_16_next_: _dafny.Seq
                            out6_: _dafny.Seq
                            out6_ = (d_0_helpers_).SafeTemperatureConstrainedStep(lm, parser, d_15_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('15e-1'), eosToken)
                            d_16_next_ = out6_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_16_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_17_ag_: _dafny.Seq
                                d_18_ai_: bool
                                d_19_ac_: _dafny.Seq
                                out7_: _dafny.Seq
                                out8_: bool
                                out9_: _dafny.Seq
                                out7_, out8_, out9_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_16_next_)
                                d_17_ag_ = out7_
                                d_18_ai_ = out8_
                                d_19_ac_ = out9_
                                generated = d_17_ag_
                                insideConstrainedOut = d_18_ai_
                                currentConstrainedOut = d_19_ac_
                                d_7_tokenCount_ = (d_7_tokenCount_) + (1)
                    elif True:
                        d_20_constrainedPrompt_: _dafny.Seq
                        d_20_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_21_next_: _dafny.Seq
                        d_21_next_ = eosToken
                        if (d_7_tokenCount_) < (4):
                            out10_: _dafny.Seq
                            out10_ = (d_0_helpers_).SafePenalizedConstrainedStep(lm, parser, d_20_constrainedPrompt_, currentConstrainedOut, d_8_earlyPenaltyTokens_, _dafny.BigRational('5e0'), eosToken)
                            d_21_next_ = out10_
                        elif (d_7_tokenCount_) < (8):
                            out11_: _dafny.Seq
                            out11_ = (d_0_helpers_).SafeTemperatureConstrainedStep(lm, parser, d_20_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('18e-1'), eosToken)
                            d_21_next_ = out11_
                        elif (d_7_tokenCount_) < (20):
                            out12_: _dafny.Seq
                            out12_ = (d_0_helpers_).SafeTemperatureConstrainedStep(lm, parser, d_20_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('17e-1'), eosToken)
                            d_21_next_ = out12_
                        elif True:
                            out13_: _dafny.Seq
                            out13_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, d_20_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), eosToken)
                            d_21_next_ = out13_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_21_next_) == (eosToken):
                            if (((parser).IsCompletePrefix(currentConstrainedOut)) and ((len(currentConstrainedOut)) >= (d_5_minSpanLength_))) and ((d_1_steps_) < (maxSteps)):
                                d_22_smilesStr_: _dafny.Seq
                                d_22_smilesStr_ = VerifiedDecoderAgent.CSDHelpers.PrefixToString(currentConstrainedOut)
                                d_23_hasNCO_: int
                                d_23_hasNCO_ = VerifiedDecoderAgent.CSDHelpers.CountSubstring(d_22_smilesStr_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "N=C=O")))
                                d_24_smilesLen_: int
                                d_24_smilesLen_ = len(d_22_smilesStr_)
                                if ((d_23_hasNCO_) > (0)) and ((d_24_smilesLen_) >= (d_6_minSmilesChars_)):
                                    d_25_cg_: _dafny.Seq
                                    d_26_ci_: bool
                                    d_27_cc_: _dafny.Seq
                                    out14_: _dafny.Seq
                                    out15_: bool
                                    out16_: _dafny.Seq
                                    out14_, out15_, out16_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_25_cg_ = out14_
                                    d_26_ci_ = out15_
                                    d_27_cc_ = out16_
                                    generated = d_25_cg_
                                    insideConstrainedOut = d_26_ci_
                                    currentConstrainedOut = d_27_cc_
                                    d_1_steps_ = (d_1_steps_) + (1)
                            raise _dafny.Break("0")
                        elif True:
                            d_28_ag_: _dafny.Seq
                            d_29_ai_: bool
                            d_30_ac_: _dafny.Seq
                            out17_: _dafny.Seq
                            out18_: bool
                            out19_: _dafny.Seq
                            out17_, out18_, out19_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_21_next_)
                            d_28_ag_ = out17_
                            d_29_ai_ = out18_
                            d_30_ac_ = out19_
                            generated = d_28_ag_
                            insideConstrainedOut = d_29_ai_
                            currentConstrainedOut = d_30_ac_
                            d_7_tokenCount_ = (d_7_tokenCount_) + (1)
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_31_closeBudget_: int
            d_31_closeBudget_ = (maxSteps) - (d_1_steps_)
            d_32_cg_: _dafny.Seq
            d_33_ci_: bool
            d_34_cc_: _dafny.Seq
            out20_: _dafny.Seq
            out21_: bool
            out22_: _dafny.Seq
            out20_, out21_, out22_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_31_closeBudget_)
            d_32_cg_ = out20_
            d_33_ci_ = out21_
            d_34_cc_ = out22_
            generated = d_32_cg_
            insideConstrainedOut = d_33_ci_
            currentConstrainedOut = d_34_cc_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

