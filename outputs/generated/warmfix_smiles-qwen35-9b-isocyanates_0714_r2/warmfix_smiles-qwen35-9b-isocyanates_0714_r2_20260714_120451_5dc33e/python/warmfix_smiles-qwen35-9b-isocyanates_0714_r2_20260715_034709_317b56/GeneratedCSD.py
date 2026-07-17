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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate exactly one valid SMILES for a NEW isocyanate molecule. Isocyanates contain N=C=O. Output ONLY the SMILES string. Generate DIVERSE molecules - pick different structural classes each time: (A) alkyl: O=C=NC(C)(C)C, O=C=NCCC, O=C=NCC(C)C; (B) cycloalkyl: O=C=NC1CCCCC1, O=C=NC1CCCC1, O=C=NC1CCC1; (C) aryl: O=C=Nc1ccccc1, O=C=Nc1ccc(Cl)cc1, O=C=Nc1ccccn1; (D) benzyl: O=C=NCc1ccccc1, O=C=NCCc1ccccc1; (E) haloalkyl: O=C=NCCC(F)(F)F, O=C=NCCCl, O=C=NCC(F)F; (F) alkenyl: O=C=NCC=C, O=C=NC(C)=C; (G) alkynyl: O=C=NCC#C. Pick one class and generate a valid example.")))
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
        d_6_maxSpanLength_: int
        d_6_maxSpanLength_ = 35
        d_7_tokenCount_: int
        d_7_tokenCount_ = 0
        d_8_earlyPenaltyTokens_: _dafny.Seq
        d_8_earlyPenaltyTokens_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "C")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "CC"))])
        with _dafny.label("0"):
            while ((d_1_steps_) < (maxSteps)) and (insideConstrainedOut):
                with _dafny.c_label("0"):
                    if ((len(currentConstrainedOut)) >= (d_6_maxSpanLength_)) and ((d_1_steps_) < (maxSteps)):
                        d_9_closeBudget_: int
                        d_9_closeBudget_ = (maxSteps) - (d_1_steps_)
                        d_10_cg_: _dafny.Seq
                        d_11_ci_: bool
                        d_12_cc_: _dafny.Seq
                        out3_: _dafny.Seq
                        out4_: bool
                        out5_: _dafny.Seq
                        out3_, out4_, out5_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_9_closeBudget_)
                        d_10_cg_ = out3_
                        d_11_ci_ = out4_
                        d_12_cc_ = out5_
                        generated = d_10_cg_
                        insideConstrainedOut = d_11_ci_
                        currentConstrainedOut = d_12_cc_
                        d_1_steps_ = maxSteps
                        raise _dafny.Break("0")
                    elif ((parser).IsCompletePrefix(currentConstrainedOut)) and ((len(currentConstrainedOut)) >= (d_5_minSpanLength_)):
                        d_13_smilesStr_: _dafny.Seq
                        d_13_smilesStr_ = VerifiedDecoderAgent.CSDHelpers.PrefixToString(currentConstrainedOut)
                        d_14_hasNCO_: int
                        d_14_hasNCO_ = VerifiedDecoderAgent.CSDHelpers.CountSubstring(d_13_smilesStr_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "N=C=O")))
                        if (d_14_hasNCO_) > (0):
                            if (d_1_steps_) < (maxSteps):
                                d_15_cg_: _dafny.Seq
                                d_16_ci_: bool
                                d_17_cc_: _dafny.Seq
                                out6_: _dafny.Seq
                                out7_: bool
                                out8_: _dafny.Seq
                                out6_, out7_, out8_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_15_cg_ = out6_
                                d_16_ci_ = out7_
                                d_17_cc_ = out8_
                                generated = d_15_cg_
                                insideConstrainedOut = d_16_ci_
                                currentConstrainedOut = d_17_cc_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                raise _dafny.Break("0")
                        elif True:
                            d_18_constrainedPrompt_: _dafny.Seq
                            d_18_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_19_next_: _dafny.Seq
                            out9_: _dafny.Seq
                            out9_ = (d_0_helpers_).SafeTemperatureConstrainedStep(lm, parser, d_18_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('14e-1'), eosToken)
                            d_19_next_ = out9_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_19_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_20_ag_: _dafny.Seq
                                d_21_ai_: bool
                                d_22_ac_: _dafny.Seq
                                out10_: _dafny.Seq
                                out11_: bool
                                out12_: _dafny.Seq
                                out10_, out11_, out12_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_19_next_)
                                d_20_ag_ = out10_
                                d_21_ai_ = out11_
                                d_22_ac_ = out12_
                                generated = d_20_ag_
                                insideConstrainedOut = d_21_ai_
                                currentConstrainedOut = d_22_ac_
                                d_7_tokenCount_ = (d_7_tokenCount_) + (1)
                    elif True:
                        d_23_constrainedPrompt_: _dafny.Seq
                        d_23_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_24_next_: _dafny.Seq
                        d_24_next_ = eosToken
                        if (d_7_tokenCount_) < (5):
                            out13_: _dafny.Seq
                            out13_ = (d_0_helpers_).SafePenalizedConstrainedStep(lm, parser, d_23_constrainedPrompt_, currentConstrainedOut, d_8_earlyPenaltyTokens_, _dafny.BigRational('5e0'), eosToken)
                            d_24_next_ = out13_
                        elif (d_7_tokenCount_) < (15):
                            out14_: _dafny.Seq
                            out14_ = (d_0_helpers_).SafeTemperatureConstrainedStep(lm, parser, d_23_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('14e-1'), eosToken)
                            d_24_next_ = out14_
                        elif (d_7_tokenCount_) < (25):
                            out15_: _dafny.Seq
                            out15_ = (d_0_helpers_).SafeTemperatureConstrainedStep(lm, parser, d_23_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('12e-1'), eosToken)
                            d_24_next_ = out15_
                        elif True:
                            out16_: _dafny.Seq
                            out16_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, d_23_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), eosToken)
                            d_24_next_ = out16_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_24_next_) == (eosToken):
                            if (((parser).IsCompletePrefix(currentConstrainedOut)) and ((len(currentConstrainedOut)) >= (d_5_minSpanLength_))) and ((d_1_steps_) < (maxSteps)):
                                d_25_smilesStr_: _dafny.Seq
                                d_25_smilesStr_ = VerifiedDecoderAgent.CSDHelpers.PrefixToString(currentConstrainedOut)
                                d_26_hasNCO_: int
                                d_26_hasNCO_ = VerifiedDecoderAgent.CSDHelpers.CountSubstring(d_25_smilesStr_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "N=C=O")))
                                if (d_26_hasNCO_) > (0):
                                    d_27_cg_: _dafny.Seq
                                    d_28_ci_: bool
                                    d_29_cc_: _dafny.Seq
                                    out17_: _dafny.Seq
                                    out18_: bool
                                    out19_: _dafny.Seq
                                    out17_, out18_, out19_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_27_cg_ = out17_
                                    d_28_ci_ = out18_
                                    d_29_cc_ = out19_
                                    generated = d_27_cg_
                                    insideConstrainedOut = d_28_ci_
                                    currentConstrainedOut = d_29_cc_
                                    d_1_steps_ = (d_1_steps_) + (1)
                            raise _dafny.Break("0")
                        elif True:
                            d_30_ag_: _dafny.Seq
                            d_31_ai_: bool
                            d_32_ac_: _dafny.Seq
                            out20_: _dafny.Seq
                            out21_: bool
                            out22_: _dafny.Seq
                            out20_, out21_, out22_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_24_next_)
                            d_30_ag_ = out20_
                            d_31_ai_ = out21_
                            d_32_ac_ = out22_
                            generated = d_30_ag_
                            insideConstrainedOut = d_31_ai_
                            currentConstrainedOut = d_32_ac_
                            d_7_tokenCount_ = (d_7_tokenCount_) + (1)
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_33_closeBudget_: int
            d_33_closeBudget_ = (maxSteps) - (d_1_steps_)
            d_34_cg_: _dafny.Seq
            d_35_ci_: bool
            d_36_cc_: _dafny.Seq
            out23_: _dafny.Seq
            out24_: bool
            out25_: _dafny.Seq
            out23_, out24_, out25_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_33_closeBudget_)
            d_34_cg_ = out23_
            d_35_ci_ = out24_
            d_36_cc_ = out25_
            generated = d_34_cg_
            insideConstrainedOut = d_35_ci_
            currentConstrainedOut = d_36_cc_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

