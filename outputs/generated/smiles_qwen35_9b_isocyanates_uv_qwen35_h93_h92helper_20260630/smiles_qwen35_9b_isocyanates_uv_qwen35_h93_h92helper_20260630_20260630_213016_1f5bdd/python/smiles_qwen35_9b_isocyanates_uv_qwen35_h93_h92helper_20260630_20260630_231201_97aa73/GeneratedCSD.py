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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "You are generating a SMILES string for an ISOCYANATE molecule. Isocyanates MUST contain the N=C=O functional group. The SMILES string MUST include 'N=C=O' as a substring. Start with an organic group then N=C=O. Examples of valid isocyanate SMILES: CCN=C=O, CCCN=C=O, CN=C=O, CC(C)N=C=O, c1ccccc1N=C=O, CCCCN=C=O, CC(CC)N=C=O. Your output MUST contain N=C=O. Do NOT generate simple atoms like just 'C' or 'N'. Generate a complete isocyanate SMILES with the N=C=O group.")))
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
        with _dafny.label("0"):
            while ((d_1_steps_) < (maxSteps)) and (insideConstrainedOut):
                with _dafny.c_label("0"):
                    d_5_currentStr_: _dafny.Seq
                    d_5_currentStr_ = VerifiedDecoderAgent.CSDHelpers.PrefixToString(currentConstrainedOut)
                    d_6_hasIsocyanate_: bool
                    d_6_hasIsocyanate_ = (VerifiedDecoderAgent.CSDHelpers.CountSubstring(d_5_currentStr_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "N=C=O")))) > (0)
                    if ((parser).IsCompletePrefix(currentConstrainedOut)) and (d_6_hasIsocyanate_):
                        d_7_cg_: _dafny.Seq
                        d_8_ci_: bool
                        d_9_cc_: _dafny.Seq
                        out3_: _dafny.Seq
                        out4_: bool
                        out5_: _dafny.Seq
                        out3_, out4_, out5_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_7_cg_ = out3_
                        d_8_ci_ = out4_
                        d_9_cc_ = out5_
                        generated = d_7_cg_
                        insideConstrainedOut = d_8_ci_
                        currentConstrainedOut = d_9_cc_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif ((parser).IsCompletePrefix(currentConstrainedOut)) and (not(d_6_hasIsocyanate_)):
                        d_10_closeBudget_: int
                        d_10_closeBudget_ = (maxSteps) - (d_1_steps_)
                        if (d_10_closeBudget_) > (0):
                            d_11_cg_: _dafny.Seq
                            d_12_ci_: bool
                            d_13_cc_: _dafny.Seq
                            out6_: _dafny.Seq
                            out7_: bool
                            out8_: _dafny.Seq
                            out6_, out7_, out8_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_10_closeBudget_)
                            d_11_cg_ = out6_
                            d_12_ci_ = out7_
                            d_13_cc_ = out8_
                            generated = d_11_cg_
                            insideConstrainedOut = d_12_ci_
                            currentConstrainedOut = d_13_cc_
                            d_1_steps_ = maxSteps
                        elif True:
                            d_14_cg_: _dafny.Seq
                            d_15_ci_: bool
                            d_16_cc_: _dafny.Seq
                            out9_: _dafny.Seq
                            out10_: bool
                            out11_: _dafny.Seq
                            out9_, out10_, out11_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_14_cg_ = out9_
                            d_15_ci_ = out10_
                            d_16_cc_ = out11_
                            generated = d_14_cg_
                            insideConstrainedOut = d_15_ci_
                            currentConstrainedOut = d_16_cc_
                            d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_17_constrainedPrompt_: _dafny.Seq
                        d_17_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_18_nTok_: _dafny.Seq
                        d_18_nTok_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "N"))
                        d_19_eqTok_: _dafny.Seq
                        d_19_eqTok_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "="))
                        d_20_cTok_: _dafny.Seq
                        d_20_cTok_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "C"))
                        d_21_oTok_: _dafny.Seq
                        d_21_oTok_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "O"))
                        d_22_nValid_: bool
                        out12_: bool
                        out12_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_18_nTok_)
                        d_22_nValid_ = out12_
                        d_23_nInVocab_: bool
                        d_23_nInVocab_ = (d_18_nTok_) in ((lm).Tokens)
                        if (d_22_nValid_) and (d_23_nInVocab_):
                            (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([d_18_nTok_]), _dafny.BigRational('3e0'))
                        d_24_next_: _dafny.Seq
                        out13_: _dafny.Seq
                        out13_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_17_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('15e-1'), eosToken)
                        d_24_next_ = out13_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_24_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_25_valid_: bool
                            out14_: bool
                            out14_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_24_next_)
                            d_25_valid_ = out14_
                            if (d_25_valid_) and (not((parser).IsCompletePrefix(currentConstrainedOut))):
                                d_26_ag_: _dafny.Seq
                                d_27_ai_: bool
                                d_28_ac_: _dafny.Seq
                                out15_: _dafny.Seq
                                out16_: bool
                                out17_: _dafny.Seq
                                out15_, out16_, out17_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_24_next_)
                                d_26_ag_ = out15_
                                d_27_ai_ = out16_
                                d_28_ac_ = out17_
                                generated = d_26_ag_
                                insideConstrainedOut = d_27_ai_
                                currentConstrainedOut = d_28_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_29_closeBudget_: int
            d_29_closeBudget_ = (maxSteps) - (d_1_steps_)
            d_30_cg_: _dafny.Seq
            d_31_ci_: bool
            d_32_cc_: _dafny.Seq
            out18_: _dafny.Seq
            out19_: bool
            out20_: _dafny.Seq
            out18_, out19_, out20_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_29_closeBudget_)
            d_30_cg_ = out18_
            d_31_ci_ = out19_
            d_32_cc_ = out20_
            generated = d_30_cg_
            insideConstrainedOut = d_31_ci_
            currentConstrainedOut = d_32_cc_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

