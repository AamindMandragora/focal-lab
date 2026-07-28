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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate ONE valid SMILES for an acrylate ester molecule. The acrylate core substructure is C=CC(=O)O (acryloyloxy group). Diverse novel examples: C=CC(=O)OCCC, C=CC(=O)OCC(C)C, C=CC(=O)OCCO, C=CC(=O)OC(C)(C)C, C=CC(=O)OCC(F)(F)F, C=CC(=O)OCCN(C)C, C=CC(=O)OC1CCCCC1, C=CC(=O)OCCOCCO, C=CC(=O)OCC(O)CO, C=CC(=O)OCCSCC, C=CC(=O)OCCCOC, C=CC(=O)OCC1CC1. Output a NOVEL acrylate SMILES only."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_minLength_: int
        d_3_minLength_ = 9
        d_4_narrowThreshold_: int
        d_4_narrowThreshold_ = 20
        if (not(insideConstrainedOut)) and ((d_2_steps_) < (maxSteps)):
            d_5_og_: _dafny.Seq
            d_6_oi_: bool
            d_7_oc_: _dafny.Seq
            out0_: _dafny.Seq
            out1_: bool
            out2_: _dafny.Seq
            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_5_og_ = out0_
            d_6_oi_ = out1_
            d_7_oc_ = out2_
            generated = d_5_og_
            insideConstrainedOut = d_6_oi_
            currentConstrainedOut = d_7_oc_
            d_2_steps_ = (d_2_steps_) + (1)
        d_8_coreTokens_: _dafny.Seq
        d_8_coreTokens_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "C")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "C")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "C")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "(")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "O")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ")")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "O"))])
        d_9_coreIdx_: int
        d_9_coreIdx_ = 0
        with _dafny.label("0"):
            while (((d_9_coreIdx_) < (len(d_8_coreTokens_))) and ((d_2_steps_) < (maxSteps))) and (insideConstrainedOut):
                with _dafny.c_label("0"):
                    d_10_coreToken_: _dafny.Seq
                    d_10_coreToken_ = (d_8_coreTokens_)[d_9_coreIdx_]
                    d_11_isValid_: bool
                    out3_: bool
                    out3_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_10_coreToken_)
                    d_11_isValid_ = out3_
                    d_12_constrainedPrompt_: _dafny.Seq
                    d_12_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                    if d_11_isValid_:
                        (d_0_helpers_).SafeBoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([d_10_coreToken_]), _dafny.BigRational('8e0'))
                        d_13_next_: _dafny.Seq
                        out4_: _dafny.Seq
                        out4_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_12_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('8e0'), d_4_narrowThreshold_, eosToken)
                        d_13_next_ = out4_
                        d_2_steps_ = (d_2_steps_) + (1)
                        if (d_13_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_14_ag_: _dafny.Seq
                            d_15_ai_: bool
                            d_16_ac_: _dafny.Seq
                            out5_: _dafny.Seq
                            out6_: bool
                            out7_: _dafny.Seq
                            out5_, out6_, out7_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_13_next_)
                            d_14_ag_ = out5_
                            d_15_ai_ = out6_
                            d_16_ac_ = out7_
                            generated = d_14_ag_
                            insideConstrainedOut = d_15_ai_
                            currentConstrainedOut = d_16_ac_
                            if (d_13_next_) == (d_10_coreToken_):
                                d_9_coreIdx_ = (d_9_coreIdx_) + (1)
                            elif True:
                                d_9_coreIdx_ = len(d_8_coreTokens_)
                    elif True:
                        d_17_next_: _dafny.Seq
                        out8_: _dafny.Seq
                        out8_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_12_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_4_narrowThreshold_, eosToken)
                        d_17_next_ = out8_
                        d_2_steps_ = (d_2_steps_) + (1)
                        if (d_17_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_18_ag_: _dafny.Seq
                            d_19_ai_: bool
                            d_20_ac_: _dafny.Seq
                            out9_: _dafny.Seq
                            out10_: bool
                            out11_: _dafny.Seq
                            out9_, out10_, out11_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_17_next_)
                            d_18_ag_ = out9_
                            d_19_ai_ = out10_
                            d_20_ac_ = out11_
                            generated = d_18_ag_
                            insideConstrainedOut = d_19_ai_
                            currentConstrainedOut = d_20_ac_
                            d_9_coreIdx_ = len(d_8_coreTokens_)
                    pass
            pass
        with _dafny.label("1"):
            while (d_2_steps_) < (maxSteps):
                with _dafny.c_label("1"):
                    if not(insideConstrainedOut):
                        raise _dafny.Break("1")
                    elif ((parser).IsCompletePrefix(currentConstrainedOut)) and ((len(currentConstrainedOut)) >= (d_3_minLength_)):
                        d_21_cg_: _dafny.Seq
                        d_22_ci_: bool
                        d_23_cc_: _dafny.Seq
                        out12_: _dafny.Seq
                        out13_: bool
                        out14_: _dafny.Seq
                        out12_, out13_, out14_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_21_cg_ = out12_
                        d_22_ci_ = out13_
                        d_23_cc_ = out14_
                        generated = d_21_cg_
                        insideConstrainedOut = d_22_ci_
                        currentConstrainedOut = d_23_cc_
                        d_2_steps_ = (d_2_steps_) + (1)
                        raise _dafny.Break("1")
                    elif True:
                        d_24_constrainedPrompt_: _dafny.Seq
                        d_24_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_25_next_: _dafny.Seq
                        d_25_next_ = eosToken
                        d_26_curLen_: int
                        d_26_curLen_ = len(currentConstrainedOut)
                        if (d_26_curLen_) < (12):
                            out15_: _dafny.Seq
                            out15_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_24_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_4_narrowThreshold_, eosToken)
                            d_25_next_ = out15_
                        elif (_dafny.euclidian_modulus(d_2_steps_, 3)) == (0):
                            out16_: _dafny.Seq
                            out16_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_24_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('2e0'), eosToken)
                            d_25_next_ = out16_
                        elif True:
                            out17_: _dafny.Seq
                            out17_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_24_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('3e0'), d_4_narrowThreshold_, eosToken)
                            d_25_next_ = out17_
                        d_2_steps_ = (d_2_steps_) + (1)
                        if (d_25_next_) == (eosToken):
                            if (((parser).IsCompletePrefix(currentConstrainedOut)) and ((len(currentConstrainedOut)) >= (d_3_minLength_))) and ((d_2_steps_) < (maxSteps)):
                                d_27_cg_: _dafny.Seq
                                d_28_ci_: bool
                                d_29_cc_: _dafny.Seq
                                out18_: _dafny.Seq
                                out19_: bool
                                out20_: _dafny.Seq
                                out18_, out19_, out20_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_27_cg_ = out18_
                                d_28_ci_ = out19_
                                d_29_cc_ = out20_
                                generated = d_27_cg_
                                insideConstrainedOut = d_28_ci_
                                currentConstrainedOut = d_29_cc_
                                d_2_steps_ = (d_2_steps_) + (1)
                            raise _dafny.Break("1")
                        elif True:
                            d_30_ag_: _dafny.Seq
                            d_31_ai_: bool
                            d_32_ac_: _dafny.Seq
                            out21_: _dafny.Seq
                            out22_: bool
                            out23_: _dafny.Seq
                            out21_, out22_, out23_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_25_next_)
                            d_30_ag_ = out21_
                            d_31_ai_ = out22_
                            d_32_ac_ = out23_
                            generated = d_30_ag_
                            insideConstrainedOut = d_31_ai_
                            currentConstrainedOut = d_32_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_33_closeBudget_: int
            d_33_closeBudget_ = (maxSteps) - (d_2_steps_)
            d_34_cg_: _dafny.Seq
            d_35_ci_: bool
            d_36_cc_: _dafny.Seq
            out24_: _dafny.Seq
            out25_: bool
            out26_: _dafny.Seq
            out24_, out25_, out26_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_33_closeBudget_)
            d_34_cg_ = out24_
            d_35_ci_ = out25_
            d_36_cc_ = out26_
            generated = d_34_cg_
            insideConstrainedOut = d_35_ci_
            currentConstrainedOut = d_36_cc_
            d_2_steps_ = maxSteps
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

