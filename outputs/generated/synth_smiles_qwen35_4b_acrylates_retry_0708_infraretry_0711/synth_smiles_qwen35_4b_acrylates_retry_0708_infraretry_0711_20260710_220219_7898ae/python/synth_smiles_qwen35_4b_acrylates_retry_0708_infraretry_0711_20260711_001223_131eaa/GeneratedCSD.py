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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate ONE valid SMILES for an acrylate compound. An acrylate MUST contain C=CC(=O)O (acryloyl ester). Diverse examples: C=CC(=O)OCCC, C=CC(=O)OCC(C)C, C=CC(=O)OCCO, C=CC(=O)OC(C)(C)C, C=CC(=O)OCC(F)(F)F, C=CC(=O)OCCN(C)C, C=CC(=O)OC1CCCCC1, C=CC(=O)OCCOCCO, CC(=C)C(=O)OCC, C=CC(=O)OCC(O)CO, C=CC(=O)OCCSCC. Generate a NOVEL acrylate not in the list. Output SMILES only."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_minLength_: int
        d_3_minLength_ = 12
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
        with _dafny.label("0"):
            while (d_2_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        raise _dafny.Break("0")
                    elif ((parser).IsCompletePrefix(currentConstrainedOut)) and ((len(currentConstrainedOut)) >= (d_3_minLength_)):
                        d_8_cg_: _dafny.Seq
                        d_9_ci_: bool
                        d_10_cc_: _dafny.Seq
                        out3_: _dafny.Seq
                        out4_: bool
                        out5_: _dafny.Seq
                        out3_, out4_, out5_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_8_cg_ = out3_
                        d_9_ci_ = out4_
                        d_10_cc_ = out5_
                        generated = d_8_cg_
                        insideConstrainedOut = d_9_ci_
                        currentConstrainedOut = d_10_cc_
                        d_2_steps_ = (d_2_steps_) + (1)
                        raise _dafny.Break("0")
                    elif True:
                        d_11_constrainedPrompt_: _dafny.Seq
                        d_11_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_12_curLen_: int
                        d_12_curLen_ = len(currentConstrainedOut)
                        d_13_next_: _dafny.Seq
                        d_13_next_ = eosToken
                        if (d_12_curLen_) < (6):
                            d_14_coreTokens_: _dafny.Seq
                            d_14_coreTokens_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "O")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "C")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "(")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ")"))])
                            (d_0_helpers_).SafeBoostTokenLogits(lm, d_14_coreTokens_, _dafny.BigRational('3e0'))
                            out6_: _dafny.Seq
                            out6_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_11_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('8e0'), d_4_narrowThreshold_, eosToken)
                            d_13_next_ = out6_
                        elif (d_12_curLen_) < (20):
                            d_15_softNext_: _dafny.Seq
                            d_16_softOk_: bool
                            out7_: _dafny.Seq
                            out8_: bool
                            out7_, out8_ = (d_0_helpers_).SoftConstrainedStep(lm, parser, d_11_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('12e0'), eosToken)
                            d_15_softNext_ = out7_
                            d_16_softOk_ = out8_
                            if d_16_softOk_:
                                d_13_next_ = d_15_softNext_
                            elif True:
                                out9_: _dafny.Seq
                                out9_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_11_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_4_narrowThreshold_, eosToken)
                                d_13_next_ = out9_
                        elif (d_12_curLen_) < (35):
                            if (_dafny.euclidian_modulus(d_12_curLen_, 3)) == (0):
                                out10_: _dafny.Seq
                                out10_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_11_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('2e0'), eosToken)
                                d_13_next_ = out10_
                            elif True:
                                d_17_softNext2_: _dafny.Seq
                                d_18_softOk2_: bool
                                out11_: _dafny.Seq
                                out12_: bool
                                out11_, out12_ = (d_0_helpers_).SoftConstrainedStep(lm, parser, d_11_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('8e0'), eosToken)
                                d_17_softNext2_ = out11_
                                d_18_softOk2_ = out12_
                                if d_18_softOk2_:
                                    d_13_next_ = d_17_softNext2_
                                elif True:
                                    out13_: _dafny.Seq
                                    out13_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_11_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_4_narrowThreshold_, eosToken)
                                    d_13_next_ = out13_
                        elif True:
                            out14_: _dafny.Seq
                            out14_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_11_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_4_narrowThreshold_, eosToken)
                            d_13_next_ = out14_
                        d_2_steps_ = (d_2_steps_) + (1)
                        if (d_13_next_) == (eosToken):
                            if insideConstrainedOut:
                                d_19_rg_: _dafny.Seq
                                d_20_rc_: _dafny.Seq
                                out15_: _dafny.Seq
                                out16_: _dafny.Seq
                                out15_, out16_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                                d_19_rg_ = out15_
                                d_20_rc_ = out16_
                                generated = d_19_rg_
                                currentConstrainedOut = d_20_rc_
                                if (((parser).IsCompletePrefix(currentConstrainedOut)) and ((len(currentConstrainedOut)) >= (d_3_minLength_))) and ((d_2_steps_) < (maxSteps)):
                                    d_21_cg_: _dafny.Seq
                                    d_22_ci_: bool
                                    d_23_cc_: _dafny.Seq
                                    out17_: _dafny.Seq
                                    out18_: bool
                                    out19_: _dafny.Seq
                                    out17_, out18_, out19_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_21_cg_ = out17_
                                    d_22_ci_ = out18_
                                    d_23_cc_ = out19_
                                    generated = d_21_cg_
                                    insideConstrainedOut = d_22_ci_
                                    currentConstrainedOut = d_23_cc_
                                    d_2_steps_ = (d_2_steps_) + (1)
                            raise _dafny.Break("0")
                        elif True:
                            d_24_ag_: _dafny.Seq
                            d_25_ai_: bool
                            d_26_ac_: _dafny.Seq
                            out20_: _dafny.Seq
                            out21_: bool
                            out22_: _dafny.Seq
                            out20_, out21_, out22_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_13_next_)
                            d_24_ag_ = out20_
                            d_25_ai_ = out21_
                            d_26_ac_ = out22_
                            generated = d_24_ag_
                            insideConstrainedOut = d_25_ai_
                            currentConstrainedOut = d_26_ac_
                    pass
            pass
        if insideConstrainedOut:
            d_27_rg_: _dafny.Seq
            d_28_rc_: _dafny.Seq
            out23_: _dafny.Seq
            out24_: _dafny.Seq
            out23_, out24_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
            d_27_rg_ = out23_
            d_28_rc_ = out24_
            generated = d_27_rg_
            currentConstrainedOut = d_28_rc_
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_29_closeBudget_: int
            d_29_closeBudget_ = (maxSteps) - (d_2_steps_)
            d_30_cg_: _dafny.Seq
            d_31_ci_: bool
            d_32_cc_: _dafny.Seq
            out25_: _dafny.Seq
            out26_: bool
            out27_: _dafny.Seq
            out25_, out26_, out27_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_29_closeBudget_)
            d_30_cg_ = out25_
            d_31_ci_ = out26_
            d_32_cc_ = out27_
            generated = d_30_cg_
            insideConstrainedOut = d_31_ci_
            currentConstrainedOut = d_32_cc_
            d_2_steps_ = maxSteps
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

