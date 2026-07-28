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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Task: Output exactly one valid SMILES for an acrylate ester molecule. Acrylate esters MUST start with C=CC(=O)O (the acryloyl group CH2=CH-C(=O)-O-). Valid examples: C=CC(=O)OCC, C=CC(=O)OC, C=CC(=O)OCCO, C=CC(=O)OC(C)C, C=CC(=O)OCCC, C=CC(=O)OCCCC, C=CC(=O)OCC(C)C, C=CC(=O)OCCOCCO, C=CC(=O)OCC(F)(F)F, C=CC(=O)OCCN. Output the SMILES string only."))
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
        d_7_coreTokens_: _dafny.Seq
        d_7_coreTokens_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "C")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "C")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "C")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "(")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "O")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ")")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "O"))])
        d_8_coreIdx_: int
        d_8_coreIdx_ = 0
        with _dafny.label("0"):
            while ((insideConstrainedOut) and ((d_2_steps_) < (maxSteps))) and ((d_8_coreIdx_) < (len(d_7_coreTokens_))):
                with _dafny.c_label("0"):
                    d_9_targetToken_: _dafny.Seq
                    d_9_targetToken_ = (d_7_coreTokens_)[d_8_coreIdx_]
                    d_10_constrainedPrompt_: _dafny.Seq
                    d_10_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                    d_11_targetValid_: bool
                    out3_: bool
                    out3_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_9_targetToken_)
                    d_11_targetValid_ = out3_
                    d_12_next_: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                    if d_11_targetValid_:
                        d_13_targetGroup_: _dafny.Seq
                        d_13_targetGroup_ = _dafny.SeqWithoutIsStrInference([d_9_targetToken_])
                        d_14_targetGroups_: _dafny.Seq
                        d_14_targetGroups_ = _dafny.SeqWithoutIsStrInference([d_13_targetGroup_])
                        out4_: _dafny.Seq
                        out4_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_10_constrainedPrompt_, currentConstrainedOut, d_14_targetGroups_, _dafny.BigRational('2e1'), 1000, eosToken)
                        d_12_next_ = out4_
                    elif True:
                        out5_: _dafny.Seq
                        out5_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_10_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                        d_12_next_ = out5_
                        d_8_coreIdx_ = len(d_7_coreTokens_)
                    d_2_steps_ = (d_2_steps_) + (1)
                    if (d_12_next_) == (eosToken):
                        raise _dafny.Break("0")
                    elif True:
                        d_15_valid_: bool
                        out6_: bool
                        out6_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_12_next_)
                        d_15_valid_ = out6_
                        if d_15_valid_:
                            d_16_ag_: _dafny.Seq
                            d_17_ai_: bool
                            d_18_ac_: _dafny.Seq
                            out7_: _dafny.Seq
                            out8_: bool
                            out9_: _dafny.Seq
                            out7_, out8_, out9_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_12_next_)
                            d_16_ag_ = out7_
                            d_17_ai_ = out8_
                            d_18_ac_ = out9_
                            generated = d_16_ag_
                            insideConstrainedOut = d_17_ai_
                            currentConstrainedOut = d_18_ac_
                            if (d_12_next_) == (d_9_targetToken_):
                                d_8_coreIdx_ = (d_8_coreIdx_) + (1)
                            elif True:
                                d_8_coreIdx_ = len(d_7_coreTokens_)
                    pass
            pass
        d_19_repPenaltyCount_: int
        d_19_repPenaltyCount_ = 0
        with _dafny.label("1"):
            while (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
                with _dafny.c_label("1"):
                    if not(insideConstrainedOut):
                        raise _dafny.Break("1")
                    elif ((parser).IsCompletePrefix(currentConstrainedOut)) and ((len(currentConstrainedOut)) >= (d_3_minLength_)):
                        d_20_cg_: _dafny.Seq
                        d_21_ci_: bool
                        d_22_cc_: _dafny.Seq
                        out10_: _dafny.Seq
                        out11_: bool
                        out12_: _dafny.Seq
                        out10_, out11_, out12_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_20_cg_ = out10_
                        d_21_ci_ = out11_
                        d_22_cc_ = out12_
                        generated = d_20_cg_
                        insideConstrainedOut = d_21_ci_
                        currentConstrainedOut = d_22_cc_
                        d_2_steps_ = (d_2_steps_) + (1)
                        raise _dafny.Break("1")
                    elif True:
                        d_23_constrainedPrompt_: _dafny.Seq
                        d_23_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_24_next_: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                        if ((_dafny.euclidian_modulus(d_19_repPenaltyCount_, 4)) == (1)) and ((len(currentConstrainedOut)) >= (9)):
                            out13_: _dafny.Seq
                            out13_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_23_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('15e-1'), eosToken)
                            d_24_next_ = out13_
                        elif True:
                            out14_: _dafny.Seq
                            out14_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_23_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                            d_24_next_ = out14_
                        d_19_repPenaltyCount_ = (d_19_repPenaltyCount_) + (1)
                        d_2_steps_ = (d_2_steps_) + (1)
                        if (d_24_next_) == (eosToken):
                            if (((parser).IsCompletePrefix(currentConstrainedOut)) and ((len(currentConstrainedOut)) >= (d_3_minLength_))) and ((d_2_steps_) < (maxSteps)):
                                d_25_cg_: _dafny.Seq
                                d_26_ci_: bool
                                d_27_cc_: _dafny.Seq
                                out15_: _dafny.Seq
                                out16_: bool
                                out17_: _dafny.Seq
                                out15_, out16_, out17_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_25_cg_ = out15_
                                d_26_ci_ = out16_
                                d_27_cc_ = out17_
                                generated = d_25_cg_
                                insideConstrainedOut = d_26_ci_
                                currentConstrainedOut = d_27_cc_
                                d_2_steps_ = (d_2_steps_) + (1)
                            raise _dafny.Break("1")
                        elif True:
                            d_28_valid_: bool
                            out18_: bool
                            out18_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_24_next_)
                            d_28_valid_ = out18_
                            if d_28_valid_:
                                d_29_ag_: _dafny.Seq
                                d_30_ai_: bool
                                d_31_ac_: _dafny.Seq
                                out19_: _dafny.Seq
                                out20_: bool
                                out21_: _dafny.Seq
                                out19_, out20_, out21_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_24_next_)
                                d_29_ag_ = out19_
                                d_30_ai_ = out20_
                                d_31_ac_ = out21_
                                generated = d_29_ag_
                                insideConstrainedOut = d_30_ai_
                                currentConstrainedOut = d_31_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            if (len(currentConstrainedOut)) >= (d_3_minLength_):
                d_32_closeBudget_: int
                d_32_closeBudget_ = (maxSteps) - (d_2_steps_)
                d_33_cg_: _dafny.Seq
                d_34_ci_: bool
                d_35_cc_: _dafny.Seq
                out22_: _dafny.Seq
                out23_: bool
                out24_: _dafny.Seq
                out22_, out23_, out24_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_32_closeBudget_)
                d_33_cg_ = out22_
                d_34_ci_ = out23_
                d_35_cc_ = out24_
                generated = d_33_cg_
                insideConstrainedOut = d_34_ci_
                currentConstrainedOut = d_35_cc_
                d_2_steps_ = maxSteps
            elif True:
                with _dafny.label("3_1_0"):
                    while (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
                        with _dafny.c_label("3_1_0"):
                            if not(insideConstrainedOut):
                                raise _dafny.Break("3_1_0")
                            elif ((parser).IsCompletePrefix(currentConstrainedOut)) and ((len(currentConstrainedOut)) >= (d_3_minLength_)):
                                d_36_cg_: _dafny.Seq
                                d_37_ci_: bool
                                d_38_cc_: _dafny.Seq
                                out25_: _dafny.Seq
                                out26_: bool
                                out27_: _dafny.Seq
                                out25_, out26_, out27_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_36_cg_ = out25_
                                d_37_ci_ = out26_
                                d_38_cc_ = out27_
                                generated = d_36_cg_
                                insideConstrainedOut = d_37_ci_
                                currentConstrainedOut = d_38_cc_
                                d_2_steps_ = (d_2_steps_) + (1)
                                raise _dafny.Break("3_1_0")
                            elif True:
                                d_39_constrainedPrompt2_: _dafny.Seq
                                d_39_constrainedPrompt2_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_40_next2_: _dafny.Seq
                                out28_: _dafny.Seq
                                out28_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_39_constrainedPrompt2_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                                d_40_next2_ = out28_
                                d_2_steps_ = (d_2_steps_) + (1)
                                if (d_40_next2_) == (eosToken):
                                    raise _dafny.Break("3_1_0")
                                elif True:
                                    d_41_valid2_: bool
                                    out29_: bool
                                    out29_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_40_next2_)
                                    d_41_valid2_ = out29_
                                    if d_41_valid2_:
                                        d_42_ag2_: _dafny.Seq
                                        d_43_ai2_: bool
                                        d_44_ac2_: _dafny.Seq
                                        out30_: _dafny.Seq
                                        out31_: bool
                                        out32_: _dafny.Seq
                                        out30_, out31_, out32_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_40_next2_)
                                        d_42_ag2_ = out30_
                                        d_43_ai2_ = out31_
                                        d_44_ac2_ = out32_
                                        generated = d_42_ag2_
                                        insideConstrainedOut = d_43_ai2_
                                        currentConstrainedOut = d_44_ac2_
                            pass
                    pass
                if ((insideConstrainedOut) and ((d_2_steps_) < (maxSteps))) and ((len(currentConstrainedOut)) >= (d_3_minLength_)):
                    d_45_closeBudget2_: int
                    d_45_closeBudget2_ = (maxSteps) - (d_2_steps_)
                    d_46_cg2_: _dafny.Seq
                    d_47_ci2_: bool
                    d_48_cc2_: _dafny.Seq
                    out33_: _dafny.Seq
                    out34_: bool
                    out35_: _dafny.Seq
                    out33_, out34_, out35_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_45_closeBudget2_)
                    d_46_cg2_ = out33_
                    d_47_ci2_ = out34_
                    d_48_cc2_ = out35_
                    generated = d_46_cg2_
                    insideConstrainedOut = d_47_ci2_
                    currentConstrainedOut = d_48_cc2_
                    d_2_steps_ = maxSteps
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

