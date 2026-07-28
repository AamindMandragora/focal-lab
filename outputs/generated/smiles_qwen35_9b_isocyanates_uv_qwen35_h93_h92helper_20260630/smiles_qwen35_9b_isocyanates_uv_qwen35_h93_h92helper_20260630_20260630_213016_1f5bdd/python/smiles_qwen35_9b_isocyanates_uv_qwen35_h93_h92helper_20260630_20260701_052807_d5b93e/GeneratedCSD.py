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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate a SMILES for an ISOCYANATE. The SMILES MUST contain N=C=O. The isocyanate functional group is -N=C=O. You MUST generate the atoms N, =, C, =, O in sequence. Valid examples: CCN=C=O, CN=C=O, CCCN=C=O. REQUIRED: include N=C=O in your output.")))
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
                    d_7_isComplete_: bool
                    d_7_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                    if (d_7_isComplete_) and (d_6_hasIsocyanate_):
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
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif (d_7_isComplete_) and (not(d_6_hasIsocyanate_)):
                        d_11_closeBudget_: int
                        d_11_closeBudget_ = (maxSteps) - (d_1_steps_)
                        if (d_11_closeBudget_) > (0):
                            d_12_cg_: _dafny.Seq
                            d_13_ci_: bool
                            d_14_cc_: _dafny.Seq
                            out6_: _dafny.Seq
                            out7_: bool
                            out8_: _dafny.Seq
                            out6_, out7_, out8_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_11_closeBudget_)
                            d_12_cg_ = out6_
                            d_13_ci_ = out7_
                            d_14_cc_ = out8_
                            generated = d_12_cg_
                            insideConstrainedOut = d_13_ci_
                            currentConstrainedOut = d_14_cc_
                            d_1_steps_ = maxSteps
                        elif True:
                            d_15_cg_: _dafny.Seq
                            d_16_ci_: bool
                            d_17_cc_: _dafny.Seq
                            out9_: _dafny.Seq
                            out10_: bool
                            out11_: _dafny.Seq
                            out9_, out10_, out11_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_15_cg_ = out9_
                            d_16_ci_ = out10_
                            d_17_cc_ = out11_
                            generated = d_15_cg_
                            insideConstrainedOut = d_16_ci_
                            currentConstrainedOut = d_17_cc_
                            d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_18_constrainedPrompt_: _dafny.Seq
                        d_18_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_19_hasNCeqO_: bool
                        d_19_hasNCeqO_ = (VerifiedDecoderAgent.CSDHelpers.CountSubstring(d_5_currentStr_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "N=C=O")))) > (0)
                        d_20_hasNCeq_: bool
                        d_20_hasNCeq_ = (VerifiedDecoderAgent.CSDHelpers.CountSubstring(d_5_currentStr_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "N=C=")))) > (0)
                        d_21_hasNCtoken_: bool
                        d_21_hasNCtoken_ = (VerifiedDecoderAgent.CSDHelpers.CountSubstring(d_5_currentStr_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "N=C")))) > (0)
                        d_22_hasNeqToken_: bool
                        d_22_hasNeqToken_ = (VerifiedDecoderAgent.CSDHelpers.CountSubstring(d_5_currentStr_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "N=")))) > (0)
                        d_23_hasNToken_: bool
                        d_23_hasNToken_ = (VerifiedDecoderAgent.CSDHelpers.CountSubstring(d_5_currentStr_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "N")))) > (0)
                        if not(d_23_hasNToken_):
                            if (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "N"))) in ((lm).Tokens):
                                d_24_nValid_: bool
                                out12_: bool
                                out12_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "N")))
                                d_24_nValid_ = out12_
                                if d_24_nValid_:
                                    (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "N"))]), _dafny.BigRational('15e0'))
                            if (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "C"))) in ((lm).Tokens):
                                d_25_cValid_: bool
                                out13_: bool
                                out13_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "C")))
                                d_25_cValid_ = out13_
                                if d_25_cValid_:
                                    (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "C"))]), _dafny.BigRational('4e0'))
                        elif (d_23_hasNToken_) and (not(d_22_hasNeqToken_)):
                            if (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "="))) in ((lm).Tokens):
                                d_26_eqValid_: bool
                                out14_: bool
                                out14_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")))
                                d_26_eqValid_ = out14_
                                if d_26_eqValid_:
                                    (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "="))]), _dafny.BigRational('15e0'))
                        elif (d_22_hasNeqToken_) and (not(d_21_hasNCtoken_)):
                            if (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "C"))) in ((lm).Tokens):
                                d_27_cValid_: bool
                                out15_: bool
                                out15_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "C")))
                                d_27_cValid_ = out15_
                                if d_27_cValid_:
                                    (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "C"))]), _dafny.BigRational('15e0'))
                        elif (d_21_hasNCtoken_) and (not(d_20_hasNCeq_)):
                            if (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "="))) in ((lm).Tokens):
                                d_28_eqValid_: bool
                                out16_: bool
                                out16_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")))
                                d_28_eqValid_ = out16_
                                if d_28_eqValid_:
                                    (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "="))]), _dafny.BigRational('15e0'))
                        elif (d_20_hasNCeq_) and (not(d_19_hasNCeqO_)):
                            if (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "O"))) in ((lm).Tokens):
                                d_29_oValid_: bool
                                out17_: bool
                                out17_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "O")))
                                d_29_oValid_ = out17_
                                if d_29_oValid_:
                                    (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "O"))]), _dafny.BigRational('15e0'))
                        d_30_next_: _dafny.Seq
                        d_31_usedFallback_: bool
                        out18_: _dafny.Seq
                        out19_: bool
                        out18_, out19_ = (d_0_helpers_).SafeSoftConstrainedStep(lm, parser, d_18_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('2e0'), eosToken)
                        d_30_next_ = out18_
                        d_31_usedFallback_ = out19_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_30_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_32_valid_: bool
                            out20_: bool
                            out20_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_30_next_)
                            d_32_valid_ = out20_
                            d_33_notComplete_: bool
                            d_33_notComplete_ = not((parser).IsCompletePrefix(currentConstrainedOut))
                            if (d_32_valid_) and (d_33_notComplete_):
                                d_34_ag_: _dafny.Seq
                                d_35_ai_: bool
                                d_36_ac_: _dafny.Seq
                                out21_: _dafny.Seq
                                out22_: bool
                                out23_: _dafny.Seq
                                out21_, out22_, out23_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_30_next_)
                                d_34_ag_ = out21_
                                d_35_ai_ = out22_
                                d_36_ac_ = out23_
                                generated = d_34_ag_
                                insideConstrainedOut = d_35_ai_
                                currentConstrainedOut = d_36_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_37_closeBudget_: int
            d_37_closeBudget_ = (maxSteps) - (d_1_steps_)
            d_38_cg_: _dafny.Seq
            d_39_ci_: bool
            d_40_cc_: _dafny.Seq
            out24_: _dafny.Seq
            out25_: bool
            out26_: _dafny.Seq
            out24_, out25_, out26_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_37_closeBudget_)
            d_38_cg_ = out24_
            d_39_ci_ = out25_
            d_40_cc_ = out26_
            generated = d_38_cg_
            insideConstrainedOut = d_39_ci_
            currentConstrainedOut = d_40_cc_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

