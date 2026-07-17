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
                    if (parser).IsCompletePrefix(currentConstrainedOut):
                        d_5_currentStr_: _dafny.Seq
                        d_5_currentStr_ = VerifiedDecoderAgent.CSDHelpers.PrefixToString(currentConstrainedOut)
                        d_6_hasIsocyanate_: bool
                        d_6_hasIsocyanate_ = (VerifiedDecoderAgent.CSDHelpers.CountSubstring(d_5_currentStr_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "N=C=O")))) > (0)
                        if d_6_hasIsocyanate_:
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
                        elif (len(currentConstrainedOut)) >= (20):
                            d_10_cg_: _dafny.Seq
                            d_11_ci_: bool
                            d_12_cc_: _dafny.Seq
                            out6_: _dafny.Seq
                            out7_: bool
                            out8_: _dafny.Seq
                            out6_, out7_, out8_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_10_cg_ = out6_
                            d_11_ci_ = out7_
                            d_12_cc_ = out8_
                            generated = d_10_cg_
                            insideConstrainedOut = d_11_ci_
                            currentConstrainedOut = d_12_cc_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_13_currentStr2_: _dafny.Seq
                            d_13_currentStr2_ = VerifiedDecoderAgent.CSDHelpers.PrefixToString(currentConstrainedOut)
                            d_14_nValid_: bool
                            out9_: bool
                            out9_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "N")))
                            d_14_nValid_ = out9_
                            d_15_nInVocab_: bool
                            d_15_nInVocab_ = (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "N"))) in ((lm).Tokens)
                            if (d_14_nValid_) and (d_15_nInVocab_):
                                (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "N"))]), _dafny.BigRational('2e1'))
                            d_16_constrainedPrompt_: _dafny.Seq
                            d_16_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_17_next_: _dafny.Seq
                            out10_: _dafny.Seq
                            out10_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_16_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_17_next_ = out10_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_17_next_) == (eosToken):
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
                                raise _dafny.Break("0")
                            elif True:
                                d_21_valid_: bool
                                out14_: bool
                                out14_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_17_next_)
                                d_21_valid_ = out14_
                                if (d_21_valid_) and (not((parser).IsCompletePrefix(currentConstrainedOut))):
                                    d_22_ag_: _dafny.Seq
                                    d_23_ai_: bool
                                    d_24_ac_: _dafny.Seq
                                    out15_: _dafny.Seq
                                    out16_: bool
                                    out17_: _dafny.Seq
                                    out15_, out16_, out17_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_17_next_)
                                    d_22_ag_ = out15_
                                    d_23_ai_ = out16_
                                    d_24_ac_ = out17_
                                    generated = d_22_ag_
                                    insideConstrainedOut = d_23_ai_
                                    currentConstrainedOut = d_24_ac_
                    elif True:
                        d_25_currentStr_: _dafny.Seq
                        d_25_currentStr_ = VerifiedDecoderAgent.CSDHelpers.PrefixToString(currentConstrainedOut)
                        d_26_hasN_: bool
                        d_26_hasN_ = (VerifiedDecoderAgent.CSDHelpers.CountSubstring(d_25_currentStr_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "N")))) > (0)
                        d_27_hasNEq_: bool
                        d_27_hasNEq_ = (VerifiedDecoderAgent.CSDHelpers.CountSubstring(d_25_currentStr_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "N=")))) > (0)
                        d_28_hasNEqC_: bool
                        d_28_hasNEqC_ = (VerifiedDecoderAgent.CSDHelpers.CountSubstring(d_25_currentStr_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "N=C")))) > (0)
                        d_29_hasNEqCEq_: bool
                        d_29_hasNEqCEq_ = (VerifiedDecoderAgent.CSDHelpers.CountSubstring(d_25_currentStr_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "N=C=")))) > (0)
                        d_30_hasIsocyanate_: bool
                        d_30_hasIsocyanate_ = (VerifiedDecoderAgent.CSDHelpers.CountSubstring(d_25_currentStr_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "N=C=O")))) > (0)
                        if not(d_30_hasIsocyanate_):
                            if not(d_26_hasN_):
                                d_31_nValid_: bool
                                out18_: bool
                                out18_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "N")))
                                d_31_nValid_ = out18_
                                d_32_nInVocab_: bool
                                d_32_nInVocab_ = (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "N"))) in ((lm).Tokens)
                                if (d_31_nValid_) and (d_32_nInVocab_):
                                    (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "N"))]), _dafny.BigRational('15e0'))
                            elif not(d_27_hasNEq_):
                                d_33_eqValid_: bool
                                out19_: bool
                                out19_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")))
                                d_33_eqValid_ = out19_
                                d_34_eqInVocab_: bool
                                d_34_eqInVocab_ = (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "="))) in ((lm).Tokens)
                                if (d_33_eqValid_) and (d_34_eqInVocab_):
                                    (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "="))]), _dafny.BigRational('15e0'))
                            elif not(d_28_hasNEqC_):
                                d_35_cValid_: bool
                                out20_: bool
                                out20_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "C")))
                                d_35_cValid_ = out20_
                                d_36_cInVocab_: bool
                                d_36_cInVocab_ = (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "C"))) in ((lm).Tokens)
                                if (d_35_cValid_) and (d_36_cInVocab_):
                                    (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "C"))]), _dafny.BigRational('15e0'))
                            elif not(d_29_hasNEqCEq_):
                                d_37_eqValid_: bool
                                out21_: bool
                                out21_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")))
                                d_37_eqValid_ = out21_
                                d_38_eqInVocab_: bool
                                d_38_eqInVocab_ = (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "="))) in ((lm).Tokens)
                                if (d_37_eqValid_) and (d_38_eqInVocab_):
                                    (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "="))]), _dafny.BigRational('15e0'))
                            elif True:
                                d_39_oValid_: bool
                                out22_: bool
                                out22_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "O")))
                                d_39_oValid_ = out22_
                                d_40_oInVocab_: bool
                                d_40_oInVocab_ = (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "O"))) in ((lm).Tokens)
                                if (d_39_oValid_) and (d_40_oInVocab_):
                                    (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "O"))]), _dafny.BigRational('15e0'))
                        d_41_constrainedPrompt_: _dafny.Seq
                        d_41_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_42_next_: _dafny.Seq
                        out23_: _dafny.Seq
                        out23_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_41_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('15e-1'), eosToken)
                        d_42_next_ = out23_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_42_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_43_valid_: bool
                            out24_: bool
                            out24_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_42_next_)
                            d_43_valid_ = out24_
                            if (d_43_valid_) and (not((parser).IsCompletePrefix(currentConstrainedOut))):
                                d_44_ag_: _dafny.Seq
                                d_45_ai_: bool
                                d_46_ac_: _dafny.Seq
                                out25_: _dafny.Seq
                                out26_: bool
                                out27_: _dafny.Seq
                                out25_, out26_, out27_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_42_next_)
                                d_44_ag_ = out25_
                                d_45_ai_ = out26_
                                d_46_ac_ = out27_
                                generated = d_44_ag_
                                insideConstrainedOut = d_45_ai_
                                currentConstrainedOut = d_46_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_47_closeBudget_: int
            d_47_closeBudget_ = (maxSteps) - (d_1_steps_)
            if (d_47_closeBudget_) > (15):
                d_47_closeBudget_ = 15
            d_48_cg_: _dafny.Seq
            d_49_ci_: bool
            d_50_cc_: _dafny.Seq
            out28_: _dafny.Seq
            out29_: bool
            out30_: _dafny.Seq
            out28_, out29_, out30_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_47_closeBudget_)
            d_48_cg_ = out28_
            d_49_ci_ = out29_
            d_50_cc_ = out30_
            generated = d_48_cg_
            insideConstrainedOut = d_49_ci_
            currentConstrainedOut = d_50_cc_
            d_1_steps_ = (d_1_steps_) + (d_47_closeBudget_)
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

