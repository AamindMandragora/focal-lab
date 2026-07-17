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
                    d_5_currentStr_: _dafny.Seq
                    d_5_currentStr_ = VerifiedDecoderAgent.CSDHelpers.PrefixToString(currentConstrainedOut)
                    d_6_hasIsocyanate_: bool
                    d_6_hasIsocyanate_ = (VerifiedDecoderAgent.CSDHelpers.CountSubstring(d_5_currentStr_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "N=C=O")))) > (0)
                    d_7_spanLen_: int
                    d_7_spanLen_ = len(currentConstrainedOut)
                    if ((parser).IsCompletePrefix(currentConstrainedOut)) and (d_6_hasIsocyanate_):
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
                    elif ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_7_spanLen_) >= (25)):
                        d_11_cg_: _dafny.Seq
                        d_12_ci_: bool
                        d_13_cc_: _dafny.Seq
                        out6_: _dafny.Seq
                        out7_: bool
                        out8_: _dafny.Seq
                        out6_, out7_, out8_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_11_cg_ = out6_
                        d_12_ci_ = out7_
                        d_13_cc_ = out8_
                        generated = d_11_cg_
                        insideConstrainedOut = d_12_ci_
                        currentConstrainedOut = d_13_cc_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_14_hasN_: bool
                        d_14_hasN_ = (VerifiedDecoderAgent.CSDHelpers.CountSubstring(d_5_currentStr_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "N")))) > (0)
                        d_15_hasNEq_: bool
                        d_15_hasNEq_ = (VerifiedDecoderAgent.CSDHelpers.CountSubstring(d_5_currentStr_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "N=")))) > (0)
                        d_16_hasNEqC_: bool
                        d_16_hasNEqC_ = (VerifiedDecoderAgent.CSDHelpers.CountSubstring(d_5_currentStr_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "N=C")))) > (0)
                        d_17_hasNEqCEq_: bool
                        d_17_hasNEqCEq_ = (VerifiedDecoderAgent.CSDHelpers.CountSubstring(d_5_currentStr_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "N=C=")))) > (0)
                        if not(d_6_hasIsocyanate_):
                            if not(d_14_hasN_):
                                d_18_nValid_: bool
                                out9_: bool
                                out9_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "N")))
                                d_18_nValid_ = out9_
                                if (d_18_nValid_) and ((_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "N"))) in ((lm).Tokens)):
                                    (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "N"))]), _dafny.BigRational('2e1'))
                                d_19_cValid_: bool
                                out10_: bool
                                out10_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "C")))
                                d_19_cValid_ = out10_
                                if ((d_19_cValid_) and ((_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "C"))) in ((lm).Tokens))) and ((d_7_spanLen_) == (0)):
                                    (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "C"))]), _dafny.BigRational('5e0'))
                            elif not(d_15_hasNEq_):
                                d_20_eqValid_: bool
                                out11_: bool
                                out11_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")))
                                d_20_eqValid_ = out11_
                                if (d_20_eqValid_) and ((_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "="))) in ((lm).Tokens)):
                                    (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "="))]), _dafny.BigRational('2e1'))
                            elif not(d_16_hasNEqC_):
                                d_21_cValid_: bool
                                out12_: bool
                                out12_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "C")))
                                d_21_cValid_ = out12_
                                if (d_21_cValid_) and ((_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "C"))) in ((lm).Tokens)):
                                    (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "C"))]), _dafny.BigRational('2e1'))
                            elif not(d_17_hasNEqCEq_):
                                d_22_eqValid_: bool
                                out13_: bool
                                out13_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")))
                                d_22_eqValid_ = out13_
                                if (d_22_eqValid_) and ((_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "="))) in ((lm).Tokens)):
                                    (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "="))]), _dafny.BigRational('2e1'))
                            elif True:
                                d_23_oValid_: bool
                                out14_: bool
                                out14_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "O")))
                                d_23_oValid_ = out14_
                                if (d_23_oValid_) and ((_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "O"))) in ((lm).Tokens)):
                                    (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "O"))]), _dafny.BigRational('2e1'))
                        d_24_constrainedPrompt_: _dafny.Seq
                        d_24_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_25_next_: _dafny.Seq
                        out15_: _dafny.Seq
                        out15_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_24_constrainedPrompt_, currentConstrainedOut, eosToken)
                        d_25_next_ = out15_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_25_next_) == (eosToken):
                            if ((parser).IsCompletePrefix(currentConstrainedOut)) and (d_6_hasIsocyanate_):
                                d_26_cg_: _dafny.Seq
                                d_27_ci_: bool
                                d_28_cc_: _dafny.Seq
                                out16_: _dafny.Seq
                                out17_: bool
                                out18_: _dafny.Seq
                                out16_, out17_, out18_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_26_cg_ = out16_
                                d_27_ci_ = out17_
                                d_28_cc_ = out18_
                                generated = d_26_cg_
                                insideConstrainedOut = d_27_ci_
                                currentConstrainedOut = d_28_cc_
                            raise _dafny.Break("0")
                        elif True:
                            d_29_valid_: bool
                            out19_: bool
                            out19_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_25_next_)
                            d_29_valid_ = out19_
                            if (d_29_valid_) and (not((parser).IsCompletePrefix(currentConstrainedOut))):
                                d_30_ag_: _dafny.Seq
                                d_31_ai_: bool
                                d_32_ac_: _dafny.Seq
                                out20_: _dafny.Seq
                                out21_: bool
                                out22_: _dafny.Seq
                                out20_, out21_, out22_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_25_next_)
                                d_30_ag_ = out20_
                                d_31_ai_ = out21_
                                d_32_ac_ = out22_
                                generated = d_30_ag_
                                insideConstrainedOut = d_31_ai_
                                currentConstrainedOut = d_32_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_33_closeBudget_: int
            d_33_closeBudget_ = (maxSteps) - (d_1_steps_)
            if (d_33_closeBudget_) > (10):
                d_33_closeBudget_ = 10
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
            d_1_steps_ = (d_1_steps_) + (d_33_closeBudget_)
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

