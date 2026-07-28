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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate a SMILES string for an ISOCYANATE. Isocyanate SMILES must contain N=C=O. Start with an alkyl group then N=C=O. Examples: CCN=C=O, CN=C=O, CCCN=C=O, CC(C)N=C=O, CCCCN=C=O.")))
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
                    if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_6_hasIsocyanate_) or ((d_7_spanLen_) > (15))):
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
                    elif True:
                        d_11_constrainedPrompt_: _dafny.Seq
                        d_11_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_12_hasNCeq_: bool
                        d_12_hasNCeq_ = ((VerifiedDecoderAgent.CSDHelpers.CountSubstring(d_5_currentStr_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "N=C=")))) > (0)) and (not(d_6_hasIsocyanate_))
                        d_13_hasNC_: bool
                        d_13_hasNC_ = ((VerifiedDecoderAgent.CSDHelpers.CountSubstring(d_5_currentStr_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "N=C")))) > (0)) and ((VerifiedDecoderAgent.CSDHelpers.CountSubstring(d_5_currentStr_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "N=C=")))) == (0))
                        d_14_hasNeq_: bool
                        d_14_hasNeq_ = ((VerifiedDecoderAgent.CSDHelpers.CountSubstring(d_5_currentStr_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "N=")))) > (0)) and ((VerifiedDecoderAgent.CSDHelpers.CountSubstring(d_5_currentStr_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "N=C")))) == (0))
                        d_15_lastIsN_: bool
                        d_15_lastIsN_ = ((d_7_spanLen_) > (0)) and (((currentConstrainedOut)[(d_7_spanLen_) - (1)]) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "N"))))
                        d_16_nValid_: bool
                        out6_: bool
                        out6_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "N")))
                        d_16_nValid_ = out6_
                        d_17_eqValid_: bool
                        out7_: bool
                        out7_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")))
                        d_17_eqValid_ = out7_
                        d_18_cValid_: bool
                        out8_: bool
                        out8_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "C")))
                        d_18_cValid_ = out8_
                        d_19_oValid_: bool
                        out9_: bool
                        out9_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "O")))
                        d_19_oValid_ = out9_
                        if ((d_12_hasNCeq_) and (d_19_oValid_)) and ((_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "O"))) in ((lm).Tokens)):
                            (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "O"))]), _dafny.BigRational('1e1'))
                        elif ((d_13_hasNC_) and (d_17_eqValid_)) and ((_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "="))) in ((lm).Tokens)):
                            (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "="))]), _dafny.BigRational('1e1'))
                        elif ((d_15_lastIsN_) and (d_17_eqValid_)) and ((_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "="))) in ((lm).Tokens)):
                            (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "="))]), _dafny.BigRational('1e1'))
                        elif ((d_14_hasNeq_) and (d_18_cValid_)) and ((_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "C"))) in ((lm).Tokens)):
                            (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "C"))]), _dafny.BigRational('1e1'))
                        elif True:
                            if ((d_16_nValid_) and ((_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "N"))) in ((lm).Tokens))) and (not(d_6_hasIsocyanate_)):
                                (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "N"))]), _dafny.BigRational('5e0'))
                            if (d_17_eqValid_) and ((_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "="))) in ((lm).Tokens)):
                                (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "="))]), _dafny.BigRational('3e0'))
                            if ((d_18_cValid_) and ((_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "C"))) in ((lm).Tokens))) and ((d_7_spanLen_) < (3)):
                                (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "C"))]), _dafny.BigRational('4e0'))
                            if ((d_19_oValid_) and ((_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "O"))) in ((lm).Tokens))) and (d_13_hasNC_):
                                (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "O"))]), _dafny.BigRational('8e0'))
                        d_20_next_: _dafny.Seq
                        out10_: _dafny.Seq
                        out10_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_11_constrainedPrompt_, currentConstrainedOut, eosToken)
                        d_20_next_ = out10_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_20_next_) == (eosToken):
                            if (parser).IsCompletePrefix(currentConstrainedOut):
                                d_21_cg_: _dafny.Seq
                                d_22_ci_: bool
                                d_23_cc_: _dafny.Seq
                                out11_: _dafny.Seq
                                out12_: bool
                                out13_: _dafny.Seq
                                out11_, out12_, out13_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_21_cg_ = out11_
                                d_22_ci_ = out12_
                                d_23_cc_ = out13_
                                generated = d_21_cg_
                                insideConstrainedOut = d_22_ci_
                                currentConstrainedOut = d_23_cc_
                            raise _dafny.Break("0")
                        elif True:
                            d_24_valid_: bool
                            out14_: bool
                            out14_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_20_next_)
                            d_24_valid_ = out14_
                            if (d_24_valid_) and (not((parser).IsCompletePrefix(currentConstrainedOut))):
                                d_25_ag_: _dafny.Seq
                                d_26_ai_: bool
                                d_27_ac_: _dafny.Seq
                                out15_: _dafny.Seq
                                out16_: bool
                                out17_: _dafny.Seq
                                out15_, out16_, out17_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_20_next_)
                                d_25_ag_ = out15_
                                d_26_ai_ = out16_
                                d_27_ac_ = out17_
                                generated = d_25_ag_
                                insideConstrainedOut = d_26_ai_
                                currentConstrainedOut = d_27_ac_
                            elif (d_24_valid_) and ((parser).IsCompletePrefix(currentConstrainedOut)):
                                if (d_1_steps_) < (maxSteps):
                                    d_28_cg_: _dafny.Seq
                                    d_29_ci_: bool
                                    d_30_cc_: _dafny.Seq
                                    out18_: _dafny.Seq
                                    out19_: bool
                                    out20_: _dafny.Seq
                                    out18_, out19_, out20_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_28_cg_ = out18_
                                    d_29_ci_ = out19_
                                    d_30_cc_ = out20_
                                    generated = d_28_cg_
                                    insideConstrainedOut = d_29_ci_
                                    currentConstrainedOut = d_30_cc_
                                    d_1_steps_ = (d_1_steps_) + (1)
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_31_closeBudget_: int
            d_31_closeBudget_ = (maxSteps) - (d_1_steps_)
            d_32_cg_: _dafny.Seq
            d_33_ci_: bool
            d_34_cc_: _dafny.Seq
            out21_: _dafny.Seq
            out22_: bool
            out23_: _dafny.Seq
            out21_, out22_, out23_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_31_closeBudget_)
            d_32_cg_ = out21_
            d_33_ci_ = out22_
            d_34_cc_ = out23_
            generated = d_32_cg_
            insideConstrainedOut = d_33_ci_
            currentConstrainedOut = d_34_cc_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

