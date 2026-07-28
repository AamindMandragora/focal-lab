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
                    d_8_isComplete_: bool
                    d_8_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                    if d_8_isComplete_:
                        d_9_cg_: _dafny.Seq
                        d_10_ci_: bool
                        d_11_cc_: _dafny.Seq
                        out3_: _dafny.Seq
                        out4_: bool
                        out5_: _dafny.Seq
                        out3_, out4_, out5_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_9_cg_ = out3_
                        d_10_ci_ = out4_
                        d_11_cc_ = out5_
                        generated = d_9_cg_
                        insideConstrainedOut = d_10_ci_
                        currentConstrainedOut = d_11_cc_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_12_constrainedPrompt_: _dafny.Seq
                        d_12_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_13_hasNCeqO_: bool
                        d_13_hasNCeqO_ = (VerifiedDecoderAgent.CSDHelpers.CountSubstring(d_5_currentStr_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "N=C=O")))) > (0)
                        d_14_hasNCeq_: bool
                        d_14_hasNCeq_ = (VerifiedDecoderAgent.CSDHelpers.CountSubstring(d_5_currentStr_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "N=C=")))) > (0)
                        d_15_hasNC_: bool
                        d_15_hasNC_ = (VerifiedDecoderAgent.CSDHelpers.CountSubstring(d_5_currentStr_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "N=C")))) > (0)
                        d_16_hasNeq_: bool
                        d_16_hasNeq_ = (VerifiedDecoderAgent.CSDHelpers.CountSubstring(d_5_currentStr_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "N=")))) > (0)
                        if (d_14_hasNCeq_) and (not(d_13_hasNCeqO_)):
                            if (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "O"))) in ((lm).Tokens):
                                d_17_oValid_: bool
                                out6_: bool
                                out6_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "O")))
                                d_17_oValid_ = out6_
                                if d_17_oValid_:
                                    (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "O"))]), _dafny.BigRational('12e0'))
                        elif (d_15_hasNC_) and (not(d_14_hasNCeq_)):
                            if (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "="))) in ((lm).Tokens):
                                d_18_eqValid_: bool
                                out7_: bool
                                out7_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")))
                                d_18_eqValid_ = out7_
                                if d_18_eqValid_:
                                    (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "="))]), _dafny.BigRational('12e0'))
                        elif (d_16_hasNeq_) and (not(d_15_hasNC_)):
                            if (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "C"))) in ((lm).Tokens):
                                d_19_cValid_: bool
                                out8_: bool
                                out8_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "C")))
                                d_19_cValid_ = out8_
                                if d_19_cValid_:
                                    (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "C"))]), _dafny.BigRational('12e0'))
                        elif not(d_16_hasNeq_):
                            if (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "N"))) in ((lm).Tokens):
                                d_20_nValid_: bool
                                out9_: bool
                                out9_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "N")))
                                d_20_nValid_ = out9_
                                if d_20_nValid_:
                                    d_21_boostAmt_: _dafny.BigRational
                                    if (d_7_spanLen_) < (3):
                                        d_21_boostAmt_ = _dafny.BigRational('2e0')
                                    elif True:
                                        d_21_boostAmt_ = _dafny.BigRational('8e0')
                                    (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "N"))]), d_21_boostAmt_)
                            if ((d_7_spanLen_) < (6)) and ((_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "C"))) in ((lm).Tokens)):
                                d_22_cValid_: bool
                                out10_: bool
                                out10_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "C")))
                                d_22_cValid_ = out10_
                                if d_22_cValid_:
                                    (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "C"))]), _dafny.BigRational('3e0'))
                        d_23_next_: _dafny.Seq
                        out11_: _dafny.Seq
                        out11_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_12_constrainedPrompt_, currentConstrainedOut, eosToken)
                        d_23_next_ = out11_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_23_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_24_valid_: bool
                            out12_: bool
                            out12_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_23_next_)
                            d_24_valid_ = out12_
                            d_25_notComplete_: bool
                            d_25_notComplete_ = not((parser).IsCompletePrefix(currentConstrainedOut))
                            if (d_24_valid_) and (d_25_notComplete_):
                                d_26_ag_: _dafny.Seq
                                d_27_ai_: bool
                                d_28_ac_: _dafny.Seq
                                out13_: _dafny.Seq
                                out14_: bool
                                out15_: _dafny.Seq
                                out13_, out14_, out15_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_23_next_)
                                d_26_ag_ = out13_
                                d_27_ai_ = out14_
                                d_28_ac_ = out15_
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
            out16_: _dafny.Seq
            out17_: bool
            out18_: _dafny.Seq
            out16_, out17_, out18_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_29_closeBudget_)
            d_30_cg_ = out16_
            d_31_ci_ = out17_
            d_32_cc_ = out18_
            generated = d_30_cg_
            insideConstrainedOut = d_31_ci_
            currentConstrainedOut = d_32_cc_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

