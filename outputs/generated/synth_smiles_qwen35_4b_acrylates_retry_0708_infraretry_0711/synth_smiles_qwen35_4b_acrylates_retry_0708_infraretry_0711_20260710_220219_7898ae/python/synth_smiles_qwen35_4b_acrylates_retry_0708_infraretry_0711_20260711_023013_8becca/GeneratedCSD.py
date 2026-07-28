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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Task: output one valid SMILES string for a molecule in the acrylate ester class. Acrylates have the acrylic acid ester backbone C=CC(=O)O connected to a carbon-containing R group. Output the SMILES string only."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        if (not(insideConstrainedOut)) and ((d_2_steps_) < (maxSteps)):
            d_3_og_: _dafny.Seq
            d_4_oi_: bool
            d_5_oc_: _dafny.Seq
            out0_: _dafny.Seq
            out1_: bool
            out2_: _dafny.Seq
            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_3_og_ = out0_
            d_4_oi_ = out1_
            d_5_oc_ = out2_
            generated = d_3_og_
            insideConstrainedOut = d_4_oi_
            currentConstrainedOut = d_5_oc_
            d_2_steps_ = (d_2_steps_) + (1)
        d_6_coreTokens_: _dafny.Seq
        d_6_coreTokens_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "C")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "C")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "C")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "(")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "O")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ")")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "O"))])
        d_7_coreIdx_: int
        d_7_coreIdx_ = 0
        with _dafny.label("0"):
            while ((insideConstrainedOut) and ((d_2_steps_) < (maxSteps))) and ((d_7_coreIdx_) < (len(d_6_coreTokens_))):
                with _dafny.c_label("0"):
                    d_8_targetToken_: _dafny.Seq
                    d_8_targetToken_ = (d_6_coreTokens_)[d_7_coreIdx_]
                    d_9_constrainedPrompt_: _dafny.Seq
                    d_9_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                    d_10_targetValid_: bool
                    out3_: bool
                    out3_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_8_targetToken_)
                    d_10_targetValid_ = out3_
                    d_11_next_: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                    if (d_10_targetValid_) and ((d_8_targetToken_) in ((lm).Tokens)):
                        (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([d_8_targetToken_]), _dafny.BigRational('1e1'))
                        out4_: _dafny.Seq
                        out4_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_9_constrainedPrompt_, currentConstrainedOut, eosToken)
                        d_11_next_ = out4_
                    elif True:
                        out5_: _dafny.Seq
                        out5_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_9_constrainedPrompt_, currentConstrainedOut, eosToken)
                        d_11_next_ = out5_
                    d_2_steps_ = (d_2_steps_) + (1)
                    if (d_11_next_) == (eosToken):
                        raise _dafny.Break("0")
                    elif True:
                        d_12_ag_: _dafny.Seq
                        d_13_ai_: bool
                        d_14_ac_: _dafny.Seq
                        out6_: _dafny.Seq
                        out7_: bool
                        out8_: _dafny.Seq
                        out6_, out7_, out8_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_11_next_)
                        d_12_ag_ = out6_
                        d_13_ai_ = out7_
                        d_14_ac_ = out8_
                        generated = d_12_ag_
                        insideConstrainedOut = d_13_ai_
                        currentConstrainedOut = d_14_ac_
                        if (d_11_next_) == (d_8_targetToken_):
                            d_7_coreIdx_ = (d_7_coreIdx_) + (1)
                        elif True:
                            d_7_coreIdx_ = len(d_6_coreTokens_)
                    pass
            pass
        d_15_narrowThreshold_: int
        d_15_narrowThreshold_ = 12
        d_16_minLength_: int
        d_16_minLength_ = 8
        with _dafny.label("1"):
            while (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
                with _dafny.c_label("1"):
                    if not(insideConstrainedOut):
                        raise _dafny.Break("1")
                    elif ((parser).IsCompletePrefix(currentConstrainedOut)) and ((len(currentConstrainedOut)) >= (d_16_minLength_)):
                        d_17_cg_: _dafny.Seq
                        d_18_ci_: bool
                        d_19_cc_: _dafny.Seq
                        out9_: _dafny.Seq
                        out10_: bool
                        out11_: _dafny.Seq
                        out9_, out10_, out11_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_17_cg_ = out9_
                        d_18_ci_ = out10_
                        d_19_cc_ = out11_
                        generated = d_17_cg_
                        insideConstrainedOut = d_18_ci_
                        currentConstrainedOut = d_19_cc_
                        d_2_steps_ = (d_2_steps_) + (1)
                        raise _dafny.Break("1")
                    elif True:
                        d_20_constrainedPrompt_: _dafny.Seq
                        d_20_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_21_next_: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                        if (_dafny.euclidian_modulus(d_2_steps_, 2)) == (0):
                            out12_: _dafny.Seq
                            out12_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_20_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('15e-1'), eosToken)
                            d_21_next_ = out12_
                        elif True:
                            out13_: _dafny.Seq
                            out13_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_20_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('3e0'), d_15_narrowThreshold_, eosToken)
                            d_21_next_ = out13_
                        d_2_steps_ = (d_2_steps_) + (1)
                        if (d_21_next_) == (eosToken):
                            if (((parser).IsCompletePrefix(currentConstrainedOut)) and ((len(currentConstrainedOut)) >= (d_16_minLength_))) and ((d_2_steps_) < (maxSteps)):
                                d_22_cg_: _dafny.Seq
                                d_23_ci_: bool
                                d_24_cc_: _dafny.Seq
                                out14_: _dafny.Seq
                                out15_: bool
                                out16_: _dafny.Seq
                                out14_, out15_, out16_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_22_cg_ = out14_
                                d_23_ci_ = out15_
                                d_24_cc_ = out16_
                                generated = d_22_cg_
                                insideConstrainedOut = d_23_ci_
                                currentConstrainedOut = d_24_cc_
                                d_2_steps_ = (d_2_steps_) + (1)
                            raise _dafny.Break("1")
                        elif True:
                            d_25_ag_: _dafny.Seq
                            d_26_ai_: bool
                            d_27_ac_: _dafny.Seq
                            out17_: _dafny.Seq
                            out18_: bool
                            out19_: _dafny.Seq
                            out17_, out18_, out19_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_21_next_)
                            d_25_ag_ = out17_
                            d_26_ai_ = out18_
                            d_27_ac_ = out19_
                            generated = d_25_ag_
                            insideConstrainedOut = d_26_ai_
                            currentConstrainedOut = d_27_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_28_closeBudget_: int
            d_28_closeBudget_ = (maxSteps) - (d_2_steps_)
            d_29_cg_: _dafny.Seq
            d_30_ci_: bool
            d_31_cc_: _dafny.Seq
            out20_: _dafny.Seq
            out21_: bool
            out22_: _dafny.Seq
            out20_, out21_, out22_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_28_closeBudget_)
            d_29_cg_ = out20_
            d_30_ci_ = out21_
            d_31_cc_ = out22_
            generated = d_29_cg_
            insideConstrainedOut = d_30_ci_
            currentConstrainedOut = d_31_cc_
            d_2_steps_ = maxSteps
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

