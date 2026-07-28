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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate a SMILES string for an isocyanate molecule. The SMILES MUST contain the N=C=O functional group. Isocyanates have the pattern R-N=C=O where R is any organic group. Examples: CCCN=C=O (propyl isocyanate), c1ccc(N=C=O)cc1 (phenyl isocyanate), CC(C)CCN=C=O (isoamyl isocyanate). The token sequence must include N, =, C, =, O representing the isocyanate group N=C=O. Do NOT output just C, O, N, or other single-atom SMILES. The molecule must have the -N=C=O group attached to a carbon chain or ring.")))
        d_2_isocyanateTokens_: _dafny.Seq
        d_2_isocyanateTokens_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "N")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "C")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "O")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "c")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "n"))])
        d_3_chainTokens_: _dafny.Seq
        d_3_chainTokens_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "C")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "c")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "C")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "C")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "C")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "C"))])
        d_4_specialTokens_: _dafny.Seq
        d_4_specialTokens_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "(")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ")")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "[")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "]"))])
        d_5_allBoostGroups_: _dafny.Seq
        d_5_allBoostGroups_ = _dafny.SeqWithoutIsStrInference([d_2_isocyanateTokens_, d_3_chainTokens_, d_4_specialTokens_])
        if (not(insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
            d_6_og_: _dafny.Seq
            d_7_oi_: bool
            d_8_oc_: _dafny.Seq
            out0_: _dafny.Seq
            out1_: bool
            out2_: _dafny.Seq
            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_6_og_ = out0_
            d_7_oi_ = out1_
            d_8_oc_ = out2_
            generated = d_6_og_
            insideConstrainedOut = d_7_oi_
            currentConstrainedOut = d_8_oc_
            d_1_steps_ = (d_1_steps_) + (1)
        d_9_minIsocyanateTokens_: int
        d_9_minIsocyanateTokens_ = 6
        with _dafny.label("0"):
            while ((d_1_steps_) < (maxSteps)) and (insideConstrainedOut):
                with _dafny.c_label("0"):
                    if (parser).IsCompletePrefix(currentConstrainedOut):
                        d_10_countN_: int
                        out3_: int
                        out3_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "N")))
                        d_10_countN_ = out3_
                        d_11_countEq_: int
                        out4_: int
                        out4_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")))
                        d_11_countEq_ = out4_
                        d_12_countO_: int
                        out5_: int
                        out5_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "O")))
                        d_12_countO_ = out5_
                        d_13_spanLen_: int
                        d_13_spanLen_ = len(currentConstrainedOut)
                        d_14_hasIsocyanateSignature_: bool
                        d_14_hasIsocyanateSignature_ = ((((d_10_countN_) >= (1)) and ((d_11_countEq_) >= (2))) and ((d_12_countO_) >= (1))) and ((d_13_spanLen_) >= (d_9_minIsocyanateTokens_))
                        d_15_forcedClose_: bool
                        d_15_forcedClose_ = (d_1_steps_) > (200)
                        if (d_14_hasIsocyanateSignature_) or (d_15_forcedClose_):
                            d_16_cg_: _dafny.Seq
                            d_17_ci_: bool
                            d_18_cc_: _dafny.Seq
                            out6_: _dafny.Seq
                            out7_: bool
                            out8_: _dafny.Seq
                            out6_, out7_, out8_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_16_cg_ = out6_
                            d_17_ci_ = out7_
                            d_18_cc_ = out8_
                            generated = d_16_cg_
                            insideConstrainedOut = d_17_ci_
                            currentConstrainedOut = d_18_cc_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_19_cg_: _dafny.Seq
                            d_20_ci_: bool
                            d_21_cc_: _dafny.Seq
                            out9_: _dafny.Seq
                            out10_: bool
                            out11_: _dafny.Seq
                            out9_, out10_, out11_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_19_cg_ = out9_
                            d_20_ci_ = out10_
                            d_21_cc_ = out11_
                            generated = d_19_cg_
                            insideConstrainedOut = d_20_ci_
                            currentConstrainedOut = d_21_cc_
                            d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_22_constrainedPrompt_: _dafny.Seq
                        d_22_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_23_spanLen2_: int
                        d_23_spanLen2_ = len(currentConstrainedOut)
                        d_24_next_: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                        if (d_23_spanLen2_) < (3):
                            out12_: _dafny.Seq
                            out12_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, d_22_constrainedPrompt_, currentConstrainedOut, d_5_allBoostGroups_, _dafny.BigRational('6e0'), eosToken)
                            d_24_next_ = out12_
                        elif True:
                            out13_: _dafny.Seq
                            out13_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_22_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('15e-1'), eosToken)
                            d_24_next_ = out13_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_24_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_25_ag_: _dafny.Seq
                            d_26_ai_: bool
                            d_27_ac_: _dafny.Seq
                            out14_: _dafny.Seq
                            out15_: bool
                            out16_: _dafny.Seq
                            out14_, out15_, out16_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_24_next_)
                            d_25_ag_ = out14_
                            d_26_ai_ = out15_
                            d_27_ac_ = out16_
                            generated = d_25_ag_
                            insideConstrainedOut = d_26_ai_
                            currentConstrainedOut = d_27_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_28_closeBudget_: int
            d_28_closeBudget_ = (maxSteps) - (d_1_steps_)
            d_29_cg_: _dafny.Seq
            d_30_ci_: bool
            d_31_cc_: _dafny.Seq
            out17_: _dafny.Seq
            out18_: bool
            out19_: _dafny.Seq
            out17_, out18_, out19_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_28_closeBudget_)
            d_29_cg_ = out17_
            d_30_ci_ = out18_
            d_31_cc_ = out19_
            generated = d_29_cg_
            insideConstrainedOut = d_30_ci_
            currentConstrainedOut = d_31_cc_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

