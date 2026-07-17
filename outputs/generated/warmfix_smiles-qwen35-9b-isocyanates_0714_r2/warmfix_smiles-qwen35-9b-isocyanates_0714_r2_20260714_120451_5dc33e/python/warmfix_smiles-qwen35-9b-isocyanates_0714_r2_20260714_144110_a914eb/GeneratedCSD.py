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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate exactly one valid SMILES for an isocyanate (R-N=C=O) molecule. Output ONLY the SMILES string. Require: N=C=O substructure. Generate unusual and diverse structures. Try fluorinated, chlorinated, aromatic, cyclic, branched, long-chain, bicyclic, or heterocyclic isocyanates. Avoid the most common examples like CCN=C=O or phenyl isocyanate. Be creative and diverse.")))
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
        d_5_diversityBoostTokens_: _dafny.Seq
        d_5_diversityBoostTokens_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "c")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "1")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "2")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "F")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Cl")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Br")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "n")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "S")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "N")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "(")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "CC")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "CCC"))])
        d_6_penalizeTokens_: _dafny.Seq
        d_6_penalizeTokens_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "C")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "O"))])
        d_7_minSpanLength_: int
        d_7_minSpanLength_ = 6
        d_8_tokenCount_: int
        d_8_tokenCount_ = 0
        with _dafny.label("0"):
            while ((d_1_steps_) < (maxSteps)) and (insideConstrainedOut):
                with _dafny.c_label("0"):
                    if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((len(currentConstrainedOut)) >= (d_7_minSpanLength_)):
                        d_9_smilesStr_: _dafny.Seq
                        d_9_smilesStr_ = VerifiedDecoderAgent.CSDHelpers.PrefixToString(currentConstrainedOut)
                        d_10_hasNCO_: int
                        d_10_hasNCO_ = VerifiedDecoderAgent.CSDHelpers.CountSubstring(d_9_smilesStr_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "N=C=O")))
                        if (d_10_hasNCO_) > (0):
                            if (d_1_steps_) < (maxSteps):
                                d_11_cg_: _dafny.Seq
                                d_12_ci_: bool
                                d_13_cc_: _dafny.Seq
                                out3_: _dafny.Seq
                                out4_: bool
                                out5_: _dafny.Seq
                                out3_, out4_, out5_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_11_cg_ = out3_
                                d_12_ci_ = out4_
                                d_13_cc_ = out5_
                                generated = d_11_cg_
                                insideConstrainedOut = d_12_ci_
                                currentConstrainedOut = d_13_cc_
                                d_1_steps_ = (d_1_steps_) + (1)
                            raise _dafny.Break("0")
                        elif True:
                            d_14_constrainedPrompt_: _dafny.Seq
                            d_14_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_15_next_: _dafny.Seq
                            out6_: _dafny.Seq
                            out6_ = (d_0_helpers_).SafeTemperatureConstrainedStep(lm, parser, d_14_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('18e-1'), eosToken)
                            d_15_next_ = out6_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_15_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_16_ag_: _dafny.Seq
                                d_17_ai_: bool
                                d_18_ac_: _dafny.Seq
                                out7_: _dafny.Seq
                                out8_: bool
                                out9_: _dafny.Seq
                                out7_, out8_, out9_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_15_next_)
                                d_16_ag_ = out7_
                                d_17_ai_ = out8_
                                d_18_ac_ = out9_
                                generated = d_16_ag_
                                insideConstrainedOut = d_17_ai_
                                currentConstrainedOut = d_18_ac_
                                d_8_tokenCount_ = (d_8_tokenCount_) + (1)
                    elif True:
                        d_19_constrainedPrompt_: _dafny.Seq
                        d_19_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_20_next_: _dafny.Seq
                        d_20_next_ = eosToken
                        if (d_8_tokenCount_) < (4):
                            (d_0_helpers_).SafeBoostTokenLogits(lm, d_5_diversityBoostTokens_, _dafny.BigRational('5e0'))
                            (d_0_helpers_).SafePenalizeTokenLogits(lm, d_6_penalizeTokens_, _dafny.BigRational('3e0'))
                            out10_: _dafny.Seq
                            out10_ = (d_0_helpers_).SafeTemperatureConstrainedStep(lm, parser, d_19_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('18e-1'), eosToken)
                            d_20_next_ = out10_
                        elif (d_8_tokenCount_) < (12):
                            out11_: _dafny.Seq
                            out11_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_19_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('3e0'), d_6_penalizeTokens_, _dafny.BigRational('2e0'), 15, eosToken)
                            d_20_next_ = out11_
                        elif True:
                            out12_: _dafny.Seq
                            out12_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_19_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                            d_20_next_ = out12_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_20_next_) == (eosToken):
                            if (((parser).IsCompletePrefix(currentConstrainedOut)) and ((len(currentConstrainedOut)) >= (d_7_minSpanLength_))) and ((d_1_steps_) < (maxSteps)):
                                d_21_smilesStr_: _dafny.Seq
                                d_21_smilesStr_ = VerifiedDecoderAgent.CSDHelpers.PrefixToString(currentConstrainedOut)
                                d_22_hasNCO_: int
                                d_22_hasNCO_ = VerifiedDecoderAgent.CSDHelpers.CountSubstring(d_21_smilesStr_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "N=C=O")))
                                if (d_22_hasNCO_) > (0):
                                    d_23_cg_: _dafny.Seq
                                    d_24_ci_: bool
                                    d_25_cc_: _dafny.Seq
                                    out13_: _dafny.Seq
                                    out14_: bool
                                    out15_: _dafny.Seq
                                    out13_, out14_, out15_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_23_cg_ = out13_
                                    d_24_ci_ = out14_
                                    d_25_cc_ = out15_
                                    generated = d_23_cg_
                                    insideConstrainedOut = d_24_ci_
                                    currentConstrainedOut = d_25_cc_
                                    d_1_steps_ = (d_1_steps_) + (1)
                            raise _dafny.Break("0")
                        elif True:
                            d_26_ag_: _dafny.Seq
                            d_27_ai_: bool
                            d_28_ac_: _dafny.Seq
                            out16_: _dafny.Seq
                            out17_: bool
                            out18_: _dafny.Seq
                            out16_, out17_, out18_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_20_next_)
                            d_26_ag_ = out16_
                            d_27_ai_ = out17_
                            d_28_ac_ = out18_
                            generated = d_26_ag_
                            insideConstrainedOut = d_27_ai_
                            currentConstrainedOut = d_28_ac_
                            d_8_tokenCount_ = (d_8_tokenCount_) + (1)
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_29_closeBudget_: int
            d_29_closeBudget_ = (maxSteps) - (d_1_steps_)
            d_30_cg_: _dafny.Seq
            d_31_ci_: bool
            d_32_cc_: _dafny.Seq
            out19_: _dafny.Seq
            out20_: bool
            out21_: _dafny.Seq
            out19_, out20_, out21_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_29_closeBudget_)
            d_30_cg_ = out19_
            d_31_ci_ = out20_
            d_32_cc_ = out21_
            generated = d_30_cg_
            insideConstrainedOut = d_31_ci_
            currentConstrainedOut = d_32_cc_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

