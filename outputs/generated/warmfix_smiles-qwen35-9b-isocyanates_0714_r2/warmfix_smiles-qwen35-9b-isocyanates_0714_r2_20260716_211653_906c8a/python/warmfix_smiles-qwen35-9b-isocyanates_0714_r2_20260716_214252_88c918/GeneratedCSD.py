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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate exactly one valid SMILES string for an isocyanate molecule (must contain N=C=O functional group). Maximize structural diversity. Prioritize aliphatic structures: O=C=NC (methyl), O=C=NCC (ethyl), O=C=NCCC (propyl), O=C=NCCCC (butyl), O=C=NCCCCC (pentyl), O=C=NC(C)C (isopropyl), O=C=NC(C)(C)C (tert-butyl), O=C=NCC(C)C (isobutyl), O=C=NCC(C)(C)C (neopentyl), O=C=NCF (fluoromethyl), O=C=NCCF (2-fluoroethyl), O=C=NCCCl (3-chloropropyl), O=C=NCCBr (2-bromoethyl), O=C=NCC(F)(F)F (3,3,3-trifluoropropyl), O=C=NCCCC(F)(F)F (4,4,4-trifluorobutyl), O=C=NCC=C (allyl), O=C=NCC#C (propargyl), O=C=NCCO (2-hydroxyethyl), O=C=NCCOC (2-methoxyethyl), O=C=NCCN (2-aminoethyl), O=C=NC1CC1 (cyclopropyl), O=C=NC1CCC1 (cyclobutyl), O=C=NC1CCCC1 (cyclopentyl), O=C=NC1CCCCC1 (cyclohexyl), O=C=NCc1ccccc1 (benzyl), O=C=Nc1ccccc1 (phenyl), O=C=Nc1ccccn1 (3-pyridyl), O=C=Nc1ccc(F)cc1 (4-fluorophenyl), O=C=Nc1ccc(Cl)cc1 (4-chlorophenyl). Output ONLY the SMILES string, nothing else.")))
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
        d_5_minSpanLength_: int
        d_5_minSpanLength_ = 5
        d_6_tokenCount_: int
        d_6_tokenCount_ = 0
        d_7_commonTokens_: _dafny.Seq
        d_7_commonTokens_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "C")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "CC"))])
        with _dafny.label("0"):
            while ((d_1_steps_) < (maxSteps)) and (insideConstrainedOut):
                with _dafny.c_label("0"):
                    if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((len(currentConstrainedOut)) >= (d_5_minSpanLength_)):
                        d_8_smilesStr_: _dafny.Seq
                        d_8_smilesStr_ = VerifiedDecoderAgent.CSDHelpers.PrefixToString(currentConstrainedOut)
                        d_9_hasNCO_: int
                        d_9_hasNCO_ = VerifiedDecoderAgent.CSDHelpers.CountSubstring(d_8_smilesStr_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "N=C=O")))
                        if (d_9_hasNCO_) > (0):
                            if (d_1_steps_) < (maxSteps):
                                d_10_cg_: _dafny.Seq
                                d_11_ci_: bool
                                d_12_cc_: _dafny.Seq
                                out3_: _dafny.Seq
                                out4_: bool
                                out5_: _dafny.Seq
                                out3_, out4_, out5_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_10_cg_ = out3_
                                d_11_ci_ = out4_
                                d_12_cc_ = out5_
                                generated = d_10_cg_
                                insideConstrainedOut = d_11_ci_
                                currentConstrainedOut = d_12_cc_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                raise _dafny.Break("0")
                        elif True:
                            d_13_constrainedPrompt_: _dafny.Seq
                            d_13_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_14_next_: _dafny.Seq
                            out6_: _dafny.Seq
                            out6_ = (d_0_helpers_).SafeTemperatureConstrainedStep(lm, parser, d_13_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('15e-1'), eosToken)
                            d_14_next_ = out6_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_14_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_15_ag_: _dafny.Seq
                                d_16_ai_: bool
                                d_17_ac_: _dafny.Seq
                                out7_: _dafny.Seq
                                out8_: bool
                                out9_: _dafny.Seq
                                out7_, out8_, out9_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_14_next_)
                                d_15_ag_ = out7_
                                d_16_ai_ = out8_
                                d_17_ac_ = out9_
                                generated = d_15_ag_
                                insideConstrainedOut = d_16_ai_
                                currentConstrainedOut = d_17_ac_
                                d_6_tokenCount_ = (d_6_tokenCount_) + (1)
                    elif True:
                        d_18_constrainedPrompt_: _dafny.Seq
                        d_18_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_19_next_: _dafny.Seq
                        d_19_next_ = eosToken
                        if (d_6_tokenCount_) < (6):
                            out10_: _dafny.Seq
                            out10_ = (d_0_helpers_).SafePenalizedConstrainedStep(lm, parser, d_18_constrainedPrompt_, currentConstrainedOut, d_7_commonTokens_, _dafny.BigRational('4e0'), eosToken)
                            d_19_next_ = out10_
                        elif (d_6_tokenCount_) < (20):
                            out11_: _dafny.Seq
                            out11_ = (d_0_helpers_).SafeTemperatureConstrainedStep(lm, parser, d_18_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('17e-1'), eosToken)
                            d_19_next_ = out11_
                        elif True:
                            out12_: _dafny.Seq
                            out12_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, d_18_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), eosToken)
                            d_19_next_ = out12_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_19_next_) == (eosToken):
                            if (((parser).IsCompletePrefix(currentConstrainedOut)) and ((len(currentConstrainedOut)) >= (d_5_minSpanLength_))) and ((d_1_steps_) < (maxSteps)):
                                d_20_smilesStr_: _dafny.Seq
                                d_20_smilesStr_ = VerifiedDecoderAgent.CSDHelpers.PrefixToString(currentConstrainedOut)
                                d_21_hasNCO_: int
                                d_21_hasNCO_ = VerifiedDecoderAgent.CSDHelpers.CountSubstring(d_20_smilesStr_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "N=C=O")))
                                if (d_21_hasNCO_) > (0):
                                    d_22_cg_: _dafny.Seq
                                    d_23_ci_: bool
                                    d_24_cc_: _dafny.Seq
                                    out13_: _dafny.Seq
                                    out14_: bool
                                    out15_: _dafny.Seq
                                    out13_, out14_, out15_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_22_cg_ = out13_
                                    d_23_ci_ = out14_
                                    d_24_cc_ = out15_
                                    generated = d_22_cg_
                                    insideConstrainedOut = d_23_ci_
                                    currentConstrainedOut = d_24_cc_
                                    d_1_steps_ = (d_1_steps_) + (1)
                            raise _dafny.Break("0")
                        elif True:
                            d_25_ag_: _dafny.Seq
                            d_26_ai_: bool
                            d_27_ac_: _dafny.Seq
                            out16_: _dafny.Seq
                            out17_: bool
                            out18_: _dafny.Seq
                            out16_, out17_, out18_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_19_next_)
                            d_25_ag_ = out16_
                            d_26_ai_ = out17_
                            d_27_ac_ = out18_
                            generated = d_25_ag_
                            insideConstrainedOut = d_26_ai_
                            currentConstrainedOut = d_27_ac_
                            d_6_tokenCount_ = (d_6_tokenCount_) + (1)
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_28_closeBudget_: int
            d_28_closeBudget_ = (maxSteps) - (d_1_steps_)
            d_29_cg_: _dafny.Seq
            d_30_ci_: bool
            d_31_cc_: _dafny.Seq
            out19_: _dafny.Seq
            out20_: bool
            out21_: _dafny.Seq
            out19_, out20_, out21_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_28_closeBudget_)
            d_29_cg_ = out19_
            d_30_ci_ = out20_
            d_31_cc_ = out21_
            generated = d_29_cg_
            insideConstrainedOut = d_30_ci_
            currentConstrainedOut = d_31_cc_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

