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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate exactly ONE unique valid SMILES for an isocyanate (containing N=C=O group). IMPORTANT: Each call must produce a COMPLETELY DIFFERENT molecule. Diverse examples: methyl isocyanate O=C=NC, ethyl O=C=NCC, propyl O=C=NCCC, isopropyl O=C=NC(C)C, tert-butyl O=C=NC(C)(C)C, cyclopropyl O=C=NC1CC1, cyclopentyl O=C=NC1CCCC1, cyclohexyl O=C=NC1CCCCC1, phenyl O=C=Nc1ccccc1, benzyl O=C=NCc1ccccc1, 4-chlorophenyl O=C=Nc1ccc(Cl)cc1, allyl O=C=NCC=C, propargyl O=C=NCC#C, 2-fluoroethyl O=C=NCCF, trifluoromethyl O=C=NC(F)(F)F, pyridin-3-yl O=C=Nc1cccnc1, naphthyl O=C=Nc1cccc2ccccc12. Output ONLY the SMILES string, nothing else.")))
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
        d_5_minSpanLength_ = 4
        d_6_tokenCount_: int
        d_6_tokenCount_ = 0
        d_7_earlyPenalties_: _dafny.Seq
        d_7_earlyPenalties_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "C")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "O"))])
        d_8_diversityBoost_: _dafny.Seq
        d_8_diversityBoost_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "N")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "F")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Cl")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Br")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "c")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "1")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "2")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "(")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "="))])
        with _dafny.label("0"):
            while ((d_1_steps_) < (maxSteps)) and (insideConstrainedOut):
                with _dafny.c_label("0"):
                    if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((len(currentConstrainedOut)) >= (d_5_minSpanLength_)):
                        if (d_1_steps_) < (maxSteps):
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
                            raise _dafny.Break("0")
                    elif True:
                        d_12_constrainedPrompt_: _dafny.Seq
                        d_12_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_13_next_: _dafny.Seq
                        d_13_next_ = eosToken
                        if (d_6_tokenCount_) < (2):
                            out6_: _dafny.Seq
                            out6_ = (d_0_helpers_).SafeBoostedConstrainedStep(lm, parser, d_12_constrainedPrompt_, currentConstrainedOut, d_8_diversityBoost_, _dafny.BigRational('3e0'), eosToken)
                            d_13_next_ = out6_
                        elif (d_6_tokenCount_) < (8):
                            out7_: _dafny.Seq
                            out7_ = (d_0_helpers_).SafeTemperatureConstrainedStep(lm, parser, d_12_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('19e-1'), eosToken)
                            d_13_next_ = out7_
                        elif (d_6_tokenCount_) < (16):
                            if (_dafny.euclidian_modulus(d_6_tokenCount_, 2)) == (0):
                                out8_: _dafny.Seq
                                out8_ = (d_0_helpers_).SafeTemperatureConstrainedStep(lm, parser, d_12_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('17e-1'), eosToken)
                                d_13_next_ = out8_
                            elif True:
                                out9_: _dafny.Seq
                                out9_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, d_12_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), eosToken)
                                d_13_next_ = out9_
                        elif True:
                            out10_: _dafny.Seq
                            out10_ = (d_0_helpers_).SafeTemperatureConstrainedStep(lm, parser, d_12_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('15e-1'), eosToken)
                            d_13_next_ = out10_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_13_next_) == (eosToken):
                            if (((parser).IsCompletePrefix(currentConstrainedOut)) and ((len(currentConstrainedOut)) >= (d_5_minSpanLength_))) and ((d_1_steps_) < (maxSteps)):
                                d_14_cg_: _dafny.Seq
                                d_15_ci_: bool
                                d_16_cc_: _dafny.Seq
                                out11_: _dafny.Seq
                                out12_: bool
                                out13_: _dafny.Seq
                                out11_, out12_, out13_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_14_cg_ = out11_
                                d_15_ci_ = out12_
                                d_16_cc_ = out13_
                                generated = d_14_cg_
                                insideConstrainedOut = d_15_ci_
                                currentConstrainedOut = d_16_cc_
                                d_1_steps_ = (d_1_steps_) + (1)
                            raise _dafny.Break("0")
                        elif True:
                            d_17_ag_: _dafny.Seq
                            d_18_ai_: bool
                            d_19_ac_: _dafny.Seq
                            out14_: _dafny.Seq
                            out15_: bool
                            out16_: _dafny.Seq
                            out14_, out15_, out16_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_13_next_)
                            d_17_ag_ = out14_
                            d_18_ai_ = out15_
                            d_19_ac_ = out16_
                            generated = d_17_ag_
                            insideConstrainedOut = d_18_ai_
                            currentConstrainedOut = d_19_ac_
                            d_6_tokenCount_ = (d_6_tokenCount_) + (1)
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_20_closeBudget_: int
            d_20_closeBudget_ = (maxSteps) - (d_1_steps_)
            d_21_cg_: _dafny.Seq
            d_22_ci_: bool
            d_23_cc_: _dafny.Seq
            out17_: _dafny.Seq
            out18_: bool
            out19_: _dafny.Seq
            out17_, out18_, out19_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_20_closeBudget_)
            d_21_cg_ = out17_
            d_22_ci_ = out18_
            d_23_cc_ = out19_
            generated = d_21_cg_
            insideConstrainedOut = d_22_ci_
            currentConstrainedOut = d_23_cc_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

