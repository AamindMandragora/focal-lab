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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate one valid SMILES for a molecule in the isocyanate class (R-N=C=O). Examples of valid isocyanate SMILES: O=C=NC (methyl isocyanate), O=C=NCC (ethyl isocyanate), O=C=NCCC (propyl isocyanate), O=C=NCCCC (butyl isocyanate), O=C=NC(C)C (isopropyl isocyanate), O=C=NC(C)(C)C (tert-butyl isocyanate), O=C=NC1CC1 (cyclopropyl isocyanate), O=C=NC1CCCC1 (cyclopentyl isocyanate), O=C=NC1CCCCC1 (cyclohexyl isocyanate), O=C=NCc1ccccc1 (benzyl isocyanate), O=C=Nc1ccccc1 (phenyl isocyanate), O=C=Nc1ccc(C)cc1 (4-tolyl isocyanate), O=C=Nc1ccc(Cl)cc1 (4-chlorophenyl isocyanate), O=C=Nc1ccc(F)cc1 (4-fluorophenyl isocyanate), O=C=Nc1ccccn1 (3-pyridyl isocyanate), O=C=NCC=C (allyl isocyanate), O=C=NCC#C (propargyl isocyanate), O=C=NCF (fluoromethyl isocyanate), O=C=NCCF (2-fluoroethyl isocyanate). Output ONLY the SMILES string with no other text.")))
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
        d_5_tokenCount_: int
        d_5_tokenCount_ = 0
        d_6_commonTokens_: _dafny.Seq
        d_6_commonTokens_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "C")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "CC")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "CCC")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "O")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "N"))])
        with _dafny.label("0"):
            while ((d_1_steps_) < (maxSteps)) and (insideConstrainedOut):
                with _dafny.c_label("0"):
                    if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((len(currentConstrainedOut)) >= (5)):
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
                    elif True:
                        d_10_constrainedPrompt_: _dafny.Seq
                        d_10_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_11_next_: _dafny.Seq
                        d_11_next_ = eosToken
                        if (d_5_tokenCount_) < (4):
                            out6_: _dafny.Seq
                            out6_ = (d_0_helpers_).SafePenalizedConstrainedStep(lm, parser, d_10_constrainedPrompt_, currentConstrainedOut, d_6_commonTokens_, _dafny.BigRational('5e0'), eosToken)
                            d_11_next_ = out6_
                        elif (d_5_tokenCount_) < (12):
                            out7_: _dafny.Seq
                            out7_ = (d_0_helpers_).SafeTemperatureConstrainedStep(lm, parser, d_10_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('18e-1'), eosToken)
                            d_11_next_ = out7_
                        elif (d_5_tokenCount_) < (25):
                            out8_: _dafny.Seq
                            out8_ = (d_0_helpers_).SafeTemperatureConstrainedStep(lm, parser, d_10_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('15e-1'), eosToken)
                            d_11_next_ = out8_
                        elif True:
                            out9_: _dafny.Seq
                            out9_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, d_10_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), eosToken)
                            d_11_next_ = out9_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_11_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_12_ag_: _dafny.Seq
                            d_13_ai_: bool
                            d_14_ac_: _dafny.Seq
                            out10_: _dafny.Seq
                            out11_: bool
                            out12_: _dafny.Seq
                            out10_, out11_, out12_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_11_next_)
                            d_12_ag_ = out10_
                            d_13_ai_ = out11_
                            d_14_ac_ = out12_
                            generated = d_12_ag_
                            insideConstrainedOut = d_13_ai_
                            currentConstrainedOut = d_14_ac_
                            d_5_tokenCount_ = (d_5_tokenCount_) + (1)
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_15_closeBudget_: int
            d_15_closeBudget_ = (maxSteps) - (d_1_steps_)
            d_16_cg_: _dafny.Seq
            d_17_ci_: bool
            d_18_cc_: _dafny.Seq
            out13_: _dafny.Seq
            out14_: bool
            out15_: _dafny.Seq
            out13_, out14_, out15_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_15_closeBudget_)
            d_16_cg_ = out13_
            d_17_ci_ = out14_
            d_18_cc_ = out15_
            generated = d_16_cg_
            insideConstrainedOut = d_17_ci_
            currentConstrainedOut = d_18_cc_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

