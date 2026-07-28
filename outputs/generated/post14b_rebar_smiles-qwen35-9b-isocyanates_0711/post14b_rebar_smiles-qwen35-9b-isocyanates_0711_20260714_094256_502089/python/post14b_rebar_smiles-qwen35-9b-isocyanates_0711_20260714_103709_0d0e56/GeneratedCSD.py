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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate a valid SMILES string for an isocyanate molecule (contains N=C=O group). Generate diverse isocyanates - use different R groups: methyl (CN=C=O), ethyl (CCN=C=O), propyl (CCCN=C=O), isopropyl (CC(C)N=C=O), tert-butyl (CC(C)(C)N=C=O), butyl (CCCCN=C=O), phenyl (O=C=Nc1ccccc1), 4-chlorophenyl (O=C=Nc1ccc(Cl)cc1), 4-methylphenyl (O=C=Nc1ccc(C)cc1), 4-fluorophenyl (O=C=Nc1ccc(F)cc1), 2-naphthyl, cyclohexyl (O=C=NC1CCCCC1), benzyl (O=C=NCc1ccccc1). Aromatic rings MUST be fully closed.")))
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
        d_5_constrainedTokenCount_: int
        d_5_constrainedTokenCount_ = 0
        d_6_minTokens_: int
        d_6_minTokens_ = 8
        with _dafny.label("0"):
            while ((d_1_steps_) < (maxSteps)) and (insideConstrainedOut):
                with _dafny.c_label("0"):
                    if ((d_5_constrainedTokenCount_) >= (d_6_minTokens_)) and ((parser).IsCompletePrefix(currentConstrainedOut)):
                        if (d_1_steps_) < (maxSteps):
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
                        raise _dafny.Break("0")
                    elif True:
                        d_10_penaltyTokens_: _dafny.Seq
                        d_10_penaltyTokens_ = currentConstrainedOut
                        d_11_constrainedPrompt_: _dafny.Seq
                        d_11_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_12_next_: _dafny.Seq
                        out6_: _dafny.Seq
                        out6_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_11_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_10_penaltyTokens_, _dafny.BigRational('3e0'), 12, eosToken)
                        d_12_next_ = out6_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_12_next_) == (eosToken):
                            if (((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_5_constrainedTokenCount_) >= (d_6_minTokens_))) and ((d_1_steps_) < (maxSteps)):
                                d_13_cg_: _dafny.Seq
                                d_14_ci_: bool
                                d_15_cc_: _dafny.Seq
                                out7_: _dafny.Seq
                                out8_: bool
                                out9_: _dafny.Seq
                                out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_13_cg_ = out7_
                                d_14_ci_ = out8_
                                d_15_cc_ = out9_
                                generated = d_13_cg_
                                insideConstrainedOut = d_14_ci_
                                currentConstrainedOut = d_15_cc_
                                d_1_steps_ = (d_1_steps_) + (1)
                            raise _dafny.Break("0")
                        elif True:
                            d_16_ag_: _dafny.Seq
                            d_17_ai_: bool
                            d_18_ac_: _dafny.Seq
                            out10_: _dafny.Seq
                            out11_: bool
                            out12_: _dafny.Seq
                            out10_, out11_, out12_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_12_next_)
                            d_16_ag_ = out10_
                            d_17_ai_ = out11_
                            d_18_ac_ = out12_
                            generated = d_16_ag_
                            insideConstrainedOut = d_17_ai_
                            currentConstrainedOut = d_18_ac_
                            d_5_constrainedTokenCount_ = (d_5_constrainedTokenCount_) + (1)
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_19_closeBudget_: int
            d_19_closeBudget_ = (maxSteps) - (d_1_steps_)
            d_20_cg_: _dafny.Seq
            d_21_ci_: bool
            d_22_cc_: _dafny.Seq
            out13_: _dafny.Seq
            out14_: bool
            out15_: _dafny.Seq
            out13_, out14_, out15_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_19_closeBudget_)
            d_20_cg_ = out13_
            d_21_ci_ = out14_
            d_22_cc_ = out15_
            generated = d_20_cg_
            insideConstrainedOut = d_21_ci_
            currentConstrainedOut = d_22_cc_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

