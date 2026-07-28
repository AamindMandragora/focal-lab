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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate a unique SMILES string for an acrylate or methacrylate ester. The SMILES must contain C=CC(=O)O or C=C(C)C(=O)O as the core group. Be creative and diverse: use different alcohol groups (primary, secondary, tertiary, cyclic, aromatic, fluorinated, with ethers, with hydroxy groups). Examples: C=CC(=O)OCC (ethyl acrylate), C=CC(=O)OCCC (propyl acrylate), C=CC(=O)OCCCC (butyl acrylate), C=CC(=O)OC(C)C (isopropyl acrylate), C=CC(=O)OC(C)(C)C (tert-butyl acrylate), C=CC(=O)OCCO (2-hydroxyethyl acrylate), C=CC(=O)OCCOC (2-methoxyethyl acrylate), C=CC(=O)OCC(C)C (isobutyl acrylate), C=CC(=O)OCCCCC (pentyl acrylate), C=C(C)C(=O)OCC (ethyl methacrylate), C=C(C)C(=O)OCCC (propyl methacrylate), C=CC(=O)OCC(O)C (glycidyl acrylate variant), C=CC(=O)OC1CCCCC1 (cyclohexyl acrylate), C=CC(=O)OCc1ccccc1 (benzyl acrylate). Output ONLY the SMILES string with no explanation."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_minAcrylateLen_: int
        d_3_minAcrylateLen_ = 9
        if (not(insideConstrainedOut)) and ((d_2_steps_) < (maxSteps)):
            d_4_og_: _dafny.Seq
            d_5_oi_: bool
            d_6_oc_: _dafny.Seq
            out0_: _dafny.Seq
            out1_: bool
            out2_: _dafny.Seq
            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_4_og_ = out0_
            d_5_oi_ = out1_
            d_6_oc_ = out2_
            generated = d_4_og_
            insideConstrainedOut = d_5_oi_
            currentConstrainedOut = d_6_oc_
            d_2_steps_ = (d_2_steps_) + (1)
        with _dafny.label("0"):
            while (d_2_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        raise _dafny.Break("0")
                    elif ((parser).IsCompletePrefix(currentConstrainedOut)) and ((len(currentConstrainedOut)) >= (d_3_minAcrylateLen_)):
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
                        d_2_steps_ = (d_2_steps_) + (1)
                        raise _dafny.Break("0")
                    elif True:
                        d_10_constrainedPrompt_: _dafny.Seq
                        d_10_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_11_penaltyTokens_: _dafny.Seq
                        d_11_penaltyTokens_ = _dafny.SeqWithoutIsStrInference([eosToken])
                        d_12_next_: _dafny.Seq
                        out6_: _dafny.Seq
                        out6_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_10_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_11_penaltyTokens_, _dafny.BigRational('6e0'), 8, eosToken)
                        d_12_next_ = out6_
                        d_2_steps_ = (d_2_steps_) + (1)
                        if (d_12_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_13_valid_: bool
                            out7_: bool
                            out7_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_12_next_)
                            d_13_valid_ = out7_
                            if d_13_valid_:
                                if not((parser).IsCompletePrefix(currentConstrainedOut)):
                                    d_14_ag_: _dafny.Seq
                                    d_15_ai_: bool
                                    d_16_ac_: _dafny.Seq
                                    out8_: _dafny.Seq
                                    out9_: bool
                                    out10_: _dafny.Seq
                                    out8_, out9_, out10_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_12_next_)
                                    d_14_ag_ = out8_
                                    d_15_ai_ = out9_
                                    d_16_ac_ = out10_
                                    generated = d_14_ag_
                                    insideConstrainedOut = d_15_ai_
                                    currentConstrainedOut = d_16_ac_
                                elif True:
                                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_12_next_]))
                                    currentConstrainedOut = (currentConstrainedOut) + (_dafny.SeqWithoutIsStrInference([d_12_next_]))
                    pass
            pass
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_17_rg_: _dafny.Seq
            d_18_rc_: _dafny.Seq
            out11_: _dafny.Seq
            out12_: _dafny.Seq
            out11_, out12_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
            d_17_rg_ = out11_
            d_18_rc_ = out12_
            if (len(d_18_rc_)) >= (d_3_minAcrylateLen_):
                generated = d_17_rg_
                currentConstrainedOut = d_18_rc_
                d_19_closeBudget_: int
                d_19_closeBudget_ = (maxSteps) - (d_2_steps_)
                if (d_19_closeBudget_) > (0):
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
                    d_2_steps_ = maxSteps
            elif True:
                d_23_closeBudget_: int
                d_23_closeBudget_ = (maxSteps) - (d_2_steps_)
                d_24_cg_: _dafny.Seq
                d_25_ci_: bool
                d_26_cc_: _dafny.Seq
                out16_: _dafny.Seq
                out17_: bool
                out18_: _dafny.Seq
                out16_, out17_, out18_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_23_closeBudget_)
                d_24_cg_ = out16_
                d_25_ci_ = out17_
                d_26_cc_ = out18_
                generated = d_24_cg_
                insideConstrainedOut = d_25_ci_
                currentConstrainedOut = d_26_cc_
                d_2_steps_ = maxSteps
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

