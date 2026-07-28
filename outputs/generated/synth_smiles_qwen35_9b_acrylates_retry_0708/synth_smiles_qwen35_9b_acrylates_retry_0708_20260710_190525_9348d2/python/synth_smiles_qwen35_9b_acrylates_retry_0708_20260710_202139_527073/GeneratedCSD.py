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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Task: output exactly ONE valid SMILES string for a molecule in the ACRYLATE class. Acrylate core scaffold: C=CC(=O)O- (acrylic ester) or C=C(C)C(=O)O- (methacrylic ester). Generate diverse acrylates with linear or branched alkyl or heteroatom R-groups. Examples of valid acrylates: C=CC(=O)OCCO, C=CC(=O)OCCCO, C=CC(=O)OCCCCO, C=CC(=O)OCCCCC, C=CC(=O)OCCCCCC, C=CC(=O)OCCCCCCC, C=CC(=O)OCCCCCCCC, C=C(C)C(=O)OCCC, C=C(C)C(=O)OCCCC, C=C(C)C(=O)OCCCCC, C=C(C)C(=O)OCCO, C=C(C)C(=O)OCCCO, C=CC(=O)OCC(C)C, C=CC(=O)OC(C)(C)C, C=CC(=O)OCCC(C)C, C=CC(=O)OCC(C)(C)C, C=CC(=O)OCC(CC)CC, C=CC(=O)OCCOCCO, C=CC(=O)OCC(O)CO, C=CC(=O)OCCN, C=CC(=O)OCCCN, C=CC(=O)OCCOCCOC, C=CC(=O)OCCC(C)(C)C, C=C(C)C(=O)OCC(C)C, C=C(C)C(=O)OC(C)(C)C, C=C(C)C(=O)OCCC(C)C, C=CC(=O)OCCCCCCCCC, C=CC(=O)OCCCCCCCCCC, C=C(C)C(=O)OCCCCCCC, C=CC(=O)OCCOCCOCCO. Output the SMILES string and NOTHING else."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_minLength_: int
        d_3_minLength_ = 14
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
            while ((d_2_steps_) + (1)) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        raise _dafny.Break("0")
                    elif ((parser).IsCompletePrefix(currentConstrainedOut)) and ((len(currentConstrainedOut)) >= (d_3_minLength_)):
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
                        d_11_spanLen_: int
                        d_11_spanLen_ = len(currentConstrainedOut)
                        d_12_next_: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                        if (d_11_spanLen_) < (6):
                            d_13_nextSoft_: _dafny.Seq
                            d_14_softOk_: bool
                            out6_: _dafny.Seq
                            out7_: bool
                            out6_, out7_ = (d_0_helpers_).SafeSoftConstrainedStep(lm, parser, d_10_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('8e0'), eosToken)
                            d_13_nextSoft_ = out6_
                            d_14_softOk_ = out7_
                            d_12_next_ = d_13_nextSoft_
                        elif (d_11_spanLen_) < (22):
                            d_15_nextCG_: _dafny.Seq
                            d_16_wasCG_: bool
                            out8_: _dafny.Seq
                            out9_: bool
                            out8_, out9_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, d_10_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_15_nextCG_ = out8_
                            d_16_wasCG_ = out9_
                            d_12_next_ = d_15_nextCG_
                        elif True:
                            out10_: _dafny.Seq
                            out10_ = (d_0_helpers_).SafeTemperatureConstrainedStep(lm, parser, d_10_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('13e-1'), eosToken)
                            d_12_next_ = out10_
                        d_2_steps_ = (d_2_steps_) + (1)
                        if (d_12_next_) == (eosToken):
                            if (((parser).IsCompletePrefix(currentConstrainedOut)) and ((len(currentConstrainedOut)) >= (d_3_minLength_))) and ((d_2_steps_) < (maxSteps)):
                                d_17_cg_: _dafny.Seq
                                d_18_ci_: bool
                                d_19_cc_: _dafny.Seq
                                out11_: _dafny.Seq
                                out12_: bool
                                out13_: _dafny.Seq
                                out11_, out12_, out13_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_17_cg_ = out11_
                                d_18_ci_ = out12_
                                d_19_cc_ = out13_
                                generated = d_17_cg_
                                insideConstrainedOut = d_18_ci_
                                currentConstrainedOut = d_19_cc_
                                d_2_steps_ = (d_2_steps_) + (1)
                            raise _dafny.Break("0")
                        elif True:
                            d_20_ag_: _dafny.Seq
                            d_21_ai_: bool
                            d_22_ac_: _dafny.Seq
                            out14_: _dafny.Seq
                            out15_: bool
                            out16_: _dafny.Seq
                            out14_, out15_, out16_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_12_next_)
                            d_20_ag_ = out14_
                            d_21_ai_ = out15_
                            d_22_ac_ = out16_
                            generated = d_20_ag_
                            insideConstrainedOut = d_21_ai_
                            currentConstrainedOut = d_22_ac_
                    pass
            pass
        if (((insideConstrainedOut) and ((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) >= (d_3_minLength_))) and ((d_2_steps_) < (maxSteps)):
            d_23_cg_: _dafny.Seq
            d_24_ci_: bool
            d_25_cc_: _dafny.Seq
            out17_: _dafny.Seq
            out18_: bool
            out19_: _dafny.Seq
            out17_, out18_, out19_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
            d_23_cg_ = out17_
            d_24_ci_ = out18_
            d_25_cc_ = out19_
            generated = d_23_cg_
            insideConstrainedOut = d_24_ci_
            currentConstrainedOut = d_25_cc_
            d_2_steps_ = (d_2_steps_) + (1)
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

