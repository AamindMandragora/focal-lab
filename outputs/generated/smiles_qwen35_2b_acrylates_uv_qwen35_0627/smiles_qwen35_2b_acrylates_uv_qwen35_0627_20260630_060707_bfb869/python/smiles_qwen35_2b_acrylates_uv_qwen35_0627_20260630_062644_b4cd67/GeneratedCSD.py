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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate a valid SMILES string for an acrylate molecule. Acrylates contain the acryloyl group C=CC(=O)O. Valid acrylate SMILES examples: C=CC(=O)OCC, C=CC(=O)OCCC, C=CC(=O)OC(C)C, C=CC(=O)OCCOC. Output ONLY the SMILES string with no explanation."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_acrylateCoreGroup_: _dafny.Seq
        d_2_acrylateCoreGroup_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "C")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "(")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ")")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "O"))])
        d_3_acrylateGroups_: _dafny.Seq
        d_3_acrylateGroups_ = _dafny.SeqWithoutIsStrInference([d_2_acrylateCoreGroup_])
        d_4_minSpanLen_: int
        d_4_minSpanLen_ = 10
        d_5_steps_: int
        d_5_steps_ = 0
        with _dafny.label("0"):
            while (d_5_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
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
                        d_5_steps_ = (d_5_steps_) + (1)
                    elif True:
                        d_9_spanLen_: int
                        d_9_spanLen_ = len(currentConstrainedOut)
                        d_10_isComplete_: bool
                        d_10_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if (d_10_isComplete_) and ((d_9_spanLen_) >= (d_4_minSpanLen_)):
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
                            d_5_steps_ = (d_5_steps_) + (1)
                            raise _dafny.Break("0")
                        elif d_10_isComplete_:
                            d_14_cg_: _dafny.Seq
                            d_15_ci_: bool
                            d_16_cc_: _dafny.Seq
                            out6_: _dafny.Seq
                            out7_: bool
                            out8_: _dafny.Seq
                            out6_, out7_, out8_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_14_cg_ = out6_
                            d_15_ci_ = out7_
                            d_16_cc_ = out8_
                            generated = d_14_cg_
                            insideConstrainedOut = d_15_ci_
                            currentConstrainedOut = d_16_cc_
                            d_5_steps_ = (d_5_steps_) + (1)
                            raise _dafny.Break("0")
                        elif True:
                            d_17_constrainedPrompt_: _dafny.Seq
                            d_17_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_18_next_: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                            if (d_9_spanLen_) < (15):
                                out9_: _dafny.Seq
                                out9_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, d_17_constrainedPrompt_, currentConstrainedOut, d_3_acrylateGroups_, _dafny.BigRational('8e0'), eosToken)
                                d_18_next_ = out9_
                            elif True:
                                out10_: _dafny.Seq
                                out10_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_17_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                                d_18_next_ = out10_
                            d_5_steps_ = (d_5_steps_) + (1)
                            if (d_18_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_19_isCompleteNow_: bool
                                d_19_isCompleteNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                                if not(d_19_isCompleteNow_):
                                    d_20_ag_: _dafny.Seq
                                    d_21_ai_: bool
                                    d_22_ac_: _dafny.Seq
                                    out11_: _dafny.Seq
                                    out12_: bool
                                    out13_: _dafny.Seq
                                    out11_, out12_, out13_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_18_next_)
                                    d_20_ag_ = out11_
                                    d_21_ai_ = out12_
                                    d_22_ac_ = out13_
                                    generated = d_20_ag_
                                    insideConstrainedOut = d_21_ai_
                                    currentConstrainedOut = d_22_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_5_steps_) < (maxSteps)):
            d_23_isComplete_: bool
            d_23_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
            if d_23_isComplete_:
                d_24_cg_: _dafny.Seq
                d_25_ci_: bool
                d_26_cc_: _dafny.Seq
                out14_: _dafny.Seq
                out15_: bool
                out16_: _dafny.Seq
                out14_, out15_, out16_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_24_cg_ = out14_
                d_25_ci_ = out15_
                d_26_cc_ = out16_
                generated = d_24_cg_
                insideConstrainedOut = d_25_ci_
                currentConstrainedOut = d_26_cc_
                d_5_steps_ = (d_5_steps_) + (1)
        cost = d_5_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

