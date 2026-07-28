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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "You are generating a SMILES string for the acrylates chemical class. Acrylates contain the acryloyl ester group: C=CC(=O)O. Generate a novel acrylate SMILES (not just C or CC). Valid acrylate examples: C=CC(=O)OCC (ethyl acrylate), C=CC(=O)OCCCC (butyl acrylate), C=CC(=O)OC(C)C (isopropyl acrylate), C=CC(=O)OCCO (2-hydroxyethyl acrylate). The output MUST start with C=C or contain the C=CC(=O)O pattern. Generate one complete novel acrylate SMILES string."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        if (not(insideConstrainedOut)) and ((d_2_steps_) < (maxSteps)):
            d_3_preambleBudget_: int
            if (maxSteps) >= (150):
                d_3_preambleBudget_ = 100
            elif (maxSteps) >= (50):
                d_3_preambleBudget_ = _dafny.euclidian_division(maxSteps, 2)
            elif (maxSteps) > (1):
                d_3_preambleBudget_ = (maxSteps) - (1)
            elif True:
                d_3_preambleBudget_ = 0
            if (d_3_preambleBudget_) > (0):
                d_4_ug_: _dafny.Seq
                d_5_stoppedOnOpenSpan_: bool
                d_6_stoppedOnEos_: bool
                d_7_stepsUsed_: int
                out0_: _dafny.Seq
                out1_: bool
                out2_: bool
                out3_: int
                out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_3_preambleBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                d_4_ug_ = out0_
                d_5_stoppedOnOpenSpan_ = out1_
                d_6_stoppedOnEos_ = out2_
                d_7_stepsUsed_ = out3_
                generated = d_4_ug_
                d_2_steps_ = (d_2_steps_) + (d_7_stepsUsed_)
                if d_6_stoppedOnEos_:
                    cost = d_2_steps_
                    return generated, insideConstrainedOut, currentConstrainedOut, cost
                if d_5_stoppedOnOpenSpan_:
                    d_8_og_: _dafny.Seq
                    d_9_oi_: bool
                    d_10_oc_: _dafny.Seq
                    out4_: _dafny.Seq
                    out5_: bool
                    out6_: _dafny.Seq
                    out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                    d_8_og_ = out4_
                    d_9_oi_ = out5_
                    d_10_oc_ = out6_
                    generated = d_8_og_
                    insideConstrainedOut = d_9_oi_
                    currentConstrainedOut = d_10_oc_
        if (not(insideConstrainedOut)) and ((d_2_steps_) < (maxSteps)):
            d_11_og_: _dafny.Seq
            d_12_oi_: bool
            d_13_oc_: _dafny.Seq
            out7_: _dafny.Seq
            out8_: bool
            out9_: _dafny.Seq
            out7_, out8_, out9_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_11_og_ = out7_
            d_12_oi_ = out8_
            d_13_oc_ = out9_
            generated = d_11_og_
            insideConstrainedOut = d_12_oi_
            currentConstrainedOut = d_13_oc_
            d_2_steps_ = (d_2_steps_) + (1)
        d_14_acrylateTokens_: _dafny.Seq
        d_14_acrylateTokens_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "C")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "(")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ")")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "O")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "c")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "1")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "2")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "3")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "C(")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "C=")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "O)")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "OC")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=C")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "CC"))])
        with _dafny.label("0"):
            while (d_2_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        raise _dafny.Break("0")
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_15_cg_: _dafny.Seq
                        d_16_ci_: bool
                        d_17_cc_: _dafny.Seq
                        out10_: _dafny.Seq
                        out11_: bool
                        out12_: _dafny.Seq
                        out10_, out11_, out12_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_15_cg_ = out10_
                        d_16_ci_ = out11_
                        d_17_cc_ = out12_
                        generated = d_15_cg_
                        insideConstrainedOut = d_16_ci_
                        currentConstrainedOut = d_17_cc_
                        d_2_steps_ = (d_2_steps_) + (1)
                        raise _dafny.Break("0")
                    elif True:
                        d_18_constrainedPrompt_: _dafny.Seq
                        d_18_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_19_spanLen_: int
                        d_19_spanLen_ = len(currentConstrainedOut)
                        d_20_next_: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                        if (d_19_spanLen_) < (8):
                            out13_: _dafny.Seq
                            out13_ = (d_0_helpers_).SafeBoostedConstrainedStep(lm, parser, d_18_constrainedPrompt_, currentConstrainedOut, d_14_acrylateTokens_, _dafny.BigRational('6e0'), eosToken)
                            d_20_next_ = out13_
                        elif True:
                            out14_: _dafny.Seq
                            out14_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_18_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 15, eosToken)
                            d_20_next_ = out14_
                        d_2_steps_ = (d_2_steps_) + (1)
                        if (d_20_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_21_ag_: _dafny.Seq
                            d_22_ai_: bool
                            d_23_ac_: _dafny.Seq
                            out15_: _dafny.Seq
                            out16_: bool
                            out17_: _dafny.Seq
                            out15_, out16_, out17_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_20_next_)
                            d_21_ag_ = out15_
                            d_22_ai_ = out16_
                            d_23_ac_ = out17_
                            generated = d_21_ag_
                            insideConstrainedOut = d_22_ai_
                            currentConstrainedOut = d_23_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_24_closeBudget_: int
            d_24_closeBudget_ = (maxSteps) - (d_2_steps_)
            d_25_cg_: _dafny.Seq
            d_26_ci_: bool
            d_27_cc_: _dafny.Seq
            out18_: _dafny.Seq
            out19_: bool
            out20_: _dafny.Seq
            out18_, out19_, out20_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_24_closeBudget_)
            d_25_cg_ = out18_
            d_26_ci_ = out19_
            d_27_cc_ = out20_
            generated = d_25_cg_
            insideConstrainedOut = d_26_ci_
            currentConstrainedOut = d_27_cc_
            d_2_steps_ = maxSteps
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

