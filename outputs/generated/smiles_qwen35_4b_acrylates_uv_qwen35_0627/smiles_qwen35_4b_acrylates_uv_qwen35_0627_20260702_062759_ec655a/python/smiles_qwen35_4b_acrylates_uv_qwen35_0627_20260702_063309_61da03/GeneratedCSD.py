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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Task: output one valid SMILES string for a new acrylate molecule. Acrylates contain the substructure C=CC(=O)O (acrylic acid ester or acid). Examples of acrylate SMILES patterns: C=CC(=O)OCC, C=CC(=O)OCCO, C=C(C)C(=O)OC. Output format: <<SMILES>> where SMILES is a valid acrylate SMILES string. Do not copy exemplars from the prompt. Generate a novel acrylate.")))
        d_1_steps_: int
        d_1_steps_ = 0
        if (not(insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
            d_2_preambleBudget_: int
            if (maxSteps) >= (8):
                d_2_preambleBudget_ = 8
            elif True:
                d_2_preambleBudget_ = maxSteps
            d_3_freeGenerated_: _dafny.Seq
            d_4_stoppedOnSpan_: bool
            d_5_stoppedOnEos_: bool
            d_6_stepsUsed1_: int
            out0_: _dafny.Seq
            out1_: bool
            out2_: bool
            out3_: int
            out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_2_preambleBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
            d_3_freeGenerated_ = out0_
            d_4_stoppedOnSpan_ = out1_
            d_5_stoppedOnEos_ = out2_
            d_6_stepsUsed1_ = out3_
            generated = d_3_freeGenerated_
            d_1_steps_ = (d_1_steps_) + (d_6_stepsUsed1_)
            if d_4_stoppedOnSpan_:
                insideConstrainedOut = True
                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
            elif d_5_stoppedOnEos_:
                cost = d_1_steps_
                return generated, insideConstrainedOut, currentConstrainedOut, cost
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_7_closeBudget_: int
            d_7_closeBudget_ = (maxSteps) - (d_1_steps_)
            d_8_cg_: _dafny.Seq
            d_9_ci_: bool
            d_10_cc_: _dafny.Seq
            out4_: _dafny.Seq
            out5_: bool
            out6_: _dafny.Seq
            out4_, out5_, out6_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_7_closeBudget_)
            d_8_cg_ = out4_
            d_9_ci_ = out5_
            d_10_cc_ = out6_
            generated = d_8_cg_
            insideConstrainedOut = d_9_ci_
            currentConstrainedOut = d_10_cc_
            d_1_steps_ = maxSteps
        if (not(insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
            d_11_hasClose_: int
            d_11_hasClose_ = VerifiedDecoderAgent.CSDHelpers.CountSubstring(VerifiedDecoderAgent.CSDHelpers.PrefixToString(generated), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")))
            if (d_11_hasClose_) == (0):
                if (d_1_steps_) < (maxSteps):
                    d_12_og_: _dafny.Seq
                    d_13_oi_: bool
                    d_14_oc_: _dafny.Seq
                    out7_: _dafny.Seq
                    out8_: bool
                    out9_: _dafny.Seq
                    out7_, out8_, out9_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                    d_12_og_ = out7_
                    d_13_oi_ = out8_
                    d_14_oc_ = out9_
                    generated = d_12_og_
                    insideConstrainedOut = d_13_oi_
                    currentConstrainedOut = d_14_oc_
                    d_1_steps_ = (d_1_steps_) + (1)
        with _dafny.label("0"):
            while ((d_1_steps_) < (maxSteps)) and (insideConstrainedOut):
                with _dafny.c_label("0"):
                    if (parser).IsCompletePrefix(currentConstrainedOut):
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
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_18_constrainedPrompt_: _dafny.Seq
                        d_18_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_19_validCount_: int
                        out13_: int
                        out13_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                        d_19_validCount_ = out13_
                        d_20_next_: _dafny.Seq
                        d_20_next_ = eosToken
                        if (d_19_validCount_) <= (15):
                            out14_: _dafny.Seq
                            out14_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, d_18_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('5e0'), eosToken)
                            d_20_next_ = out14_
                        elif True:
                            out15_: _dafny.Seq
                            out15_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_18_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('25e-1'), eosToken)
                            d_20_next_ = out15_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_20_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_21_ag_: _dafny.Seq
                            d_22_ai_: bool
                            d_23_ac_: _dafny.Seq
                            out16_: _dafny.Seq
                            out17_: bool
                            out18_: _dafny.Seq
                            out16_, out17_, out18_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_20_next_)
                            d_21_ag_ = out16_
                            d_22_ai_ = out17_
                            d_23_ac_ = out18_
                            generated = d_21_ag_
                            insideConstrainedOut = d_22_ai_
                            currentConstrainedOut = d_23_ac_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

