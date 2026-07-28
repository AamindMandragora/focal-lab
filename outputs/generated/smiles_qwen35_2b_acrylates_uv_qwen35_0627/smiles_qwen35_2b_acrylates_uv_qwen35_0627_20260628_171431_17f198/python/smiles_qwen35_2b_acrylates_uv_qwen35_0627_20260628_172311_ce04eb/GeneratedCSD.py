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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate a single valid SMILES for a novel acrylate ester. Acrylates contain CH2=CH-C(=O)-O-R. Examples: C=CC(=O)OCC, C=CC(=O)OCCO, C=CC(=O)OC(C)C. Output only the SMILES string.")))
        if not(insideConstrainedOut):
            out0_: _dafny.Seq
            out1_: bool
            out2_: _dafny.Seq
            out0_, out1_, out2_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
            generated = out0_
            insideConstrainedOut = out1_
            currentConstrainedOut = out2_
        while ((d_1_steps_) < (maxSteps)) and (insideConstrainedOut):
            d_2_remaining_: int
            d_2_remaining_ = (maxSteps) - (d_1_steps_)
            if (parser).IsCompletePrefix(currentConstrainedOut):
                d_3_cg_: _dafny.Seq
                d_4_ci_: bool
                d_5_cc_: _dafny.Seq
                out3_: _dafny.Seq
                out4_: bool
                out5_: _dafny.Seq
                out3_, out4_, out5_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_3_cg_ = out3_
                d_4_ci_ = out4_
                d_5_cc_ = out5_
                generated = d_3_cg_
                insideConstrainedOut = d_4_ci_
                currentConstrainedOut = d_5_cc_
                d_1_steps_ = (d_1_steps_) + (1)
            elif (d_2_remaining_) <= (5):
                d_6_closeBudget_: int
                d_6_closeBudget_ = d_2_remaining_
                d_7_cg_: _dafny.Seq
                d_8_ci_: bool
                d_9_cc_: _dafny.Seq
                out6_: _dafny.Seq
                out7_: bool
                out8_: _dafny.Seq
                out6_, out7_, out8_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_6_closeBudget_)
                d_7_cg_ = out6_
                d_8_ci_ = out7_
                d_9_cc_ = out8_
                generated = d_7_cg_
                insideConstrainedOut = d_8_ci_
                currentConstrainedOut = d_9_cc_
                d_1_steps_ = maxSteps
            elif True:
                d_10_constrainedPrompt_: _dafny.Seq
                d_10_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                d_11_next_: _dafny.Seq
                out9_: _dafny.Seq
                out9_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_10_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('15e-1'), eosToken)
                d_11_next_ = out9_
                d_1_steps_ = (d_1_steps_) + (1)
                if (d_11_next_) == (eosToken):
                    if (parser).IsCompletePrefix(currentConstrainedOut):
                        d_12_cg_: _dafny.Seq
                        d_13_ci_: bool
                        d_14_cc_: _dafny.Seq
                        out10_: _dafny.Seq
                        out11_: bool
                        out12_: _dafny.Seq
                        out10_, out11_, out12_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_12_cg_ = out10_
                        d_13_ci_ = out11_
                        d_14_cc_ = out12_
                        generated = d_12_cg_
                        insideConstrainedOut = d_13_ci_
                        currentConstrainedOut = d_14_cc_
                    elif (d_1_steps_) < (maxSteps):
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
                elif True:
                    d_19_ag_: _dafny.Seq
                    d_20_ai_: bool
                    d_21_ac_: _dafny.Seq
                    out16_: _dafny.Seq
                    out17_: bool
                    out18_: _dafny.Seq
                    out16_, out17_, out18_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_11_next_)
                    d_19_ag_ = out16_
                    d_20_ai_ = out17_
                    d_21_ac_ = out18_
                    generated = d_19_ag_
                    insideConstrainedOut = d_20_ai_
                    currentConstrainedOut = d_21_ac_
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

