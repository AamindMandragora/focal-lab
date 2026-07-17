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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate exactly one novel acrylate SMILES string. Acrylates MUST contain the substructure C=CC(=O)O (acrylate ester). The SMILES must be at least 12 characters. Valid novel examples: C=CC(=O)OCCC, C=CC(=O)OCCCC, C=CC(=O)OCC(C)C, C=C(C)C(=O)OCCC, C=CC(=O)OCCCCC, C=CC(=O)OCC(CC)CC, C=CC(=O)OCCCCCC, C=CC(=O)OCCOC. Output the SMILES string only.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_minConstrainedTokens_: int
        d_2_minConstrainedTokens_ = 15
        d_3_maxPreamble_: int
        d_3_maxPreamble_ = 80
        d_4_preambleSteps_: int
        d_4_preambleSteps_ = 0
        with _dafny.label("0"):
            while (((d_1_steps_) < (maxSteps)) and ((d_4_preambleSteps_) < (d_3_maxPreamble_))) and (not(insideConstrainedOut)):
                with _dafny.c_label("0"):
                    d_5_next_: _dafny.Seq
                    out0_: _dafny.Seq
                    out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                    d_5_next_ = out0_
                    d_1_steps_ = (d_1_steps_) + (1)
                    d_4_preambleSteps_ = (d_4_preambleSteps_) + (1)
                    if (d_5_next_) == (eosToken):
                        raise _dafny.Break("0")
                    elif True:
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_5_next_]))
                        if (d_5_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                            insideConstrainedOut = True
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    pass
            pass
        if (not(insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
            d_6_og_: _dafny.Seq
            d_7_oi_: bool
            d_8_oc_: _dafny.Seq
            out1_: _dafny.Seq
            out2_: bool
            out3_: _dafny.Seq
            out1_, out2_, out3_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_6_og_ = out1_
            d_7_oi_ = out2_
            d_8_oc_ = out3_
            generated = d_6_og_
            insideConstrainedOut = d_7_oi_
            currentConstrainedOut = d_8_oc_
            d_1_steps_ = (d_1_steps_) + (1)
        with _dafny.label("1"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("1"):
                    if not(insideConstrainedOut):
                        raise _dafny.Break("1")
                    elif (len(currentConstrainedOut)) < (d_2_minConstrainedTokens_):
                        d_9_constrainedPrompt_: _dafny.Seq
                        d_9_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_10_penTokens_: _dafny.Seq
                        d_10_penTokens_ = _dafny.SeqWithoutIsStrInference([])
                        d_11_next_: _dafny.Seq
                        out4_: _dafny.Seq
                        out4_ = (d_0_helpers_).SafeTemperatureConstrainedStep(lm, parser, d_9_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('9e-1'), eosToken)
                        d_11_next_ = out4_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_11_next_) == (eosToken):
                            raise _dafny.Break("1")
                        elif True:
                            d_12_isComplete_: bool
                            d_12_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                            if not(d_12_isComplete_):
                                d_13_ag_: _dafny.Seq
                                d_14_ai_: bool
                                d_15_ac_: _dafny.Seq
                                out5_: _dafny.Seq
                                out6_: bool
                                out7_: _dafny.Seq
                                out5_, out6_, out7_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_11_next_)
                                d_13_ag_ = out5_
                                d_14_ai_ = out6_
                                d_15_ac_ = out7_
                                generated = d_13_ag_
                                insideConstrainedOut = d_14_ai_
                                currentConstrainedOut = d_15_ac_
                    elif True:
                        d_16_isNowComplete_: bool
                        d_16_isNowComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_16_isNowComplete_:
                            d_17_cg_: _dafny.Seq
                            d_18_ci_: bool
                            d_19_cc_: _dafny.Seq
                            d_20_closed_: bool
                            out8_: _dafny.Seq
                            out9_: bool
                            out10_: _dafny.Seq
                            out11_: bool
                            out8_, out9_, out10_, out11_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                            d_17_cg_ = out8_
                            d_18_ci_ = out9_
                            d_19_cc_ = out10_
                            d_20_closed_ = out11_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if d_20_closed_:
                                generated = d_17_cg_
                                insideConstrainedOut = d_18_ci_
                                currentConstrainedOut = d_19_cc_
                                raise _dafny.Break("1")
                        elif True:
                            d_21_constrainedPrompt_: _dafny.Seq
                            d_21_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_22_next_: _dafny.Seq
                            out12_: _dafny.Seq
                            out12_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_21_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('3e0'), _dafny.SeqWithoutIsStrInference([]), _dafny.BigRational('2e0'), 4, eosToken)
                            d_22_next_ = out12_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_22_next_) == (eosToken):
                                raise _dafny.Break("1")
                            elif True:
                                d_23_isComplete2_: bool
                                d_23_isComplete2_ = (parser).IsCompletePrefix(currentConstrainedOut)
                                if not(d_23_isComplete2_):
                                    d_24_ag_: _dafny.Seq
                                    d_25_ai_: bool
                                    d_26_ac_: _dafny.Seq
                                    out13_: _dafny.Seq
                                    out14_: bool
                                    out15_: _dafny.Seq
                                    out13_, out14_, out15_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_22_next_)
                                    d_24_ag_ = out13_
                                    d_25_ai_ = out14_
                                    d_26_ac_ = out15_
                                    generated = d_24_ag_
                                    insideConstrainedOut = d_25_ai_
                                    currentConstrainedOut = d_26_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_27_closeBudget_: int
            d_27_closeBudget_ = (maxSteps) - (d_1_steps_)
            d_28_cg_: _dafny.Seq
            d_29_ci_: bool
            d_30_cc_: _dafny.Seq
            out16_: _dafny.Seq
            out17_: bool
            out18_: _dafny.Seq
            out16_, out17_, out18_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_27_closeBudget_)
            d_28_cg_ = out16_
            d_29_ci_ = out17_
            d_30_cc_ = out18_
            generated = d_28_cg_
            insideConstrainedOut = d_29_ci_
            currentConstrainedOut = d_30_cc_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

