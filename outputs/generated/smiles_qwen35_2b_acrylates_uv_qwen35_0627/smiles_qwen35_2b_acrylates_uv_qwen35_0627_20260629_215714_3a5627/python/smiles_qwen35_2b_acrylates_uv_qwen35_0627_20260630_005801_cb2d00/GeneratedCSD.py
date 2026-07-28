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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "CRITICAL TASK: Generate ONE novel acrylate ester SMILES string. MANDATORY: The SMILES MUST contain the acrylate core C=CC(=O)O- where the oxygen is esterified with an alkyl group R. REQUIRED PATTERN: starts with C=CC(=O)O then an alkyl group. CORRECT EXAMPLES (use these patterns): C=CC(=O)OCCC, C=CC(=O)OCCCC, C=CC(=O)OCC(C)C, C=C(C)C(=O)OCCC, C=CC(=O)OCCCCC, C=CC(=O)OCCOC, C=CC(=O)OCC(CC)CCC. DO NOT generate single atoms like C or CC. The SMILES must be at least 12 characters long and contain the sequence C=CC(=O)O.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_minConstrainedTokens_: int
        d_2_minConstrainedTokens_ = 10
        d_3_maxPreamble_: int
        d_3_maxPreamble_ = 60
        d_4_preambleSteps_: int
        d_4_preambleSteps_ = 0
        d_5_maxRetries_: int
        d_5_maxRetries_ = 8
        d_6_retryCount_: int
        d_6_retryCount_ = 0
        with _dafny.label("0"):
            while (((d_1_steps_) < (maxSteps)) and ((d_4_preambleSteps_) < (d_3_maxPreamble_))) and (not(insideConstrainedOut)):
                with _dafny.c_label("0"):
                    d_7_next_: _dafny.Seq
                    out0_: _dafny.Seq
                    out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                    d_7_next_ = out0_
                    d_1_steps_ = (d_1_steps_) + (1)
                    d_4_preambleSteps_ = (d_4_preambleSteps_) + (1)
                    if (d_7_next_) == (eosToken):
                        raise _dafny.Break("0")
                    elif True:
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_7_next_]))
                        if (d_7_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                            insideConstrainedOut = True
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    pass
            pass
        if (not(insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
            d_8_og_: _dafny.Seq
            d_9_oi_: bool
            d_10_oc_: _dafny.Seq
            out1_: _dafny.Seq
            out2_: bool
            out3_: _dafny.Seq
            out1_, out2_, out3_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_8_og_ = out1_
            d_9_oi_ = out2_
            d_10_oc_ = out3_
            generated = d_8_og_
            insideConstrainedOut = d_9_oi_
            currentConstrainedOut = d_10_oc_
            d_1_steps_ = (d_1_steps_) + (1)
        with _dafny.label("1"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("1"):
                    if not(insideConstrainedOut):
                        d_11_next_: _dafny.Seq
                        out4_: _dafny.Seq
                        out4_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_11_next_ = out4_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_11_next_) == (eosToken):
                            raise _dafny.Break("1")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_11_next_]))
                    elif True:
                        d_12_isComplete_: bool
                        d_12_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if (d_12_isComplete_) and ((len(currentConstrainedOut)) >= (d_2_minConstrainedTokens_)):
                            d_13_cg_: _dafny.Seq
                            d_14_ci_: bool
                            d_15_cc_: _dafny.Seq
                            out5_: _dafny.Seq
                            out6_: bool
                            out7_: _dafny.Seq
                            out5_, out6_, out7_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_13_cg_ = out5_
                            d_14_ci_ = out6_
                            d_15_cc_ = out7_
                            generated = d_13_cg_
                            insideConstrainedOut = d_14_ci_
                            currentConstrainedOut = d_15_cc_
                            d_1_steps_ = (d_1_steps_) + (1)
                            raise _dafny.Break("1")
                        elif (((d_12_isComplete_) and ((len(currentConstrainedOut)) < (d_2_minConstrainedTokens_))) and ((d_6_retryCount_) < (d_5_maxRetries_))) and (((d_1_steps_) + (2)) <= (maxSteps)):
                            d_16_cg_: _dafny.Seq
                            d_17_ci_: bool
                            d_18_cc_: _dafny.Seq
                            out8_: _dafny.Seq
                            out9_: bool
                            out10_: _dafny.Seq
                            out8_, out9_, out10_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_16_cg_ = out8_
                            d_17_ci_ = out9_
                            d_18_cc_ = out10_
                            generated = d_16_cg_
                            insideConstrainedOut = d_17_ci_
                            currentConstrainedOut = d_18_cc_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_19_og2_: _dafny.Seq
                            d_20_oi2_: bool
                            d_21_oc2_: _dafny.Seq
                            out11_: _dafny.Seq
                            out12_: bool
                            out13_: _dafny.Seq
                            out11_, out12_, out13_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_19_og2_ = out11_
                            d_20_oi2_ = out12_
                            d_21_oc2_ = out13_
                            generated = d_19_og2_
                            insideConstrainedOut = d_20_oi2_
                            currentConstrainedOut = d_21_oc2_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_6_retryCount_ = (d_6_retryCount_) + (1)
                        elif (d_12_isComplete_) and ((len(currentConstrainedOut)) < (d_2_minConstrainedTokens_)):
                            d_22_cg_: _dafny.Seq
                            d_23_ci_: bool
                            d_24_cc_: _dafny.Seq
                            d_25_closed_: bool
                            out14_: _dafny.Seq
                            out15_: bool
                            out16_: _dafny.Seq
                            out17_: bool
                            out14_, out15_, out16_, out17_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                            d_22_cg_ = out14_
                            d_23_ci_ = out15_
                            d_24_cc_ = out16_
                            d_25_closed_ = out17_
                            d_1_steps_ = (d_1_steps_) + (1)
                            generated = d_22_cg_
                            insideConstrainedOut = d_23_ci_
                            currentConstrainedOut = d_24_cc_
                            raise _dafny.Break("1")
                        elif True:
                            if (len(currentConstrainedOut)) >= (d_2_minConstrainedTokens_):
                                d_26_constrainedPrompt_: _dafny.Seq
                                d_26_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_27_boostToks_: _dafny.Seq
                                d_27_boostToks_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "(")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ")")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "O")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "C"))])
                                d_28_next_: _dafny.Seq
                                out18_: _dafny.Seq
                                out18_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_26_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('3e0'), d_27_boostToks_, _dafny.BigRational('1e0'), 4, eosToken)
                                d_28_next_ = out18_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_28_next_) == (eosToken):
                                    raise _dafny.Break("1")
                                elif True:
                                    d_29_isCompleteNow_: bool
                                    d_29_isCompleteNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                                    if not(d_29_isCompleteNow_):
                                        d_30_ag_: _dafny.Seq
                                        d_31_ai_: bool
                                        d_32_ac_: _dafny.Seq
                                        out19_: _dafny.Seq
                                        out20_: bool
                                        out21_: _dafny.Seq
                                        out19_, out20_, out21_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_28_next_)
                                        d_30_ag_ = out19_
                                        d_31_ai_ = out20_
                                        d_32_ac_ = out21_
                                        generated = d_30_ag_
                                        insideConstrainedOut = d_31_ai_
                                        currentConstrainedOut = d_32_ac_
                            elif True:
                                d_33_constrainedPrompt_: _dafny.Seq
                                d_33_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_34_structToks_: _dafny.Seq
                                d_34_structToks_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "C")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "(")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ")")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "O"))])
                                d_35_next_: _dafny.Seq
                                out22_: _dafny.Seq
                                out22_ = (d_0_helpers_).SafeBoostedConstrainedStep(lm, parser, d_33_constrainedPrompt_, currentConstrainedOut, d_34_structToks_, _dafny.BigRational('3e0'), eosToken)
                                d_35_next_ = out22_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_35_next_) == (eosToken):
                                    raise _dafny.Break("1")
                                elif True:
                                    d_36_isCompleteNow_: bool
                                    d_36_isCompleteNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                                    if not(d_36_isCompleteNow_):
                                        d_37_ag_: _dafny.Seq
                                        d_38_ai_: bool
                                        d_39_ac_: _dafny.Seq
                                        out23_: _dafny.Seq
                                        out24_: bool
                                        out25_: _dafny.Seq
                                        out23_, out24_, out25_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_35_next_)
                                        d_37_ag_ = out23_
                                        d_38_ai_ = out24_
                                        d_39_ac_ = out25_
                                        generated = d_37_ag_
                                        insideConstrainedOut = d_38_ai_
                                        currentConstrainedOut = d_39_ac_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

