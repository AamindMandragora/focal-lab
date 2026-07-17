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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Task: Generate exactly one novel valid SMILES for an acrylate ester. Acrylates must contain the acryloyl group CH2=CH-C(=O)-O- attached to an ester. The SMILES must start with C=CC(=O)O followed by a carbon chain or ring. Do NOT output single atoms. Output ONLY the full SMILES string for a molecule with at least 6 heavy atoms.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_preambleBudget_: int
        d_2_preambleBudget_ = 3
        d_3_preambleSteps_: int
        d_3_preambleSteps_ = 0
        with _dafny.label("0"):
            while (((d_1_steps_) < (maxSteps)) and (not(insideConstrainedOut))) and ((d_3_preambleSteps_) < (d_2_preambleBudget_)):
                with _dafny.c_label("0"):
                    d_4_next_: _dafny.Seq
                    out0_: _dafny.Seq
                    out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                    d_4_next_ = out0_
                    d_1_steps_ = (d_1_steps_) + (1)
                    d_3_preambleSteps_ = (d_3_preambleSteps_) + (1)
                    if (d_4_next_) == (eosToken):
                        raise _dafny.Break("0")
                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_4_next_]))
                    if (d_4_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                        insideConstrainedOut = True
                        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    pass
            pass
        if (not(insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
            d_5_og_: _dafny.Seq
            d_6_oi_: bool
            d_7_oc_: _dafny.Seq
            out1_: _dafny.Seq
            out2_: bool
            out3_: _dafny.Seq
            out1_, out2_, out3_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_5_og_ = out1_
            d_6_oi_ = out2_
            d_7_oc_ = out3_
            generated = d_5_og_
            insideConstrainedOut = d_6_oi_
            currentConstrainedOut = d_7_oc_
            d_1_steps_ = (d_1_steps_) + (1)
        d_8_constrainedTokenCount_: int
        d_8_constrainedTokenCount_ = 0
        d_9_minConstrainedTokens_: int
        d_9_minConstrainedTokens_ = 10
        with _dafny.label("1"):
            while ((d_1_steps_) < (maxSteps)) and (insideConstrainedOut):
                with _dafny.c_label("1"):
                    d_10_constrainedPrompt_: _dafny.Seq
                    d_10_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                    d_11_isComplete_: bool
                    d_11_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                    if (d_11_isComplete_) and ((d_8_constrainedTokenCount_) >= (d_9_minConstrainedTokens_)):
                        d_12_cg_: _dafny.Seq
                        d_13_ci_: bool
                        d_14_cc_: _dafny.Seq
                        d_15_closed_: bool
                        out4_: _dafny.Seq
                        out5_: bool
                        out6_: _dafny.Seq
                        out7_: bool
                        out4_, out5_, out6_, out7_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                        d_12_cg_ = out4_
                        d_13_ci_ = out5_
                        d_14_cc_ = out6_
                        d_15_closed_ = out7_
                        d_1_steps_ = (d_1_steps_) + (1)
                        generated = d_12_cg_
                        insideConstrainedOut = d_13_ci_
                        currentConstrainedOut = d_14_cc_
                    elif (d_11_isComplete_) and ((d_8_constrainedTokenCount_) < (d_9_minConstrainedTokens_)):
                        d_16_penTokens_: _dafny.Seq
                        d_16_penTokens_ = _dafny.SeqWithoutIsStrInference([eosToken])
                        d_17_next_: _dafny.Seq
                        out8_: _dafny.Seq
                        out8_ = (d_0_helpers_).SafePenalizedConstrainedStep(lm, parser, d_10_constrainedPrompt_, currentConstrainedOut, d_16_penTokens_, _dafny.BigRational('8e0'), eosToken)
                        d_17_next_ = out8_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_17_next_) == (eosToken):
                            d_18_cg_: _dafny.Seq
                            d_19_ci_: bool
                            d_20_cc_: _dafny.Seq
                            d_21_closed_: bool
                            out9_: _dafny.Seq
                            out10_: bool
                            out11_: _dafny.Seq
                            out12_: bool
                            out9_, out10_, out11_, out12_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                            d_18_cg_ = out9_
                            d_19_ci_ = out10_
                            d_20_cc_ = out11_
                            d_21_closed_ = out12_
                            generated = d_18_cg_
                            insideConstrainedOut = d_19_ci_
                            currentConstrainedOut = d_20_cc_
                        elif True:
                            d_22_isCompleteNow_: bool
                            d_22_isCompleteNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                            if not(d_22_isCompleteNow_):
                                d_23_ag_: _dafny.Seq
                                d_24_ai_: bool
                                d_25_ac_: _dafny.Seq
                                out13_: _dafny.Seq
                                out14_: bool
                                out15_: _dafny.Seq
                                out13_, out14_, out15_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_17_next_)
                                d_23_ag_ = out13_
                                d_24_ai_ = out14_
                                d_25_ac_ = out15_
                                generated = d_23_ag_
                                insideConstrainedOut = d_24_ai_
                                currentConstrainedOut = d_25_ac_
                                d_8_constrainedTokenCount_ = (d_8_constrainedTokenCount_) + (1)
                            elif True:
                                d_26_cg_: _dafny.Seq
                                d_27_ci_: bool
                                d_28_cc_: _dafny.Seq
                                d_29_closed_: bool
                                out16_: _dafny.Seq
                                out17_: bool
                                out18_: _dafny.Seq
                                out19_: bool
                                out16_, out17_, out18_, out19_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                                d_26_cg_ = out16_
                                d_27_ci_ = out17_
                                d_28_cc_ = out18_
                                d_29_closed_ = out19_
                                generated = d_26_cg_
                                insideConstrainedOut = d_27_ci_
                                currentConstrainedOut = d_28_cc_
                    elif True:
                        d_30_penTokens_: _dafny.Seq
                        d_30_penTokens_ = _dafny.SeqWithoutIsStrInference([eosToken])
                        d_31_next_: _dafny.Seq
                        out20_: _dafny.Seq
                        out20_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_10_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_30_penTokens_, _dafny.BigRational('6e0'), 12, eosToken)
                        d_31_next_ = out20_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_31_next_) == (eosToken):
                            raise _dafny.Break("1")
                        elif True:
                            d_32_isCompleteNow2_: bool
                            d_32_isCompleteNow2_ = (parser).IsCompletePrefix(currentConstrainedOut)
                            if not(d_32_isCompleteNow2_):
                                d_33_ag_: _dafny.Seq
                                d_34_ai_: bool
                                d_35_ac_: _dafny.Seq
                                out21_: _dafny.Seq
                                out22_: bool
                                out23_: _dafny.Seq
                                out21_, out22_, out23_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_31_next_)
                                d_33_ag_ = out21_
                                d_34_ai_ = out22_
                                d_35_ac_ = out23_
                                generated = d_33_ag_
                                insideConstrainedOut = d_34_ai_
                                currentConstrainedOut = d_35_ac_
                                d_8_constrainedTokenCount_ = (d_8_constrainedTokenCount_) + (1)
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_36_closeBudget_: int
            d_36_closeBudget_ = (maxSteps) - (d_1_steps_)
            d_37_cg_: _dafny.Seq
            d_38_ci_: bool
            d_39_cc_: _dafny.Seq
            out24_: _dafny.Seq
            out25_: bool
            out26_: _dafny.Seq
            out24_, out25_, out26_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_36_closeBudget_)
            d_37_cg_ = out24_
            d_38_ci_ = out25_
            d_39_cc_ = out26_
            generated = d_37_cg_
            insideConstrainedOut = d_38_ci_
            currentConstrainedOut = d_39_cc_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

