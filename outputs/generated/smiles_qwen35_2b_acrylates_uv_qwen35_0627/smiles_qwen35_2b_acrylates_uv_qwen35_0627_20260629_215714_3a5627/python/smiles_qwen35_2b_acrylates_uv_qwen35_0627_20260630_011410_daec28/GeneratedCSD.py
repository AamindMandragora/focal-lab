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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate a novel acrylate SMILES. Acrylates contain CH2=CH-C(=O)-O or CH2=C(CH3)-C(=O)-O core. Examples of valid acrylates: C=CC(=O)OCCC, C=CC(=O)OCCCC, C=CC(=O)OCC(C)C, C=C(C)C(=O)OCCC, C=CC(=O)OCCCCC, C=CC(=O)OCCOC, C=C(C)C(=O)OCCCC, C=CC(=O)OCC(CC)CCCC. Output only the SMILES string, at least 10 characters long.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_minConstrainedTokens_: int
        d_2_minConstrainedTokens_ = 20
        d_3_maxPreamble_: int
        d_3_maxPreamble_ = 100
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
                        d_9_next_: _dafny.Seq
                        out4_: _dafny.Seq
                        out4_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_9_next_ = out4_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_9_next_) == (eosToken):
                            raise _dafny.Break("1")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_9_next_]))
                    elif True:
                        if (len(currentConstrainedOut)) >= (d_2_minConstrainedTokens_):
                            d_10_cg_: _dafny.Seq
                            d_11_ci_: bool
                            d_12_cc_: _dafny.Seq
                            d_13_closed_: bool
                            out5_: _dafny.Seq
                            out6_: bool
                            out7_: _dafny.Seq
                            out8_: bool
                            out5_, out6_, out7_, out8_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                            d_10_cg_ = out5_
                            d_11_ci_ = out6_
                            d_12_cc_ = out7_
                            d_13_closed_ = out8_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if d_13_closed_:
                                generated = d_10_cg_
                                insideConstrainedOut = d_11_ci_
                                currentConstrainedOut = d_12_cc_
                                raise _dafny.Break("1")
                            elif True:
                                d_14_remaining_: int
                                d_14_remaining_ = (maxSteps) - (d_1_steps_)
                                if (d_14_remaining_) <= (50):
                                    d_15_csg_: _dafny.Seq
                                    d_16_csi_: bool
                                    d_17_csc_: _dafny.Seq
                                    out9_: _dafny.Seq
                                    out10_: bool
                                    out11_: _dafny.Seq
                                    out9_, out10_, out11_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_14_remaining_)
                                    d_15_csg_ = out9_
                                    d_16_csi_ = out10_
                                    d_17_csc_ = out11_
                                    generated = d_15_csg_
                                    insideConstrainedOut = d_16_csi_
                                    currentConstrainedOut = d_17_csc_
                                    d_1_steps_ = maxSteps
                                    raise _dafny.Break("1")
                                elif True:
                                    d_18_constrainedPrompt_: _dafny.Seq
                                    d_18_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                    d_19_penTokens_: _dafny.Seq
                                    d_19_penTokens_ = _dafny.SeqWithoutIsStrInference([])
                                    d_20_next_: _dafny.Seq
                                    out12_: _dafny.Seq
                                    out12_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_18_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('3e0'), d_19_penTokens_, _dafny.BigRational('1e0'), 4, eosToken)
                                    d_20_next_ = out12_
                                    if (d_20_next_) == (eosToken):
                                        raise _dafny.Break("1")
                                    elif True:
                                        d_21_isComplete_: bool
                                        d_21_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                                        if not(d_21_isComplete_):
                                            d_22_ag_: _dafny.Seq
                                            d_23_ai_: bool
                                            d_24_ac_: _dafny.Seq
                                            out13_: _dafny.Seq
                                            out14_: bool
                                            out15_: _dafny.Seq
                                            out13_, out14_, out15_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_20_next_)
                                            d_22_ag_ = out13_
                                            d_23_ai_ = out14_
                                            d_24_ac_ = out15_
                                            generated = d_22_ag_
                                            insideConstrainedOut = d_23_ai_
                                            currentConstrainedOut = d_24_ac_
                        elif True:
                            d_25_constrainedPrompt_: _dafny.Seq
                            d_25_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_26_next_: _dafny.Seq
                            out16_: _dafny.Seq
                            out16_ = (d_0_helpers_).SafeTemperatureConstrainedStep(lm, parser, d_25_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('8e-1'), eosToken)
                            d_26_next_ = out16_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_26_next_) == (eosToken):
                                raise _dafny.Break("1")
                            elif True:
                                d_27_isComplete_: bool
                                d_27_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                                if not(d_27_isComplete_):
                                    d_28_ag_: _dafny.Seq
                                    d_29_ai_: bool
                                    d_30_ac_: _dafny.Seq
                                    out17_: _dafny.Seq
                                    out18_: bool
                                    out19_: _dafny.Seq
                                    out17_, out18_, out19_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_26_next_)
                                    d_28_ag_ = out17_
                                    d_29_ai_ = out18_
                                    d_30_ac_ = out19_
                                    generated = d_28_ag_
                                    insideConstrainedOut = d_29_ai_
                                    currentConstrainedOut = d_30_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_31_remaining2_: int
            d_31_remaining2_ = (maxSteps) - (d_1_steps_)
            d_32_csg2_: _dafny.Seq
            d_33_csi2_: bool
            d_34_csc2_: _dafny.Seq
            out20_: _dafny.Seq
            out21_: bool
            out22_: _dafny.Seq
            out20_, out21_, out22_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_31_remaining2_)
            d_32_csg2_ = out20_
            d_33_csi2_ = out21_
            d_34_csc2_ = out22_
            generated = d_32_csg2_
            insideConstrainedOut = d_33_csi2_
            currentConstrainedOut = d_34_csc2_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

