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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "You must output a SMILES string for an isocyanate molecule. Isocyanates have the -N=C=O functional group. The SMILES must contain N=C=O or O=C=N as a substructure. Valid examples: O=C=NCCC, O=C=NC1CCCCC1, O=C=NCCBr, O=C=NC(C)(C)C, O=C=NCC#N. DO NOT output water (O). DO NOT output methane (C). The isocyanate group is N=C=O. Output ONLY the SMILES with no other text."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_preambleSize_: int
        d_3_preambleSize_ = 3
        if (not(insideConstrainedOut)) and (((d_2_steps_) + (d_3_preambleSize_)) <= (maxSteps)):
            d_4_gOut_: _dafny.Seq
            d_5_stoppedOpen_: bool
            d_6_stoppedEos_: bool
            d_7_stepsUsed_: int
            out0_: _dafny.Seq
            out1_: bool
            out2_: bool
            out3_: int
            out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_3_preambleSize_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
            d_4_gOut_ = out0_
            d_5_stoppedOpen_ = out1_
            d_6_stoppedEos_ = out2_
            d_7_stepsUsed_ = out3_
            generated = d_4_gOut_
            d_2_steps_ = (d_2_steps_) + (d_7_stepsUsed_)
            if d_5_stoppedOpen_:
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
            elif d_6_stoppedEos_:
                cost = d_2_steps_
                return generated, insideConstrainedOut, currentConstrainedOut, cost
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
        d_14_forcedSteps_: int
        d_14_forcedSteps_ = 0
        d_15_minForced_: int
        d_15_minForced_ = 8
        with _dafny.label("0"):
            while (((d_2_steps_) < (maxSteps)) and (insideConstrainedOut)) and ((d_14_forcedSteps_) < (d_15_minForced_)):
                with _dafny.c_label("0"):
                    d_16_constrainedPrompt_: _dafny.Seq
                    d_16_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                    d_17_penaltyTokens_: _dafny.Seq
                    d_17_penaltyTokens_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "O"))])
                    d_18_next_: _dafny.Seq
                    out10_: _dafny.Seq
                    out10_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_16_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('3e0'), d_17_penaltyTokens_, _dafny.BigRational('6e0'), 12, eosToken)
                    d_18_next_ = out10_
                    d_2_steps_ = (d_2_steps_) + (1)
                    d_14_forcedSteps_ = (d_14_forcedSteps_) + (1)
                    if (d_18_next_) == (eosToken):
                        raise _dafny.Break("0")
                    elif True:
                        d_19_isComplete_: bool
                        d_19_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_19_isComplete_:
                            raise _dafny.Break("0")
                        elif True:
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
        with _dafny.label("1"):
            while ((d_2_steps_) < (maxSteps)) and (insideConstrainedOut):
                with _dafny.c_label("1"):
                    d_23_cg_: _dafny.Seq
                    d_24_ci_: bool
                    d_25_cc_: _dafny.Seq
                    d_26_closed_: bool
                    out14_: _dafny.Seq
                    out15_: bool
                    out16_: _dafny.Seq
                    out17_: bool
                    out14_, out15_, out16_, out17_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                    d_23_cg_ = out14_
                    d_24_ci_ = out15_
                    d_25_cc_ = out16_
                    d_26_closed_ = out17_
                    d_2_steps_ = (d_2_steps_) + (1)
                    generated = d_23_cg_
                    insideConstrainedOut = d_24_ci_
                    currentConstrainedOut = d_25_cc_
                    if d_26_closed_:
                        raise _dafny.Break("1")
                    elif True:
                        if (d_2_steps_) < (maxSteps):
                            d_27_constrainedPrompt_: _dafny.Seq
                            d_27_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_28_next_: _dafny.Seq
                            out18_: _dafny.Seq
                            out18_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_27_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('2e0'), eosToken)
                            d_28_next_ = out18_
                            d_2_steps_ = (d_2_steps_) + (1)
                            if (d_28_next_) == (eosToken):
                                raise _dafny.Break("1")
                            elif True:
                                d_29_isComplete2_: bool
                                d_29_isComplete2_ = (parser).IsCompletePrefix(currentConstrainedOut)
                                if not(d_29_isComplete2_):
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
                    pass
            pass
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_33_closeBudget_: int
            d_33_closeBudget_ = (maxSteps) - (d_2_steps_)
            d_34_cg_: _dafny.Seq
            d_35_ci_: bool
            d_36_cc_: _dafny.Seq
            out22_: _dafny.Seq
            out23_: bool
            out24_: _dafny.Seq
            out22_, out23_, out24_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_33_closeBudget_)
            d_34_cg_ = out22_
            d_35_ci_ = out23_
            d_36_cc_ = out24_
            generated = d_34_cg_
            insideConstrainedOut = d_35_ci_
            currentConstrainedOut = d_36_cc_
            d_2_steps_ = maxSteps
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

