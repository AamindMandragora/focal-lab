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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate exactly one SMILES string for a novel isocyanate. Isocyanates contain the -N=C=O group. The SMILES must be at least 6 characters long and contain N=C=O. Examples: O=C=NCCBr, O=C=NC1CCCC1, O=C=NCC(F)F. Do NOT output just 'O' or other non-isocyanates. Output only the SMILES string."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        if (not(insideConstrainedOut)) and ((d_2_steps_) < (maxSteps)):
            d_3_og_: _dafny.Seq
            d_4_oi_: bool
            d_5_oc_: _dafny.Seq
            out0_: _dafny.Seq
            out1_: bool
            out2_: _dafny.Seq
            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_3_og_ = out0_
            d_4_oi_ = out1_
            d_5_oc_ = out2_
            generated = d_3_og_
            insideConstrainedOut = d_4_oi_
            currentConstrainedOut = d_5_oc_
            d_2_steps_ = (d_2_steps_) + (1)
        d_6_minForcedTokens_: int
        d_6_minForcedTokens_ = 8
        with _dafny.label("0"):
            while (((d_2_steps_) < (maxSteps)) and (insideConstrainedOut)) and ((len(currentConstrainedOut)) < (d_6_minForcedTokens_)):
                with _dafny.c_label("0"):
                    d_7_constrainedPrompt_: _dafny.Seq
                    d_7_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                    d_8_next_: _dafny.Seq
                    out3_: _dafny.Seq
                    out3_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_7_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('3e0'), eosToken)
                    d_8_next_ = out3_
                    d_2_steps_ = (d_2_steps_) + (1)
                    if (d_8_next_) == (eosToken):
                        raise _dafny.Break("0")
                    elif True:
                        d_9_isComplete_: bool
                        d_9_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if not(d_9_isComplete_):
                            d_10_ag_: _dafny.Seq
                            d_11_ai_: bool
                            d_12_ac_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_8_next_)
                            d_10_ag_ = out4_
                            d_11_ai_ = out5_
                            d_12_ac_ = out6_
                            generated = d_10_ag_
                            insideConstrainedOut = d_11_ai_
                            currentConstrainedOut = d_12_ac_
                    pass
            pass
        with _dafny.label("1"):
            while ((d_2_steps_) < (maxSteps)) and (insideConstrainedOut):
                with _dafny.c_label("1"):
                    d_13_cg_: _dafny.Seq
                    d_14_ci_: bool
                    d_15_cc_: _dafny.Seq
                    d_16_closed_: bool
                    out7_: _dafny.Seq
                    out8_: bool
                    out9_: _dafny.Seq
                    out10_: bool
                    out7_, out8_, out9_, out10_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                    d_13_cg_ = out7_
                    d_14_ci_ = out8_
                    d_15_cc_ = out9_
                    d_16_closed_ = out10_
                    d_2_steps_ = (d_2_steps_) + (1)
                    generated = d_13_cg_
                    insideConstrainedOut = d_14_ci_
                    currentConstrainedOut = d_15_cc_
                    if d_16_closed_:
                        raise _dafny.Break("1")
                    elif True:
                        if (d_2_steps_) < (maxSteps):
                            d_17_constrainedPrompt_: _dafny.Seq
                            d_17_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_18_next_: _dafny.Seq
                            out11_: _dafny.Seq
                            out11_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_17_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('2e0'), eosToken)
                            d_18_next_ = out11_
                            d_2_steps_ = (d_2_steps_) + (1)
                            if (d_18_next_) == (eosToken):
                                raise _dafny.Break("1")
                            elif True:
                                d_19_isComplete2_: bool
                                d_19_isComplete2_ = (parser).IsCompletePrefix(currentConstrainedOut)
                                if not(d_19_isComplete2_):
                                    d_20_ag_: _dafny.Seq
                                    d_21_ai_: bool
                                    d_22_ac_: _dafny.Seq
                                    out12_: _dafny.Seq
                                    out13_: bool
                                    out14_: _dafny.Seq
                                    out12_, out13_, out14_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_18_next_)
                                    d_20_ag_ = out12_
                                    d_21_ai_ = out13_
                                    d_22_ac_ = out14_
                                    generated = d_20_ag_
                                    insideConstrainedOut = d_21_ai_
                                    currentConstrainedOut = d_22_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_23_closeBudget_: int
            d_23_closeBudget_ = (maxSteps) - (d_2_steps_)
            d_24_cg_: _dafny.Seq
            d_25_ci_: bool
            d_26_cc_: _dafny.Seq
            out15_: _dafny.Seq
            out16_: bool
            out17_: _dafny.Seq
            out15_, out16_, out17_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_23_closeBudget_)
            d_24_cg_ = out15_
            d_25_ci_ = out16_
            d_26_cc_ = out17_
            generated = d_24_cg_
            insideConstrainedOut = d_25_ci_
            currentConstrainedOut = d_26_cc_
            d_2_steps_ = maxSteps
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

