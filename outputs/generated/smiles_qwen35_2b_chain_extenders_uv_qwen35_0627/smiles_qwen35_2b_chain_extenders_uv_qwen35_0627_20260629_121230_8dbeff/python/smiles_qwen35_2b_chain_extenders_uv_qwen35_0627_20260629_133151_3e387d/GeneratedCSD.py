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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate ONE valid SMILES for a chain_extender molecule. Chain extenders are SMALL bifunctional molecules with 2-8 carbons: diols like OCCO, OCCCCO, OCCCCCO; diamines like NCCN, NCCCN; amino alcohols like NCCO, NCCCO. Output ONLY the SMILES string, nothing else."))
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
        with _dafny.label("0"):
            while (d_2_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        raise _dafny.Break("0")
                    if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((len(currentConstrainedOut)) >= (4)):
                        d_6_cg_: _dafny.Seq
                        d_7_ci_: bool
                        d_8_cc_: _dafny.Seq
                        out3_: _dafny.Seq
                        out4_: bool
                        out5_: _dafny.Seq
                        out3_, out4_, out5_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_6_cg_ = out3_
                        d_7_ci_ = out4_
                        d_8_cc_ = out5_
                        generated = d_6_cg_
                        insideConstrainedOut = d_7_ci_
                        currentConstrainedOut = d_8_cc_
                        d_2_steps_ = (d_2_steps_) + (1)
                        raise _dafny.Break("0")
                    d_9_constrainedPrompt_: _dafny.Seq
                    d_9_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                    d_10_validCount_: int
                    out6_: int
                    out6_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                    d_10_validCount_ = out6_
                    d_11_next_: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                    if (len(currentConstrainedOut)) < (2):
                        out7_: _dafny.Seq
                        out7_ = (d_0_helpers_).SafeTemperatureConstrainedStep(lm, parser, d_9_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('12e-1'), eosToken)
                        d_11_next_ = out7_
                    elif (d_10_validCount_) <= (8):
                        out8_: _dafny.Seq
                        out8_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, d_9_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('5e0'), eosToken)
                        d_11_next_ = out8_
                    elif True:
                        out9_: _dafny.Seq
                        out9_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_9_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                        d_11_next_ = out9_
                    d_2_steps_ = (d_2_steps_) + (1)
                    if (d_11_next_) == (eosToken):
                        if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_2_steps_) < (maxSteps)):
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
                            d_2_steps_ = (d_2_steps_) + (1)
                        raise _dafny.Break("0")
                    d_15_completeNow_: bool
                    d_15_completeNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                    if d_15_completeNow_:
                        if (d_2_steps_) < (maxSteps):
                            d_16_cg_: _dafny.Seq
                            d_17_ci_: bool
                            d_18_cc_: _dafny.Seq
                            out13_: _dafny.Seq
                            out14_: bool
                            out15_: _dafny.Seq
                            out13_, out14_, out15_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_16_cg_ = out13_
                            d_17_ci_ = out14_
                            d_18_cc_ = out15_
                            generated = d_16_cg_
                            insideConstrainedOut = d_17_ci_
                            currentConstrainedOut = d_18_cc_
                            d_2_steps_ = (d_2_steps_) + (1)
                        raise _dafny.Break("0")
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
                    pass
            pass
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_22_closeBudget_: int
            d_22_closeBudget_ = (maxSteps) - (d_2_steps_)
            d_23_cg_: _dafny.Seq
            d_24_ci_: bool
            d_25_cc_: _dafny.Seq
            out19_: _dafny.Seq
            out20_: bool
            out21_: _dafny.Seq
            out19_, out20_, out21_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_22_closeBudget_)
            d_23_cg_ = out19_
            d_24_ci_ = out20_
            d_25_cc_ = out21_
            generated = d_23_cg_
            insideConstrainedOut = d_24_ci_
            currentConstrainedOut = d_25_cc_
            d_2_steps_ = maxSteps
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

