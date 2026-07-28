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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Output exactly: SQL: <<SELECT ...>> where the SQL query replaces the ellipsis. The << and >> are literal tokens that must appear in your output."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        if (maxSteps) == (0):
            cost = 0
        elif True:
            d_2_steps_: int
            d_2_steps_ = 0
            d_3_hitEos_: bool
            d_3_hitEos_ = False
            if (not(insideConstrainedOut)) and ((d_2_steps_) < (maxSteps)):
                d_4_chunkBudget_: int
                d_4_chunkBudget_ = 8
                if (d_4_chunkBudget_) > ((maxSteps) - (d_2_steps_)):
                    d_4_chunkBudget_ = (maxSteps) - (d_2_steps_)
                d_5_cg_: _dafny.Seq
                d_6_stoppedOnOpenSpan_: bool
                d_7_stoppedOnEos_: bool
                d_8_stepsUsed_: int
                out0_: _dafny.Seq
                out1_: bool
                out2_: bool
                out3_: int
                out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_4_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                d_5_cg_ = out0_
                d_6_stoppedOnOpenSpan_ = out1_
                d_7_stoppedOnEos_ = out2_
                d_8_stepsUsed_ = out3_
                generated = d_5_cg_
                d_2_steps_ = (d_2_steps_) + (d_8_stepsUsed_)
                if d_7_stoppedOnEos_:
                    d_3_hitEos_ = True
                elif d_6_stoppedOnOpenSpan_:
                    d_9_eg_: _dafny.Seq
                    d_10_ei_: bool
                    d_11_ec_: _dafny.Seq
                    out4_: _dafny.Seq
                    out5_: bool
                    out6_: _dafny.Seq
                    out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                    d_9_eg_ = out4_
                    d_10_ei_ = out5_
                    d_11_ec_ = out6_
                    generated = d_9_eg_
                    insideConstrainedOut = d_10_ei_
                    currentConstrainedOut = d_11_ec_
                elif True:
                    d_12_extraSteps_: int
                    d_12_extraSteps_ = 0
                    while ((((d_12_extraSteps_) < (3)) and ((d_2_steps_) < (maxSteps))) and (not(insideConstrainedOut))) and (not(d_3_hitEos_)):
                        d_13_next_: _dafny.Seq
                        out7_: _dafny.Seq
                        out7_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_13_next_ = out7_
                        d_2_steps_ = (d_2_steps_) + (1)
                        d_12_extraSteps_ = (d_12_extraSteps_) + (1)
                        if (d_13_next_) == (eosToken):
                            d_3_hitEos_ = True
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_13_next_]))
                            if (d_13_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                d_14_eg_: _dafny.Seq
                                d_15_ei_: bool
                                d_16_ec_: _dafny.Seq
                                out8_: _dafny.Seq
                                out9_: bool
                                out10_: _dafny.Seq
                                out8_, out9_, out10_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_14_eg_ = out8_
                                d_15_ei_ = out9_
                                d_16_ec_ = out10_
                                generated = d_14_eg_
                                insideConstrainedOut = d_15_ei_
                                currentConstrainedOut = d_16_ec_
            if ((not(insideConstrainedOut)) and (not(d_3_hitEos_))) and ((d_2_steps_) < (maxSteps)):
                d_17_og_: _dafny.Seq
                d_18_oi_: bool
                d_19_oc_: _dafny.Seq
                out11_: _dafny.Seq
                out12_: bool
                out13_: _dafny.Seq
                out11_, out12_, out13_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                d_17_og_ = out11_
                d_18_oi_ = out12_
                d_19_oc_ = out13_
                generated = d_17_og_
                insideConstrainedOut = d_18_oi_
                currentConstrainedOut = d_19_oc_
                d_2_steps_ = (d_2_steps_) + (1)
            d_20_constrainedTokenCount_: int
            d_20_constrainedTokenCount_ = 0
            d_21_minConstrainedTokens_: int
            d_21_minConstrainedTokens_ = 35
            d_22_earlyPhaseTokens_: int
            d_22_earlyPhaseTokens_ = 12
            with _dafny.label("1_0"):
                while ((d_2_steps_) < (maxSteps)) and (insideConstrainedOut):
                    with _dafny.c_label("1_0"):
                        d_23_remainingBudget_: int
                        d_23_remainingBudget_ = (maxSteps) - (d_2_steps_)
                        d_24_allowClose_: bool
                        d_24_allowClose_ = ((d_20_constrainedTokenCount_) >= (d_21_minConstrainedTokens_)) or ((d_23_remainingBudget_) <= (3))
                        if (d_24_allowClose_) and ((parser).IsCompletePrefix(currentConstrainedOut)):
                            d_25_cg_: _dafny.Seq
                            d_26_ci_: bool
                            d_27_cc_: _dafny.Seq
                            out14_: _dafny.Seq
                            out15_: bool
                            out16_: _dafny.Seq
                            out14_, out15_, out16_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_25_cg_ = out14_
                            d_26_ci_ = out15_
                            d_27_cc_ = out16_
                            generated = d_25_cg_
                            insideConstrainedOut = d_26_ci_
                            currentConstrainedOut = d_27_cc_
                            d_2_steps_ = (d_2_steps_) + (1)
                        elif True:
                            d_28_constrainedPrompt_: _dafny.Seq
                            d_28_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_29_next_: _dafny.Seq
                            d_29_next_ = eosToken
                            if (d_20_constrainedTokenCount_) < (d_22_earlyPhaseTokens_):
                                out17_: _dafny.Seq
                                out17_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, d_28_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('6e0'), eosToken)
                                d_29_next_ = out17_
                            elif True:
                                out18_: _dafny.Seq
                                out18_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_28_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('5e0'), 20, eosToken)
                                d_29_next_ = out18_
                            d_2_steps_ = (d_2_steps_) + (1)
                            if (d_29_next_) == (eosToken):
                                d_30_closeBudget_: int
                                d_30_closeBudget_ = (maxSteps) - (d_2_steps_)
                                if (d_30_closeBudget_) > (0):
                                    d_31_cg_: _dafny.Seq
                                    d_32_ci_: bool
                                    d_33_cc_: _dafny.Seq
                                    out19_: _dafny.Seq
                                    out20_: bool
                                    out21_: _dafny.Seq
                                    out19_, out20_, out21_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_30_closeBudget_)
                                    d_31_cg_ = out19_
                                    d_32_ci_ = out20_
                                    d_33_cc_ = out21_
                                    generated = d_31_cg_
                                    insideConstrainedOut = d_32_ci_
                                    currentConstrainedOut = d_33_cc_
                                    d_2_steps_ = maxSteps
                                raise _dafny.Break("1_0")
                            elif True:
                                d_34_ag_: _dafny.Seq
                                d_35_ai_: bool
                                d_36_ac_: _dafny.Seq
                                out22_: _dafny.Seq
                                out23_: bool
                                out24_: _dafny.Seq
                                out22_, out23_, out24_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_29_next_)
                                d_34_ag_ = out22_
                                d_35_ai_ = out23_
                                d_36_ac_ = out24_
                                generated = d_34_ag_
                                insideConstrainedOut = d_35_ai_
                                currentConstrainedOut = d_36_ac_
                                d_20_constrainedTokenCount_ = (d_20_constrainedTokenCount_) + (1)
                        pass
                pass
            if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
                d_37_closeBudget_: int
                d_37_closeBudget_ = (maxSteps) - (d_2_steps_)
                d_38_cg_: _dafny.Seq
                d_39_ci_: bool
                d_40_cc_: _dafny.Seq
                out25_: _dafny.Seq
                out26_: bool
                out27_: _dafny.Seq
                out25_, out26_, out27_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_37_closeBudget_)
                d_38_cg_ = out25_
                d_39_ci_ = out26_
                d_40_cc_ = out27_
                generated = d_38_cg_
                insideConstrainedOut = d_39_ci_
                currentConstrainedOut = d_40_cc_
                d_2_steps_ = maxSteps
            cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

