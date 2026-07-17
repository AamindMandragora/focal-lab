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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Output the SQL query between << and >> markers. Start with << immediately, then write the complete SQL query, then close with >>."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        if (maxSteps) == (0):
            cost = 0
        elif True:
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
            d_6_constrainedTokenCount_: int
            d_6_constrainedTokenCount_ = 0
            d_7_minConstrainedTokens_: int
            d_7_minConstrainedTokens_ = 15
            d_8_earlyPhaseTokens_: int
            d_8_earlyPhaseTokens_ = 10
            d_9_midPhaseTokens_: int
            d_9_midPhaseTokens_ = 50
            while ((d_2_steps_) < (maxSteps)) and (insideConstrainedOut):
                d_10_remainingBudget_: int
                d_10_remainingBudget_ = (maxSteps) - (d_2_steps_)
                d_11_allowClose_: bool
                d_11_allowClose_ = ((d_6_constrainedTokenCount_) >= (d_7_minConstrainedTokens_)) or ((d_10_remainingBudget_) <= (5))
                if (d_11_allowClose_) and ((parser).IsCompletePrefix(currentConstrainedOut)):
                    d_12_cg_: _dafny.Seq
                    d_13_ci_: bool
                    d_14_cc_: _dafny.Seq
                    out3_: _dafny.Seq
                    out4_: bool
                    out5_: _dafny.Seq
                    out3_, out4_, out5_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                    d_12_cg_ = out3_
                    d_13_ci_ = out4_
                    d_14_cc_ = out5_
                    generated = d_12_cg_
                    insideConstrainedOut = d_13_ci_
                    currentConstrainedOut = d_14_cc_
                    d_2_steps_ = (d_2_steps_) + (1)
                elif True:
                    d_15_constrainedPrompt_: _dafny.Seq
                    d_15_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                    d_16_next_: _dafny.Seq
                    d_16_next_ = eosToken
                    if (d_6_constrainedTokenCount_) < (d_8_earlyPhaseTokens_):
                        out6_: _dafny.Seq
                        out6_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, d_15_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('8e0'), eosToken)
                        d_16_next_ = out6_
                    elif (d_6_constrainedTokenCount_) < (d_9_midPhaseTokens_):
                        out7_: _dafny.Seq
                        out7_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_15_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('6e0'), 25, eosToken)
                        d_16_next_ = out7_
                    elif True:
                        out8_: _dafny.Seq
                        out8_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_15_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 15, eosToken)
                        d_16_next_ = out8_
                    d_2_steps_ = (d_2_steps_) + (1)
                    if (d_16_next_) == (eosToken):
                        d_17_closeBudget_: int
                        d_17_closeBudget_ = (maxSteps) - (d_2_steps_)
                        if (d_17_closeBudget_) > (0):
                            d_18_cg_: _dafny.Seq
                            d_19_ci_: bool
                            d_20_cc_: _dafny.Seq
                            out9_: _dafny.Seq
                            out10_: bool
                            out11_: _dafny.Seq
                            out9_, out10_, out11_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_17_closeBudget_)
                            d_18_cg_ = out9_
                            d_19_ci_ = out10_
                            d_20_cc_ = out11_
                            generated = d_18_cg_
                            insideConstrainedOut = d_19_ci_
                            currentConstrainedOut = d_20_cc_
                            d_2_steps_ = maxSteps
                    elif True:
                        d_21_ag_: _dafny.Seq
                        d_22_ai_: bool
                        d_23_ac_: _dafny.Seq
                        out12_: _dafny.Seq
                        out13_: bool
                        out14_: _dafny.Seq
                        out12_, out13_, out14_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_16_next_)
                        d_21_ag_ = out12_
                        d_22_ai_ = out13_
                        d_23_ac_ = out14_
                        generated = d_21_ag_
                        insideConstrainedOut = d_22_ai_
                        currentConstrainedOut = d_23_ac_
                        d_6_constrainedTokenCount_ = (d_6_constrainedTokenCount_) + (1)
            if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
                d_24_closeBudget_: int
                d_24_closeBudget_ = (maxSteps) - (d_2_steps_)
                d_25_cg_: _dafny.Seq
                d_26_ci_: bool
                d_27_cc_: _dafny.Seq
                out15_: _dafny.Seq
                out16_: bool
                out17_: _dafny.Seq
                out15_, out16_, out17_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_24_closeBudget_)
                d_25_cg_ = out15_
                d_26_ci_ = out16_
                d_27_cc_ = out17_
                generated = d_25_cg_
                insideConstrainedOut = d_26_ci_
                currentConstrainedOut = d_27_cc_
                d_2_steps_ = maxSteps
            cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

