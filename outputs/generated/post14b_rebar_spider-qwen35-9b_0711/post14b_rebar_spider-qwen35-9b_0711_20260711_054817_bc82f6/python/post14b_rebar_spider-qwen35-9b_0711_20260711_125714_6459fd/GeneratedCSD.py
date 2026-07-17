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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Write a SQL query to answer the question using the exact table and column names from the schema. Output only the SQL query between << and >>. Start with SELECT."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        if (maxSteps) == (0):
            cost = 0
        elif True:
            d_2_steps_: int
            d_2_steps_ = 0
            d_3_hitEos_: bool
            d_3_hitEos_ = False
            if ((not(insideConstrainedOut)) and (not(d_3_hitEos_))) and ((d_2_steps_) < (maxSteps)):
                d_4_og_: _dafny.Seq
                d_5_oi_: bool
                d_6_oc_: _dafny.Seq
                out0_: _dafny.Seq
                out1_: bool
                out2_: _dafny.Seq
                out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                d_4_og_ = out0_
                d_5_oi_ = out1_
                d_6_oc_ = out2_
                generated = d_4_og_
                insideConstrainedOut = d_5_oi_
                currentConstrainedOut = d_6_oc_
                d_2_steps_ = (d_2_steps_) + (1)
            d_7_constrainedTokenCount_: int
            d_7_constrainedTokenCount_ = 0
            d_8_minConstrainedTokens_: int
            d_8_minConstrainedTokens_ = 35
            while ((d_2_steps_) < (maxSteps)) and (insideConstrainedOut):
                d_9_remainingBudget_: int
                d_9_remainingBudget_ = (maxSteps) - (d_2_steps_)
                d_10_allowClose_: bool
                d_10_allowClose_ = ((d_7_constrainedTokenCount_) >= (d_8_minConstrainedTokens_)) or ((d_9_remainingBudget_) <= (5))
                if (d_10_allowClose_) and ((parser).IsCompletePrefix(currentConstrainedOut)):
                    d_11_cg_: _dafny.Seq
                    d_12_ci_: bool
                    d_13_cc_: _dafny.Seq
                    out3_: _dafny.Seq
                    out4_: bool
                    out5_: _dafny.Seq
                    out3_, out4_, out5_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                    d_11_cg_ = out3_
                    d_12_ci_ = out4_
                    d_13_cc_ = out5_
                    generated = d_11_cg_
                    insideConstrainedOut = d_12_ci_
                    currentConstrainedOut = d_13_cc_
                    d_2_steps_ = (d_2_steps_) + (1)
                elif True:
                    d_14_constrainedPrompt_: _dafny.Seq
                    d_14_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                    d_15_next_: _dafny.Seq
                    out6_: _dafny.Seq
                    out6_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_14_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('5e0'), 15, eosToken)
                    d_15_next_ = out6_
                    d_2_steps_ = (d_2_steps_) + (1)
                    if (d_15_next_) == (eosToken):
                        d_16_closeBudget_: int
                        d_16_closeBudget_ = (maxSteps) - (d_2_steps_)
                        if (d_16_closeBudget_) > (0):
                            d_17_cg_: _dafny.Seq
                            d_18_ci_: bool
                            d_19_cc_: _dafny.Seq
                            out7_: _dafny.Seq
                            out8_: bool
                            out9_: _dafny.Seq
                            out7_, out8_, out9_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_16_closeBudget_)
                            d_17_cg_ = out7_
                            d_18_ci_ = out8_
                            d_19_cc_ = out9_
                            generated = d_17_cg_
                            insideConstrainedOut = d_18_ci_
                            currentConstrainedOut = d_19_cc_
                            d_2_steps_ = maxSteps
                    elif True:
                        d_20_ag_: _dafny.Seq
                        d_21_ai_: bool
                        d_22_ac_: _dafny.Seq
                        out10_: _dafny.Seq
                        out11_: bool
                        out12_: _dafny.Seq
                        out10_, out11_, out12_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_15_next_)
                        d_20_ag_ = out10_
                        d_21_ai_ = out11_
                        d_22_ac_ = out12_
                        generated = d_20_ag_
                        insideConstrainedOut = d_21_ai_
                        currentConstrainedOut = d_22_ac_
                        d_7_constrainedTokenCount_ = (d_7_constrainedTokenCount_) + (1)
            if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
                d_23_closeBudget_: int
                d_23_closeBudget_ = (maxSteps) - (d_2_steps_)
                d_24_cg_: _dafny.Seq
                d_25_ci_: bool
                d_26_cc_: _dafny.Seq
                out13_: _dafny.Seq
                out14_: bool
                out15_: _dafny.Seq
                out13_, out14_, out15_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_23_closeBudget_)
                d_24_cg_ = out13_
                d_25_ci_ = out14_
                d_26_cc_ = out15_
                generated = d_24_cg_
                insideConstrainedOut = d_25_ci_
                currentConstrainedOut = d_26_cc_
                d_2_steps_ = maxSteps
            cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

