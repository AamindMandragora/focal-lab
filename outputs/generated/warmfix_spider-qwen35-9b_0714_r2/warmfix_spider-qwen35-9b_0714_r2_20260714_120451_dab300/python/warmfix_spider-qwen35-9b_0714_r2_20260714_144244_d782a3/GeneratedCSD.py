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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate a direct SQL query using JOIN when needed. Select only the columns asked for. Do not add unnecessary WHERE clauses or table aliases."))
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
            d_7_minConstrainedTokens_ = 10
            while ((d_2_steps_) < (maxSteps)) and (insideConstrainedOut):
                d_8_remainingBudget_: int
                d_8_remainingBudget_ = (maxSteps) - (d_2_steps_)
                d_9_allowClose_: bool
                d_9_allowClose_ = ((d_6_constrainedTokenCount_) >= (d_7_minConstrainedTokens_)) or ((d_8_remainingBudget_) <= (3))
                if (d_9_allowClose_) and ((parser).IsCompletePrefix(currentConstrainedOut)):
                    d_10_cg_: _dafny.Seq
                    d_11_ci_: bool
                    d_12_cc_: _dafny.Seq
                    out3_: _dafny.Seq
                    out4_: bool
                    out5_: _dafny.Seq
                    out3_, out4_, out5_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                    d_10_cg_ = out3_
                    d_11_ci_ = out4_
                    d_12_cc_ = out5_
                    generated = d_10_cg_
                    insideConstrainedOut = d_11_ci_
                    currentConstrainedOut = d_12_cc_
                    d_2_steps_ = (d_2_steps_) + (1)
                elif True:
                    d_13_constrainedPrompt_: _dafny.Seq
                    d_13_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                    d_14_next_: _dafny.Seq
                    out6_: _dafny.Seq
                    out6_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_13_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('8e0'), 100, eosToken)
                    d_14_next_ = out6_
                    d_2_steps_ = (d_2_steps_) + (1)
                    if (d_14_next_) == (eosToken):
                        d_15_closeBudget_: int
                        d_15_closeBudget_ = (maxSteps) - (d_2_steps_)
                        if (d_15_closeBudget_) > (0):
                            d_16_cg_: _dafny.Seq
                            d_17_ci_: bool
                            d_18_cc_: _dafny.Seq
                            out7_: _dafny.Seq
                            out8_: bool
                            out9_: _dafny.Seq
                            out7_, out8_, out9_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_15_closeBudget_)
                            d_16_cg_ = out7_
                            d_17_ci_ = out8_
                            d_18_cc_ = out9_
                            generated = d_16_cg_
                            insideConstrainedOut = d_17_ci_
                            currentConstrainedOut = d_18_cc_
                            d_2_steps_ = maxSteps
                    elif True:
                        d_19_ag_: _dafny.Seq
                        d_20_ai_: bool
                        d_21_ac_: _dafny.Seq
                        out10_: _dafny.Seq
                        out11_: bool
                        out12_: _dafny.Seq
                        out10_, out11_, out12_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_14_next_)
                        d_19_ag_ = out10_
                        d_20_ai_ = out11_
                        d_21_ac_ = out12_
                        generated = d_19_ag_
                        insideConstrainedOut = d_20_ai_
                        currentConstrainedOut = d_21_ac_
                        d_6_constrainedTokenCount_ = (d_6_constrainedTokenCount_) + (1)
            if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
                d_22_closeBudget_: int
                d_22_closeBudget_ = (maxSteps) - (d_2_steps_)
                d_23_cg_: _dafny.Seq
                d_24_ci_: bool
                d_25_cc_: _dafny.Seq
                out13_: _dafny.Seq
                out14_: bool
                out15_: _dafny.Seq
                out13_, out14_, out15_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_22_closeBudget_)
                d_23_cg_ = out13_
                d_24_ci_ = out14_
                d_25_cc_ = out15_
                generated = d_23_cg_
                insideConstrainedOut = d_24_ci_
                currentConstrainedOut = d_25_cc_
                d_2_steps_ = maxSteps
            cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

