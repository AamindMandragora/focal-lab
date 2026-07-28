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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Output exactly: SQL: <<query>> where query is a complete valid SQL query. Include JOINs, WHERE, GROUP BY, ORDER BY, LIMIT as needed. Use only the provided schema tables and columns. The << and >> are required visible delimiters."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        if (maxSteps) == (0):
            cost = 0
        elif True:
            d_2_steps_: int
            d_2_steps_ = 0
            d_3_hitEos_: bool
            d_3_hitEos_ = False
            d_4_prefixBudget_: int
            d_4_prefixBudget_ = 10
            while ((((d_2_steps_) < (d_4_prefixBudget_)) and ((d_2_steps_) < (maxSteps))) and (not(insideConstrainedOut))) and (not(d_3_hitEos_)):
                d_5_next_: _dafny.Seq
                out0_: _dafny.Seq
                out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                d_5_next_ = out0_
                d_2_steps_ = (d_2_steps_) + (1)
                if (d_5_next_) == (eosToken):
                    d_3_hitEos_ = True
                elif True:
                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_5_next_]))
                    if (d_5_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                        insideConstrainedOut = True
                        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
            if ((not(insideConstrainedOut)) and (not(d_3_hitEos_))) and ((d_2_steps_) < (maxSteps)):
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
                d_2_steps_ = (d_2_steps_) + (1)
            d_9_constrainedTokenCount_: int
            d_9_constrainedTokenCount_ = 0
            d_10_minConstrainedTokens_: int
            d_10_minConstrainedTokens_ = 20
            while ((d_2_steps_) < (maxSteps)) and (insideConstrainedOut):
                d_11_remainingBudget_: int
                d_11_remainingBudget_ = (maxSteps) - (d_2_steps_)
                if ((parser).IsCompletePrefix(currentConstrainedOut)) and (((d_9_constrainedTokenCount_) >= (d_10_minConstrainedTokens_)) or ((d_11_remainingBudget_) <= (2))):
                    d_12_cg_: _dafny.Seq
                    d_13_ci_: bool
                    d_14_cc_: _dafny.Seq
                    out4_: _dafny.Seq
                    out5_: bool
                    out6_: _dafny.Seq
                    out4_, out5_, out6_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                    d_12_cg_ = out4_
                    d_13_ci_ = out5_
                    d_14_cc_ = out6_
                    generated = d_12_cg_
                    insideConstrainedOut = d_13_ci_
                    currentConstrainedOut = d_14_cc_
                    d_2_steps_ = (d_2_steps_) + (1)
                elif True:
                    d_15_constrainedPrompt_: _dafny.Seq
                    d_15_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                    d_16_next_: _dafny.Seq
                    out7_: _dafny.Seq
                    out7_ = (d_0_helpers_).SafeTemperatureConstrainedStep(lm, parser, d_15_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('7e-1'), eosToken)
                    d_16_next_ = out7_
                    d_2_steps_ = (d_2_steps_) + (1)
                    if (d_16_next_) == (eosToken):
                        d_17_closeBudget_: int
                        d_17_closeBudget_ = (maxSteps) - (d_2_steps_)
                        if (d_17_closeBudget_) > (0):
                            d_18_cg_: _dafny.Seq
                            d_19_ci_: bool
                            d_20_cc_: _dafny.Seq
                            out8_: _dafny.Seq
                            out9_: bool
                            out10_: _dafny.Seq
                            out8_, out9_, out10_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_17_closeBudget_)
                            d_18_cg_ = out8_
                            d_19_ci_ = out9_
                            d_20_cc_ = out10_
                            generated = d_18_cg_
                            insideConstrainedOut = d_19_ci_
                            currentConstrainedOut = d_20_cc_
                            d_2_steps_ = maxSteps
                    elif True:
                        d_21_ag_: _dafny.Seq
                        d_22_ai_: bool
                        d_23_ac_: _dafny.Seq
                        out11_: _dafny.Seq
                        out12_: bool
                        out13_: _dafny.Seq
                        out11_, out12_, out13_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_16_next_)
                        d_21_ag_ = out11_
                        d_22_ai_ = out12_
                        d_23_ac_ = out13_
                        generated = d_21_ag_
                        insideConstrainedOut = d_22_ai_
                        currentConstrainedOut = d_23_ac_
                        d_9_constrainedTokenCount_ = (d_9_constrainedTokenCount_) + (1)
            if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
                d_24_closeBudget_: int
                d_24_closeBudget_ = (maxSteps) - (d_2_steps_)
                d_25_cg_: _dafny.Seq
                d_26_ci_: bool
                d_27_cc_: _dafny.Seq
                out14_: _dafny.Seq
                out15_: bool
                out16_: _dafny.Seq
                out14_, out15_, out16_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_24_closeBudget_)
                d_25_cg_ = out14_
                d_26_ci_ = out15_
                d_27_cc_ = out16_
                generated = d_25_cg_
                insideConstrainedOut = d_26_ci_
                currentConstrainedOut = d_27_cc_
                d_2_steps_ = maxSteps
            cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

