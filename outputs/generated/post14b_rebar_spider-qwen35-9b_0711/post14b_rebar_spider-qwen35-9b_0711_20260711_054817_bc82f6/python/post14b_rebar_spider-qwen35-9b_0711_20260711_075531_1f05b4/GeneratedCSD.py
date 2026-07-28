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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate a valid SQL query. Output format: SQL: <<query>>. Use only schema tables and columns. Write complete SQL with all necessary JOINs, WHERE, GROUP BY, ORDER BY, LIMIT clauses as needed."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        if (maxSteps) == (0):
            cost = 0
        elif True:
            d_2_steps_: int
            d_2_steps_ = 0
            d_3_hitEos_: bool
            d_3_hitEos_ = False
            d_4_prefixBudget_: int
            d_4_prefixBudget_ = 6
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
            d_9_closedSpan_: bool
            d_9_closedSpan_ = False
            while (((d_2_steps_) < (maxSteps)) and (insideConstrainedOut)) and (not(d_9_closedSpan_)):
                d_10_remainingBudget_: int
                d_10_remainingBudget_ = (maxSteps) - (d_2_steps_)
                if (d_10_remainingBudget_) <= (1):
                    if (parser).IsCompletePrefix(currentConstrainedOut):
                        d_11_cg_: _dafny.Seq
                        d_12_ci_: bool
                        d_13_cc_: _dafny.Seq
                        out4_: _dafny.Seq
                        out5_: bool
                        out6_: _dafny.Seq
                        out4_, out5_, out6_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_11_cg_ = out4_
                        d_12_ci_ = out5_
                        d_13_cc_ = out6_
                        generated = d_11_cg_
                        insideConstrainedOut = d_12_ci_
                        currentConstrainedOut = d_13_cc_
                        d_2_steps_ = (d_2_steps_) + (1)
                        d_9_closedSpan_ = True
                    elif True:
                        d_14_cg_: _dafny.Seq
                        d_15_ci_: bool
                        d_16_cc_: _dafny.Seq
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: _dafny.Seq
                        out7_, out8_, out9_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_10_remainingBudget_)
                        d_14_cg_ = out7_
                        d_15_ci_ = out8_
                        d_16_cc_ = out9_
                        generated = d_14_cg_
                        insideConstrainedOut = d_15_ci_
                        currentConstrainedOut = d_16_cc_
                        d_2_steps_ = maxSteps
                        d_9_closedSpan_ = True
                elif True:
                    d_17_constrainedPrompt_: _dafny.Seq
                    d_17_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                    d_18_next_: _dafny.Seq
                    out10_: _dafny.Seq
                    out10_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_17_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('15e-1'), eosToken)
                    d_18_next_ = out10_
                    d_2_steps_ = (d_2_steps_) + (1)
                    if (d_18_next_) == (eosToken):
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            d_19_closeBudget_: int
                            d_19_closeBudget_ = (maxSteps) - (d_2_steps_)
                            if (d_19_closeBudget_) > (0):
                                d_20_cg_: _dafny.Seq
                                d_21_ci_: bool
                                d_22_cc_: _dafny.Seq
                                out11_: _dafny.Seq
                                out12_: bool
                                out13_: _dafny.Seq
                                out11_, out12_, out13_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_20_cg_ = out11_
                                d_21_ci_ = out12_
                                d_22_cc_ = out13_
                                generated = d_20_cg_
                                insideConstrainedOut = d_21_ci_
                                currentConstrainedOut = d_22_cc_
                                d_2_steps_ = (d_2_steps_) + (1)
                            d_9_closedSpan_ = True
                        elif True:
                            d_23_closeBudget_: int
                            d_23_closeBudget_ = (maxSteps) - (d_2_steps_)
                            if (d_23_closeBudget_) > (0):
                                d_24_cg_: _dafny.Seq
                                d_25_ci_: bool
                                d_26_cc_: _dafny.Seq
                                out14_: _dafny.Seq
                                out15_: bool
                                out16_: _dafny.Seq
                                out14_, out15_, out16_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_23_closeBudget_)
                                d_24_cg_ = out14_
                                d_25_ci_ = out15_
                                d_26_cc_ = out16_
                                generated = d_24_cg_
                                insideConstrainedOut = d_25_ci_
                                currentConstrainedOut = d_26_cc_
                                d_2_steps_ = maxSteps
                            d_9_closedSpan_ = True
                    elif True:
                        d_27_ag_: _dafny.Seq
                        d_28_ai_: bool
                        d_29_ac_: _dafny.Seq
                        out17_: _dafny.Seq
                        out18_: bool
                        out19_: _dafny.Seq
                        out17_, out18_, out19_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_18_next_)
                        d_27_ag_ = out17_
                        d_28_ai_ = out18_
                        d_29_ac_ = out19_
                        generated = d_27_ag_
                        insideConstrainedOut = d_28_ai_
                        currentConstrainedOut = d_29_ac_
            if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
                d_30_closeBudget_: int
                d_30_closeBudget_ = (maxSteps) - (d_2_steps_)
                d_31_cg_: _dafny.Seq
                d_32_ci_: bool
                d_33_cc_: _dafny.Seq
                out20_: _dafny.Seq
                out21_: bool
                out22_: _dafny.Seq
                out20_, out21_, out22_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_30_closeBudget_)
                d_31_cg_ = out20_
                d_32_ci_ = out21_
                d_33_cc_ = out22_
                generated = d_31_cg_
                insideConstrainedOut = d_32_ci_
                currentConstrainedOut = d_33_cc_
                d_2_steps_ = maxSteps
            cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

