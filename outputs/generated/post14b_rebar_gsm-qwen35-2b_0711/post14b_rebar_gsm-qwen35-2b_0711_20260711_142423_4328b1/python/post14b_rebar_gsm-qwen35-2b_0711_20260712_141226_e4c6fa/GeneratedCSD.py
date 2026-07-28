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
        if (maxSteps) == (0):
            return generated, insideConstrainedOut, currentConstrainedOut, cost
        d_1_guidance_: _dafny.Seq
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math problem step by step. After your reasoning, write the final numeric answer as a complete Python arithmetic expression using all relevant variable names. The expression MUST use arithmetic operators (+, -, *, /, //, %) and NOT be a single variable alone. Format: <<EXPRESSION>> where EXPRESSION combines variables with operators. Examples: <<n1 * p1 + n2 * p2 + n3 * p3>>, <<int(100 * (k1 + k2) / (n1 + n2))>>, <<initial_amount - quantity * unit_price>>, <<total - total * fraction - current>>. Use bare variable names only (no {braces} or $). Do not use **. Write the FULL formula."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_operatorTokens_: _dafny.Seq
        d_3_operatorTokens_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "+")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "-")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "*")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "/")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "//")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "%")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "(")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ")")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ","))])
        d_4_prefixBudget_: int
        if (maxSteps) >= (200):
            d_4_prefixBudget_ = 200
        elif True:
            d_4_prefixBudget_ = maxSteps
        while ((d_2_steps_) < (d_4_prefixBudget_)) and (not(insideConstrainedOut)):
            d_5_next_: _dafny.Seq
            out0_: _dafny.Seq
            out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
            d_5_next_ = out0_
            d_2_steps_ = (d_2_steps_) + (1)
            if (d_5_next_) == (eosToken):
                cost = d_2_steps_
                return generated, insideConstrainedOut, currentConstrainedOut, cost
            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_5_next_]))
            if (d_5_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                insideConstrainedOut = True
                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
        if (not(insideConstrainedOut)) and ((d_2_steps_) < (maxSteps)):
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
        d_10_minConstrainedTokens_ = 3
        with _dafny.label("0"):
            while ((d_2_steps_) < (maxSteps)) and (insideConstrainedOut):
                with _dafny.c_label("0"):
                    if (d_9_constrainedTokenCount_) >= (d_10_minConstrainedTokens_):
                        d_11_cg_: _dafny.Seq
                        d_12_ci_: bool
                        d_13_cc_: _dafny.Seq
                        d_14_closed_: bool
                        out4_: _dafny.Seq
                        out5_: bool
                        out6_: _dafny.Seq
                        out7_: bool
                        out4_, out5_, out6_, out7_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                        d_11_cg_ = out4_
                        d_12_ci_ = out5_
                        d_13_cc_ = out6_
                        d_14_closed_ = out7_
                        d_2_steps_ = (d_2_steps_) + (1)
                        if d_14_closed_:
                            generated = d_11_cg_
                            insideConstrainedOut = d_12_ci_
                            currentConstrainedOut = d_13_cc_
                            raise _dafny.Break("0")
                    if (not(insideConstrainedOut)) or ((d_2_steps_) >= (maxSteps)):
                        raise _dafny.Break("0")
                    d_15_constrainedPrompt_: _dafny.Seq
                    d_15_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                    d_16_next_: _dafny.Seq
                    out8_: _dafny.Seq
                    out8_ = (d_0_helpers_).SafeBoostedConstrainedStep(lm, parser, d_15_constrainedPrompt_, currentConstrainedOut, d_3_operatorTokens_, _dafny.BigRational('3e0'), eosToken)
                    d_16_next_ = out8_
                    d_2_steps_ = (d_2_steps_) + (1)
                    if (d_16_next_) == (eosToken):
                        raise _dafny.Break("0")
                    d_17_appendedGenerated_: _dafny.Seq
                    d_18_appendedInside_: bool
                    d_19_appendedCurrent_: _dafny.Seq
                    out9_: _dafny.Seq
                    out10_: bool
                    out11_: _dafny.Seq
                    out9_, out10_, out11_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_16_next_)
                    d_17_appendedGenerated_ = out9_
                    d_18_appendedInside_ = out10_
                    d_19_appendedCurrent_ = out11_
                    generated = d_17_appendedGenerated_
                    insideConstrainedOut = d_18_appendedInside_
                    currentConstrainedOut = d_19_appendedCurrent_
                    d_9_constrainedTokenCount_ = (d_9_constrainedTokenCount_) + (1)
                    pass
            pass
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_20_closeBudget_: int
            d_20_closeBudget_ = (maxSteps) - (d_2_steps_)
            d_21_cg_: _dafny.Seq
            d_22_ci_: bool
            d_23_cc_: _dafny.Seq
            out12_: _dafny.Seq
            out13_: bool
            out14_: _dafny.Seq
            out12_, out13_, out14_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_20_closeBudget_)
            d_21_cg_ = out12_
            d_22_ci_ = out13_
            d_23_cc_ = out14_
            generated = d_21_cg_
            insideConstrainedOut = d_22_ci_
            currentConstrainedOut = d_23_cc_
            d_2_steps_ = maxSteps
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

