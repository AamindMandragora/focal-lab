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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math problem step by step. At the end, write the final answer as a single Python arithmetic expression between << and >>. RULES: (1) Use bare variable names only, NO braces {} or dollar signs $. (2) Use only +, -, *, /, //, %, int(), and parentheses. (3) Do NOT use ** or pow(). (4) Write exactly ONE complete expression. Examples: <<n1 * p1 + n2 * p2 + n3 * p3>>, <<int(100 * (k1 + k2) / (n1 + n2))>>, <<(a + b) * c // d>>, <<total - n1 - mult * n1>>, <<(price + price * percent / 100) * 7 * usage + extra_price>>."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_prefixBudget_: int
        if (maxSteps) >= (120):
            d_3_prefixBudget_ = 120
        elif True:
            d_3_prefixBudget_ = maxSteps
        while ((d_2_steps_) < (d_3_prefixBudget_)) and (not(insideConstrainedOut)):
            d_4_next_: _dafny.Seq
            out0_: _dafny.Seq
            out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
            d_4_next_ = out0_
            d_2_steps_ = (d_2_steps_) + (1)
            if (d_4_next_) == (eosToken):
                cost = d_2_steps_
                return generated, insideConstrainedOut, currentConstrainedOut, cost
            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_4_next_]))
            if (d_4_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                insideConstrainedOut = True
                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
        if (not(insideConstrainedOut)) and ((d_2_steps_) < (maxSteps)):
            d_5_og_: _dafny.Seq
            d_6_oi_: bool
            d_7_oc_: _dafny.Seq
            out1_: _dafny.Seq
            out2_: bool
            out3_: _dafny.Seq
            out1_, out2_, out3_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_5_og_ = out1_
            d_6_oi_ = out2_
            d_7_oc_ = out3_
            generated = d_5_og_
            insideConstrainedOut = d_6_oi_
            currentConstrainedOut = d_7_oc_
            d_2_steps_ = (d_2_steps_) + (1)
        d_8_maxConstrainedBodySteps_: int
        d_8_maxConstrainedBodySteps_ = 40
        d_9_constrainedBodySteps_: int
        d_9_constrainedBodySteps_ = 0
        with _dafny.label("0"):
            while ((insideConstrainedOut) and ((d_9_constrainedBodySteps_) < (d_8_maxConstrainedBodySteps_))) and ((d_2_steps_) < (maxSteps)):
                with _dafny.c_label("0"):
                    d_10_constrainedPrompt_: _dafny.Seq
                    d_10_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                    d_11_penaltyTokens_: _dafny.Seq
                    d_11_penaltyTokens_ = _dafny.SeqWithoutIsStrInference([])
                    d_12_next_: _dafny.Seq
                    out4_: _dafny.Seq
                    out4_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_10_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_11_penaltyTokens_, _dafny.BigRational('4e0'), 12, eosToken)
                    d_12_next_ = out4_
                    d_2_steps_ = (d_2_steps_) + (1)
                    d_9_constrainedBodySteps_ = (d_9_constrainedBodySteps_) + (1)
                    if (d_12_next_) == (eosToken):
                        raise _dafny.Break("0")
                    d_13_valid_: bool
                    out5_: bool
                    out5_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_12_next_)
                    d_13_valid_ = out5_
                    if d_13_valid_:
                        d_14_ag_: _dafny.Seq
                        d_15_ai_: bool
                        d_16_ac_: _dafny.Seq
                        out6_: _dafny.Seq
                        out7_: bool
                        out8_: _dafny.Seq
                        out6_, out7_, out8_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_12_next_)
                        d_14_ag_ = out6_
                        d_15_ai_ = out7_
                        d_16_ac_ = out8_
                        generated = d_14_ag_
                        insideConstrainedOut = d_15_ai_
                        currentConstrainedOut = d_16_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_17_closeBudget_: int
            d_17_closeBudget_ = (maxSteps) - (d_2_steps_)
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
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

