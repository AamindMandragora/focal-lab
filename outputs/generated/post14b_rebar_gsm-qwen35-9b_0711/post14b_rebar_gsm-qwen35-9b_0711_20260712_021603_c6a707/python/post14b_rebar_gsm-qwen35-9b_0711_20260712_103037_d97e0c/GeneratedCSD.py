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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Think step by step. After all reasoning, write the final symbolic answer on the last line as <<expression>> using only these operators: +, -, *, /, //, %, int(), and parentheses. Use variable names directly without curly braces. No ** operator. Only ONE final answer span."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_preambleLimit_: int
        if (maxSteps) > (80):
            d_3_preambleLimit_ = (maxSteps) - (80)
        elif True:
            d_3_preambleLimit_ = 0
        while (d_2_steps_) < (d_3_preambleLimit_):
            d_4_next_: _dafny.Seq
            out0_: _dafny.Seq
            out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
            d_4_next_ = out0_
            d_2_steps_ = (d_2_steps_) + (1)
            if (d_4_next_) == (eosToken):
                cost = d_2_steps_
                return generated, insideConstrainedOut, currentConstrainedOut, cost
            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_4_next_]))
        if ((d_2_steps_) < (maxSteps)) and (not(insideConstrainedOut)):
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
        with _dafny.label("0"):
            while (((d_2_steps_) + (1)) < (maxSteps)) and (insideConstrainedOut):
                with _dafny.c_label("0"):
                    d_8_constrainedPrompt_: _dafny.Seq
                    d_8_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                    d_9_next_: _dafny.Seq
                    out4_: _dafny.Seq
                    out4_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_8_constrainedPrompt_, currentConstrainedOut, eosToken)
                    d_9_next_ = out4_
                    d_2_steps_ = (d_2_steps_) + (1)
                    if (d_9_next_) == (eosToken):
                        raise _dafny.Break("0")
                    d_10_ag_: _dafny.Seq
                    d_11_ai_: bool
                    d_12_ac_: _dafny.Seq
                    out5_: _dafny.Seq
                    out6_: bool
                    out7_: _dafny.Seq
                    out5_, out6_, out7_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_9_next_)
                    d_10_ag_ = out5_
                    d_11_ai_ = out6_
                    d_12_ac_ = out7_
                    generated = d_10_ag_
                    insideConstrainedOut = d_11_ai_
                    currentConstrainedOut = d_12_ac_
                    if (d_2_steps_) < (maxSteps):
                        d_13_cg_: _dafny.Seq
                        d_14_ci_: bool
                        d_15_cc_: _dafny.Seq
                        d_16_closed_: bool
                        out8_: _dafny.Seq
                        out9_: bool
                        out10_: _dafny.Seq
                        out11_: bool
                        out8_, out9_, out10_, out11_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                        d_13_cg_ = out8_
                        d_14_ci_ = out9_
                        d_15_cc_ = out10_
                        d_16_closed_ = out11_
                        d_2_steps_ = (d_2_steps_) + (1)
                        if d_16_closed_:
                            generated = d_13_cg_
                            insideConstrainedOut = d_14_ci_
                            currentConstrainedOut = d_15_cc_
                    pass
            pass
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_17_closeBudget_: int
            d_17_closeBudget_ = (maxSteps) - (d_2_steps_)
            d_18_cg_: _dafny.Seq
            d_19_ci_: bool
            d_20_cc_: _dafny.Seq
            out12_: _dafny.Seq
            out13_: bool
            out14_: _dafny.Seq
            out12_, out13_, out14_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_17_closeBudget_)
            d_18_cg_ = out12_
            d_19_ci_ = out13_
            d_20_cc_ = out14_
            generated = d_18_cg_
            insideConstrainedOut = d_19_ci_
            currentConstrainedOut = d_20_cc_
            d_2_steps_ = maxSteps
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

