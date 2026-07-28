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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the problem step by step. Write your final answer as a single Python-style arithmetic expression inside << >>. Use the exact variable names shown in the problem (the words inside {} braces, WITHOUT the braces - write n1 not {n1}, write cost not {cost}). Use int() to wrap the entire expression when the answer is an integer count or monetary amount. Use // for integer division (e.g. minutes // 60). Do not include { or } characters inside << >>. Include every relevant variable in your formula."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        while (d_2_steps_) < (maxSteps):
            if not(insideConstrainedOut):
                d_3_next_: _dafny.Seq
                out0_: _dafny.Seq
                out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                d_3_next_ = out0_
                d_2_steps_ = (d_2_steps_) + (1)
                if (d_3_next_) == (eosToken):
                    cost = d_2_steps_
                    return generated, insideConstrainedOut, currentConstrainedOut, cost
                elif True:
                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_3_next_]))
                    if (d_3_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                        d_4_og_: _dafny.Seq
                        d_5_oi_: bool
                        d_6_oc_: _dafny.Seq
                        out1_: _dafny.Seq
                        out2_: bool
                        out3_: _dafny.Seq
                        out1_, out2_, out3_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                        d_4_og_ = out1_
                        d_5_oi_ = out2_
                        d_6_oc_ = out3_
                        generated = d_4_og_
                        insideConstrainedOut = d_5_oi_
                        currentConstrainedOut = d_6_oc_
            elif True:
                if (parser).IsCompletePrefix(currentConstrainedOut):
                    d_7_cg_: _dafny.Seq
                    d_8_ci_: bool
                    d_9_cc_: _dafny.Seq
                    out4_: _dafny.Seq
                    out5_: bool
                    out6_: _dafny.Seq
                    out4_, out5_, out6_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                    d_7_cg_ = out4_
                    d_8_ci_ = out5_
                    d_9_cc_ = out6_
                    generated = d_7_cg_
                    insideConstrainedOut = d_8_ci_
                    currentConstrainedOut = d_9_cc_
                    d_2_steps_ = (d_2_steps_) + (1)
                elif True:
                    d_10_constrainedPrompt_: _dafny.Seq
                    d_10_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                    d_11_next_: _dafny.Seq
                    out7_: _dafny.Seq
                    out7_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_10_constrainedPrompt_, currentConstrainedOut, eosToken)
                    d_11_next_ = out7_
                    d_2_steps_ = (d_2_steps_) + (1)
                    if (d_11_next_) == (eosToken):
                        insideConstrainedOut = False
                        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        d_12_ag_: _dafny.Seq
                        d_13_ai_: bool
                        d_14_ac_: _dafny.Seq
                        out8_: _dafny.Seq
                        out9_: bool
                        out10_: _dafny.Seq
                        out8_, out9_, out10_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_11_next_)
                        d_12_ag_ = out8_
                        d_13_ai_ = out9_
                        d_14_ac_ = out10_
                        generated = d_12_ag_
                        insideConstrainedOut = d_13_ai_
                        currentConstrainedOut = d_14_ac_
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_15_closeBudget_: int
            d_15_closeBudget_ = (maxSteps) - (d_2_steps_)
            d_16_cg_: _dafny.Seq
            d_17_ci_: bool
            d_18_cc_: _dafny.Seq
            out11_: _dafny.Seq
            out12_: bool
            out13_: _dafny.Seq
            out11_, out12_, out13_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_15_closeBudget_)
            d_16_cg_ = out11_
            d_17_ci_ = out12_
            d_18_cc_ = out13_
            generated = d_16_cg_
            insideConstrainedOut = d_17_ci_
            currentConstrainedOut = d_18_cc_
            d_2_steps_ = maxSteps
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

