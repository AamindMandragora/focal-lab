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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Reason step by step. At the very end of your response, write ONLY ONE final answer inside << >> delimiters using arithmetic operators +, -, *, /, //, %, int() and parentheses only. No variables in curly braces, no ** exponentiation. Example final line: <<n*(1-frac)>> or <<int(a*b/c)>>."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_preambleLimit_: int
        if (maxSteps) > (60):
            d_3_preambleLimit_ = (maxSteps) - (60)
        elif True:
            d_3_preambleLimit_ = 0
        while ((d_2_steps_) < (d_3_preambleLimit_)) and (not(insideConstrainedOut)):
            d_4_next_: _dafny.Seq
            out0_: _dafny.Seq
            out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
            d_4_next_ = out0_
            d_2_steps_ = (d_2_steps_) + (1)
            if (d_4_next_) == (eosToken):
                cost = d_2_steps_
                return generated, insideConstrainedOut, currentConstrainedOut, cost
            elif True:
                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_4_next_]))
                if (d_4_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                    d_5_eg_: _dafny.Seq
                    d_6_ei_: bool
                    d_7_ec_: _dafny.Seq
                    out1_: _dafny.Seq
                    out2_: bool
                    out3_: _dafny.Seq
                    out1_, out2_, out3_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                    d_5_eg_ = out1_
                    d_6_ei_ = out2_
                    d_7_ec_ = out3_
                    generated = d_5_eg_
                    insideConstrainedOut = d_6_ei_
                    currentConstrainedOut = d_7_ec_
        if (not(insideConstrainedOut)) and ((d_2_steps_) < (maxSteps)):
            d_8_og_: _dafny.Seq
            d_9_oi_: bool
            d_10_oc_: _dafny.Seq
            out4_: _dafny.Seq
            out5_: bool
            out6_: _dafny.Seq
            out4_, out5_, out6_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_8_og_ = out4_
            d_9_oi_ = out5_
            d_10_oc_ = out6_
            generated = d_8_og_
            insideConstrainedOut = d_9_oi_
            currentConstrainedOut = d_10_oc_
            d_2_steps_ = (d_2_steps_) + (1)
        while ((d_2_steps_) < (maxSteps)) and (insideConstrainedOut):
            d_11_cg_: _dafny.Seq
            d_12_ci_: bool
            d_13_cc_: _dafny.Seq
            d_14_closed_: bool
            out7_: _dafny.Seq
            out8_: bool
            out9_: _dafny.Seq
            out10_: bool
            out7_, out8_, out9_, out10_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
            d_11_cg_ = out7_
            d_12_ci_ = out8_
            d_13_cc_ = out9_
            d_14_closed_ = out10_
            d_2_steps_ = (d_2_steps_) + (1)
            if d_14_closed_:
                generated = d_11_cg_
                insideConstrainedOut = d_12_ci_
                currentConstrainedOut = d_13_cc_
            elif True:
                d_15_constrainedPrompt_: _dafny.Seq
                d_15_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                d_16_next_: _dafny.Seq
                out11_: _dafny.Seq
                out11_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_15_constrainedPrompt_, currentConstrainedOut, eosToken)
                d_16_next_ = out11_
                if (d_16_next_) == (eosToken):
                    insideConstrainedOut = False
                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                elif True:
                    d_17_ag_: _dafny.Seq
                    d_18_ai_: bool
                    d_19_ac_: _dafny.Seq
                    out12_: _dafny.Seq
                    out13_: bool
                    out14_: _dafny.Seq
                    out12_, out13_, out14_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_16_next_)
                    d_17_ag_ = out12_
                    d_18_ai_ = out13_
                    d_19_ac_ = out14_
                    generated = d_17_ag_
                    insideConstrainedOut = d_18_ai_
                    currentConstrainedOut = d_19_ac_
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

