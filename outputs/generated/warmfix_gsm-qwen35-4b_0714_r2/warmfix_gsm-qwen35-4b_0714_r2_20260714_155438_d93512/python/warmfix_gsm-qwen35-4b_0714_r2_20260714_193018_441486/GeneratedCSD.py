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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. At the end, write the final answer as ONE expression inside << >> using only: variable names, numbers, +, -, *, /, //, %, (, ), int(). No LaTeX, no curly braces, no **. Example: <<n * price - discount>>. Put << >> at the very end."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_spanBudget_: int
        d_2_spanBudget_ = 100
        if (d_2_spanBudget_) >= (maxSteps):
            d_2_spanBudget_ = _dafny.euclidian_division(maxSteps, 2)
            if (d_2_spanBudget_) == (0):
                d_2_spanBudget_ = 1
        d_3_freeCap_: int
        d_3_freeCap_ = (maxSteps) - (d_2_spanBudget_)
        if (d_3_freeCap_) == (0):
            d_3_freeCap_ = 0
        d_4_steps_: int
        d_4_steps_ = 0
        d_5_spanDone_: bool
        d_5_spanDone_ = False
        while (((d_4_steps_) < (d_3_freeCap_)) and (not(insideConstrainedOut))) and (not(d_5_spanDone_)):
            d_6_next_: _dafny.Seq
            out0_: _dafny.Seq
            out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
            d_6_next_ = out0_
            d_4_steps_ = (d_4_steps_) + (1)
            if (d_6_next_) == (eosToken):
                d_5_spanDone_ = True
            elif True:
                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_6_next_]))
                if (d_6_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                    d_7_eg_: _dafny.Seq
                    d_8_ei_: bool
                    d_9_ec_: _dafny.Seq
                    out1_: _dafny.Seq
                    out2_: bool
                    out3_: _dafny.Seq
                    out1_, out2_, out3_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                    d_7_eg_ = out1_
                    d_8_ei_ = out2_
                    d_9_ec_ = out3_
                    generated = d_7_eg_
                    insideConstrainedOut = d_8_ei_
                    currentConstrainedOut = d_9_ec_
        if ((not(insideConstrainedOut)) and (not(d_5_spanDone_))) and ((d_4_steps_) < (maxSteps)):
            d_10_og_: _dafny.Seq
            d_11_oi_: bool
            d_12_oc_: _dafny.Seq
            out4_: _dafny.Seq
            out5_: bool
            out6_: _dafny.Seq
            out4_, out5_, out6_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_10_og_ = out4_
            d_11_oi_ = out5_
            d_12_oc_ = out6_
            generated = d_10_og_
            insideConstrainedOut = d_11_oi_
            currentConstrainedOut = d_12_oc_
            d_4_steps_ = (d_4_steps_) + (1)
        if (insideConstrainedOut) and ((d_4_steps_) < (maxSteps)):
            d_13_remainBudget_: int
            d_13_remainBudget_ = (maxSteps) - (d_4_steps_)
            d_14_cg_: _dafny.Seq
            d_15_ci_: bool
            d_16_cc_: _dafny.Seq
            out7_: _dafny.Seq
            out8_: bool
            out9_: _dafny.Seq
            out7_, out8_, out9_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_13_remainBudget_)
            d_14_cg_ = out7_
            d_15_ci_ = out8_
            d_16_cc_ = out9_
            generated = d_14_cg_
            insideConstrainedOut = d_15_ci_
            currentConstrainedOut = d_16_cc_
            d_4_steps_ = maxSteps
        cost = d_4_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

