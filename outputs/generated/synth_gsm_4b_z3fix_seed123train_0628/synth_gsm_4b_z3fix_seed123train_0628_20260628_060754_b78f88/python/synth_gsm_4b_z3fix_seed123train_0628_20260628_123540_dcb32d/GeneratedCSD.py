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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step. Show your reasoning, then at the very end write the exact formula as <<formula>> where formula uses plain variable names without curly braces (write n1 not {n1}), standard operators only (+, -, *, /, //, %, parentheses, int()). No LaTeX, no ** exponents, no {braces} whatsoever inside << >>. The formula should be a single expression. Example: <<n1 * c1 + n2 * c2>>"))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_spanOpened_: bool
        d_3_spanOpened_ = insideConstrained
        with _dafny.label("0"):
            while ((d_2_steps_) < (maxSteps)) and (not(insideConstrainedOut)):
                with _dafny.c_label("0"):
                    d_4_remaining_: int
                    d_4_remaining_ = (maxSteps) - (d_2_steps_)
                    if ((not(d_3_spanOpened_)) and ((d_2_steps_) >= (750))) and ((d_4_remaining_) >= (60)):
                        d_5_og_: _dafny.Seq
                        d_6_oi_: bool
                        d_7_oc_: _dafny.Seq
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: _dafny.Seq
                        out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                        d_5_og_ = out0_
                        d_6_oi_ = out1_
                        d_7_oc_ = out2_
                        generated = d_5_og_
                        insideConstrainedOut = d_6_oi_
                        currentConstrainedOut = d_7_oc_
                        d_2_steps_ = (d_2_steps_) + (1)
                        d_3_spanOpened_ = True
                        raise _dafny.Break("0")
                    if (d_4_remaining_) <= (5):
                        raise _dafny.Break("0")
                    d_8_next_: _dafny.Seq
                    out3_: _dafny.Seq
                    out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                    d_8_next_ = out3_
                    d_2_steps_ = (d_2_steps_) + (1)
                    if (d_8_next_) == (eosToken):
                        raise _dafny.Break("0")
                    elif True:
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_8_next_]))
                        if (d_8_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                            d_9_remAfter_: int
                            d_9_remAfter_ = (maxSteps) - (d_2_steps_)
                            if (d_9_remAfter_) >= (20):
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                d_3_spanOpened_ = True
                    pass
            pass
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_10_remaining2_: int
            d_10_remaining2_ = (maxSteps) - (d_2_steps_)
            d_11_closeBudget_: int
            if (d_10_remaining2_) <= (120):
                d_11_closeBudget_ = d_10_remaining2_
            elif True:
                d_11_closeBudget_ = 120
            d_12_cg_: _dafny.Seq
            d_13_ci_: bool
            d_14_cc_: _dafny.Seq
            out4_: _dafny.Seq
            out5_: bool
            out6_: _dafny.Seq
            out4_, out5_, out6_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_11_closeBudget_)
            d_12_cg_ = out4_
            d_13_ci_ = out5_
            d_14_cc_ = out6_
            generated = d_12_cg_
            insideConstrainedOut = d_13_ci_
            currentConstrainedOut = d_14_cc_
            d_2_steps_ = (d_2_steps_) + (d_11_closeBudget_)
            if (d_2_steps_) > (maxSteps):
                d_2_steps_ = maxSteps
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

