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
        d_1_steps_: int
        d_1_steps_ = 0
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math problem step by step using the variable names from the problem (no curly braces). At the very end, write the final answer expression inside << >>. Use ONLY: variable names without braces, integers, +, -, *, /, //, %, int(), and parentheses. No LaTeX, no {braces}, no ** exponents, no text inside << >>. Write << >> exactly once at the end. Example: The answer is <<count * (n1 + n2 + n3)>>")))
        d_2_unconstrainedBudget_: int
        d_2_unconstrainedBudget_ = _dafny.euclidian_division((maxSteps) * (80), 100)
        if ((maxSteps) >= (30)) and ((d_2_unconstrainedBudget_) > ((maxSteps) - (30))):
            d_2_unconstrainedBudget_ = (maxSteps) - (30)
        if (d_2_unconstrainedBudget_) > (maxSteps):
            d_2_unconstrainedBudget_ = maxSteps
        d_3_spanClosed_: bool
        d_3_spanClosed_ = False
        with _dafny.label("0"):
            while (((d_1_steps_) < (d_2_unconstrainedBudget_)) and (not(insideConstrainedOut))) and (not(d_3_spanClosed_)):
                with _dafny.c_label("0"):
                    d_4_next_: _dafny.Seq
                    out0_: _dafny.Seq
                    out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                    d_4_next_ = out0_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if (d_4_next_) == (eosToken):
                        raise _dafny.Break("0")
                    elif True:
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_4_next_]))
                        if (d_4_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                            out1_: _dafny.Seq
                            out2_: bool
                            out3_: _dafny.Seq
                            out1_, out2_, out3_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                            generated = out1_
                            insideConstrainedOut = out2_
                            currentConstrainedOut = out3_
                            if (d_1_steps_) < (maxSteps):
                                d_5_closeBudget1_: int
                                d_5_closeBudget1_ = (maxSteps) - (d_1_steps_)
                                if (d_5_closeBudget1_) > (25):
                                    d_5_closeBudget1_ = 25
                                out4_: _dafny.Seq
                                out5_: bool
                                out6_: _dafny.Seq
                                out4_, out5_, out6_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_5_closeBudget1_)
                                generated = out4_
                                insideConstrainedOut = out5_
                                currentConstrainedOut = out6_
                                d_1_steps_ = (d_1_steps_) + (d_5_closeBudget1_)
                                if (d_1_steps_) > (maxSteps):
                                    d_1_steps_ = maxSteps
                                if not(insideConstrainedOut):
                                    d_3_spanClosed_ = True
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_6_closeBudget2_: int
            d_6_closeBudget2_ = (maxSteps) - (d_1_steps_)
            out7_: _dafny.Seq
            out8_: bool
            out9_: _dafny.Seq
            out7_, out8_, out9_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_6_closeBudget2_)
            generated = out7_
            insideConstrainedOut = out8_
            currentConstrainedOut = out9_
            d_1_steps_ = maxSteps
            if not(insideConstrainedOut):
                d_3_spanClosed_ = True
        if ((not(insideConstrainedOut)) and (not(d_3_spanClosed_))) and ((d_1_steps_) < (maxSteps)):
            d_7_remaining_: int
            d_7_remaining_ = (maxSteps) - (d_1_steps_)
            if (d_7_remaining_) >= (2):
                out10_: _dafny.Seq
                out11_: bool
                out12_: _dafny.Seq
                out10_, out11_, out12_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                generated = out10_
                insideConstrainedOut = out11_
                currentConstrainedOut = out12_
                d_1_steps_ = (d_1_steps_) + (1)
                if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
                    d_8_closeBudget3_: int
                    d_8_closeBudget3_ = (maxSteps) - (d_1_steps_)
                    out13_: _dafny.Seq
                    out14_: bool
                    out15_: _dafny.Seq
                    out13_, out14_, out15_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_8_closeBudget3_)
                    generated = out13_
                    insideConstrainedOut = out14_
                    currentConstrainedOut = out15_
                    d_1_steps_ = maxSteps
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_9_closeBudget_: int
            d_9_closeBudget_ = (maxSteps) - (d_1_steps_)
            out16_: _dafny.Seq
            out17_: bool
            out18_: _dafny.Seq
            out16_, out17_, out18_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_9_closeBudget_)
            generated = out16_
            insideConstrainedOut = out17_
            currentConstrainedOut = out18_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

