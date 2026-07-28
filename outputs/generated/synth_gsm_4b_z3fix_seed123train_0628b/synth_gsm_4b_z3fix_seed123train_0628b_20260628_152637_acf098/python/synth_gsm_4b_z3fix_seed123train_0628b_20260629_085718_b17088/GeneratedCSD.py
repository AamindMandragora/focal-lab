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
        d_2_closedASpan_: bool
        d_2_closedASpan_ = False
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. Write the final symbolic answer expression ONCE at the very end inside << >>. Use only variable names (no braces), numbers, +, -, *, /, //, int(), and parentheses. Example: <<int(n1 * price - discount)>>. Do not open << >> until you have finished your reasoning.")))
        d_3_unconstrainedBudget_: int
        d_3_unconstrainedBudget_ = _dafny.euclidian_division((maxSteps) * (80), 100)
        if ((maxSteps) >= (50)) and ((d_3_unconstrainedBudget_) > ((maxSteps) - (50))):
            d_3_unconstrainedBudget_ = (maxSteps) - (50)
        if (d_3_unconstrainedBudget_) > (maxSteps):
            d_3_unconstrainedBudget_ = maxSteps
        with _dafny.label("0"):
            while (((d_1_steps_) < (d_3_unconstrainedBudget_)) and (not(insideConstrainedOut))) and (not(d_2_closedASpan_)):
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
                                d_5_spanBudget_: int
                                d_5_spanBudget_ = (maxSteps) - (d_1_steps_)
                                out4_: _dafny.Seq
                                out5_: bool
                                out6_: _dafny.Seq
                                out4_, out5_, out6_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_5_spanBudget_)
                                generated = out4_
                                insideConstrainedOut = out5_
                                currentConstrainedOut = out6_
                                d_1_steps_ = (d_1_steps_) + (d_5_spanBudget_)
                                if (d_1_steps_) > (maxSteps):
                                    d_1_steps_ = maxSteps
                                if not(insideConstrainedOut):
                                    d_2_closedASpan_ = True
                            raise _dafny.Break("0")
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_6_closeRemaining_: int
            d_6_closeRemaining_ = (maxSteps) - (d_1_steps_)
            out7_: _dafny.Seq
            out8_: bool
            out9_: _dafny.Seq
            out7_, out8_, out9_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_6_closeRemaining_)
            generated = out7_
            insideConstrainedOut = out8_
            currentConstrainedOut = out9_
            d_1_steps_ = maxSteps
            if not(insideConstrainedOut):
                d_2_closedASpan_ = True
        if ((not(d_2_closedASpan_)) and (not(insideConstrainedOut))) and ((d_1_steps_) < (maxSteps)):
            d_7_remaining_: int
            d_7_remaining_ = (maxSteps) - (d_1_steps_)
            if (d_7_remaining_) >= (3):
                out10_: _dafny.Seq
                out11_: bool
                out12_: _dafny.Seq
                out10_, out11_, out12_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                generated = out10_
                insideConstrainedOut = out11_
                currentConstrainedOut = out12_
                d_1_steps_ = (d_1_steps_) + (1)
                if (d_1_steps_) < (maxSteps):
                    d_8_finalCloseBudget_: int
                    d_8_finalCloseBudget_ = (maxSteps) - (d_1_steps_)
                    out13_: _dafny.Seq
                    out14_: bool
                    out15_: _dafny.Seq
                    out13_, out14_, out15_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_8_finalCloseBudget_)
                    generated = out13_
                    insideConstrainedOut = out14_
                    currentConstrainedOut = out15_
                    d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

