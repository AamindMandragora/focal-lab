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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. Write the final symbolic answer expression ONCE inside << >>. Use variable names without braces, numbers, +, -, *, /, //, int(), parentheses only. Example: <<int(n1 * price - discount)>>. Keep the expression inside << >> short and exact. Do not open << >> until you are ready to write the final answer. Only ONE << >> span.")))
        d_2_unconstrainedBudget_: int
        d_2_unconstrainedBudget_ = _dafny.euclidian_division((maxSteps) * (75), 100)
        if ((maxSteps) >= (60)) and ((d_2_unconstrainedBudget_) > ((maxSteps) - (60))):
            d_2_unconstrainedBudget_ = (maxSteps) - (60)
        if (d_2_unconstrainedBudget_) > (maxSteps):
            d_2_unconstrainedBudget_ = maxSteps
        d_3_closedASpan_: bool
        d_3_closedASpan_ = False
        with _dafny.label("0"):
            while (((d_1_steps_) < (d_2_unconstrainedBudget_)) and (not(insideConstrainedOut))) and (not(d_3_closedASpan_)):
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
                                if (d_5_spanBudget_) > (40):
                                    d_5_spanBudget_ = 40
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
                                    d_3_closedASpan_ = True
                            raise _dafny.Break("0")
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_6_closeRemaining_: int
            d_6_closeRemaining_ = (maxSteps) - (d_1_steps_)
            if (d_6_closeRemaining_) > (40):
                d_6_closeRemaining_ = 40
            out7_: _dafny.Seq
            out8_: bool
            out9_: _dafny.Seq
            out7_, out8_, out9_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_6_closeRemaining_)
            generated = out7_
            insideConstrainedOut = out8_
            currentConstrainedOut = out9_
            d_1_steps_ = (d_1_steps_) + (d_6_closeRemaining_)
            if (d_1_steps_) > (maxSteps):
                d_1_steps_ = maxSteps
            if not(insideConstrainedOut):
                d_3_closedASpan_ = True
        if ((not(d_3_closedASpan_)) and (not(insideConstrainedOut))) and ((d_1_steps_) < (maxSteps)):
            d_7_remaining_: int
            d_7_remaining_ = (maxSteps) - (d_1_steps_)
            if (d_7_remaining_) >= (5):
                out10_: _dafny.Seq
                out11_: bool
                out12_: _dafny.Seq
                out10_, out11_, out12_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                generated = out10_
                insideConstrainedOut = out11_
                currentConstrainedOut = out12_
                d_1_steps_ = (d_1_steps_) + (1)
                if ((d_1_steps_) < (maxSteps)) and (insideConstrainedOut):
                    d_8_fillBudget_: int
                    d_8_fillBudget_ = _dafny.euclidian_division((maxSteps) - (d_1_steps_), 2)
                    if (d_8_fillBudget_) < (1):
                        d_8_fillBudget_ = 1
                    if (d_8_fillBudget_) > (30):
                        d_8_fillBudget_ = 30
                    d_9_stable_: _dafny.Seq
                    d_9_stable_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                    d_10_filled_: _dafny.Seq
                    out13_: _dafny.Seq
                    out13_ = (d_0_helpers_).RegenerateUnitOnGroundingFailure(lm, parser, (prompt) + (d_9_stable_), currentConstrainedOut, eosToken, d_8_fillBudget_, 3, d_8_fillBudget_)
                    d_10_filled_ = out13_
                    generated = (d_9_stable_) + (d_10_filled_)
                    currentConstrainedOut = d_10_filled_
                    d_1_steps_ = (d_1_steps_) + (d_8_fillBudget_)
                    if (d_1_steps_) > (maxSteps):
                        d_1_steps_ = maxSteps
                    if (d_1_steps_) < (maxSteps):
                        d_11_closeBudget_: int
                        d_11_closeBudget_ = (maxSteps) - (d_1_steps_)
                        if (d_11_closeBudget_) > (40):
                            d_11_closeBudget_ = 40
                        out14_: _dafny.Seq
                        out15_: bool
                        out16_: _dafny.Seq
                        out14_, out15_, out16_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_11_closeBudget_)
                        generated = out14_
                        insideConstrainedOut = out15_
                        currentConstrainedOut = out16_
                        d_1_steps_ = (d_1_steps_) + (d_11_closeBudget_)
                        if (d_1_steps_) > (maxSteps):
                            d_1_steps_ = maxSteps
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_12_finalBudget_: int
            d_12_finalBudget_ = (maxSteps) - (d_1_steps_)
            out17_: _dafny.Seq
            out18_: bool
            out19_: _dafny.Seq
            out17_, out18_, out19_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_12_finalBudget_)
            generated = out17_
            insideConstrainedOut = out18_
            currentConstrainedOut = out19_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

