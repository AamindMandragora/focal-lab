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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. At the very end, write ONLY the final answer expression inside << >>. Use only variable names (no braces), numbers, +, -, *, /, //, int(), and parentheses. Example: <<int(n1 * price - discount)>>. Do not open << >> until you are ready to write the final answer.")))
        d_2_unconstrainedBudget_: int
        d_2_unconstrainedBudget_ = _dafny.euclidian_division((maxSteps) * (80), 100)
        if ((maxSteps) >= (50)) and ((d_2_unconstrainedBudget_) > ((maxSteps) - (50))):
            d_2_unconstrainedBudget_ = (maxSteps) - (50)
        if (d_2_unconstrainedBudget_) > (maxSteps):
            d_2_unconstrainedBudget_ = maxSteps
        with _dafny.label("0"):
            while ((d_1_steps_) < (d_2_unconstrainedBudget_)) and (not(insideConstrainedOut)):
                with _dafny.c_label("0"):
                    d_3_next_: _dafny.Seq
                    out0_: _dafny.Seq
                    out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                    d_3_next_ = out0_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if (d_3_next_) == (eosToken):
                        raise _dafny.Break("0")
                    elif True:
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_3_next_]))
                        if (d_3_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                            out1_: _dafny.Seq
                            out2_: bool
                            out3_: _dafny.Seq
                            out1_, out2_, out3_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                            generated = out1_
                            insideConstrainedOut = out2_
                            currentConstrainedOut = out3_
                            if (d_1_steps_) < (maxSteps):
                                d_4_spanBudget_: int
                                d_4_spanBudget_ = (maxSteps) - (d_1_steps_)
                                if (d_4_spanBudget_) > (60):
                                    d_4_spanBudget_ = 60
                                out4_: _dafny.Seq
                                out5_: bool
                                out6_: _dafny.Seq
                                out4_, out5_, out6_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_4_spanBudget_)
                                generated = out4_
                                insideConstrainedOut = out5_
                                currentConstrainedOut = out6_
                                d_1_steps_ = (d_1_steps_) + (d_4_spanBudget_)
                                if (d_1_steps_) > (maxSteps):
                                    d_1_steps_ = maxSteps
                            raise _dafny.Break("0")
                    pass
            pass
        with _dafny.label("1"):
            while ((d_1_steps_) < (d_2_unconstrainedBudget_)) and (not(insideConstrainedOut)):
                with _dafny.c_label("1"):
                    d_5_next_: _dafny.Seq
                    out7_: _dafny.Seq
                    out7_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                    d_5_next_ = out7_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if (d_5_next_) == (eosToken):
                        raise _dafny.Break("1")
                    elif True:
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_5_next_]))
                        if (d_5_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                            out8_: _dafny.Seq
                            out9_: bool
                            out10_: _dafny.Seq
                            out8_, out9_, out10_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                            generated = out8_
                            insideConstrainedOut = out9_
                            currentConstrainedOut = out10_
                            if (d_1_steps_) < (maxSteps):
                                d_6_spanBudget2_: int
                                d_6_spanBudget2_ = (maxSteps) - (d_1_steps_)
                                if (d_6_spanBudget2_) > (60):
                                    d_6_spanBudget2_ = 60
                                out11_: _dafny.Seq
                                out12_: bool
                                out13_: _dafny.Seq
                                out11_, out12_, out13_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_6_spanBudget2_)
                                generated = out11_
                                insideConstrainedOut = out12_
                                currentConstrainedOut = out13_
                                d_1_steps_ = (d_1_steps_) + (d_6_spanBudget2_)
                                if (d_1_steps_) > (maxSteps):
                                    d_1_steps_ = maxSteps
                            raise _dafny.Break("1")
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_7_closeRemaining_: int
            d_7_closeRemaining_ = (maxSteps) - (d_1_steps_)
            out14_: _dafny.Seq
            out15_: bool
            out16_: _dafny.Seq
            out14_, out15_, out16_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_7_closeRemaining_)
            generated = out14_
            insideConstrainedOut = out15_
            currentConstrainedOut = out16_
            d_1_steps_ = maxSteps
        if (not(insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
            d_8_remaining_: int
            d_8_remaining_ = (maxSteps) - (d_1_steps_)
            if (d_8_remaining_) >= (3):
                out17_: _dafny.Seq
                out18_: bool
                out19_: _dafny.Seq
                out17_, out18_, out19_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                generated = out17_
                insideConstrainedOut = out18_
                currentConstrainedOut = out19_
                d_1_steps_ = (d_1_steps_) + (1)
                if (d_1_steps_) < (maxSteps):
                    d_9_finalCloseBudget_: int
                    d_9_finalCloseBudget_ = (maxSteps) - (d_1_steps_)
                    out20_: _dafny.Seq
                    out21_: bool
                    out22_: _dafny.Seq
                    out20_, out21_, out22_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_9_finalCloseBudget_)
                    generated = out20_
                    insideConstrainedOut = out21_
                    currentConstrainedOut = out22_
                    d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

