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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step using plain text reasoning. At the very end, write the final answer as a single arithmetic expression inside << >>. Use ONLY: variable names without braces (write x not {x}), integers, +, -, *, /, //, %, int(), and parentheses. No LaTeX, no {braces}, no ** exponents, no text. Example: <<int(n * price + base)>> or <<(total - spent) // n>>")))
        d_2_unconstrainedBudget_: int
        d_2_unconstrainedBudget_ = maxSteps
        if (maxSteps) > (50):
            d_2_unconstrainedBudget_ = (maxSteps) - (50)
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
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_4_fillBudget_: int
            d_4_fillBudget_ = (maxSteps) - (d_1_steps_)
            if (d_4_fillBudget_) > (40):
                d_4_fillBudget_ = 40
            d_5_stable_: _dafny.Seq
            d_5_stable_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
            d_6_filled_: _dafny.Seq
            out4_: _dafny.Seq
            out4_ = (d_0_helpers_).RegenerateUnitOnGroundingFailure(lm, parser, (prompt) + (d_5_stable_), currentConstrainedOut, eosToken, d_4_fillBudget_, 3, d_4_fillBudget_)
            d_6_filled_ = out4_
            generated = (d_5_stable_) + (d_6_filled_)
            currentConstrainedOut = d_6_filled_
            d_1_steps_ = (d_1_steps_) + (d_4_fillBudget_)
            if (d_1_steps_) > (maxSteps):
                d_1_steps_ = maxSteps
            if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
                d_7_closeBudget1_: int
                d_7_closeBudget1_ = (maxSteps) - (d_1_steps_)
                if (d_7_closeBudget1_) > (20):
                    d_7_closeBudget1_ = 20
                out5_: _dafny.Seq
                out6_: bool
                out7_: _dafny.Seq
                out5_, out6_, out7_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_7_closeBudget1_)
                generated = out5_
                insideConstrainedOut = out6_
                currentConstrainedOut = out7_
                d_1_steps_ = (d_1_steps_) + (d_7_closeBudget1_)
                if (d_1_steps_) > (maxSteps):
                    d_1_steps_ = maxSteps
        with _dafny.label("1"):
            while ((d_1_steps_) < (d_2_unconstrainedBudget_)) and (not(insideConstrainedOut)):
                with _dafny.c_label("1"):
                    d_8_next2_: _dafny.Seq
                    out8_: _dafny.Seq
                    out8_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                    d_8_next2_ = out8_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if (d_8_next2_) == (eosToken):
                        raise _dafny.Break("1")
                    elif True:
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_8_next2_]))
                        if (d_8_next2_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                            out9_: _dafny.Seq
                            out10_: bool
                            out11_: _dafny.Seq
                            out9_, out10_, out11_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                            generated = out9_
                            insideConstrainedOut = out10_
                            currentConstrainedOut = out11_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_9_fillBudget2_: int
            d_9_fillBudget2_ = (maxSteps) - (d_1_steps_)
            if (d_9_fillBudget2_) > (40):
                d_9_fillBudget2_ = 40
            d_10_stable2_: _dafny.Seq
            d_10_stable2_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
            d_11_filled2_: _dafny.Seq
            out12_: _dafny.Seq
            out12_ = (d_0_helpers_).RegenerateUnitOnGroundingFailure(lm, parser, (prompt) + (d_10_stable2_), currentConstrainedOut, eosToken, d_9_fillBudget2_, 3, d_9_fillBudget2_)
            d_11_filled2_ = out12_
            generated = (d_10_stable2_) + (d_11_filled2_)
            currentConstrainedOut = d_11_filled2_
            d_1_steps_ = (d_1_steps_) + (d_9_fillBudget2_)
            if (d_1_steps_) > (maxSteps):
                d_1_steps_ = maxSteps
            if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
                d_12_closeBudget2_: int
                d_12_closeBudget2_ = (maxSteps) - (d_1_steps_)
                if (d_12_closeBudget2_) > (20):
                    d_12_closeBudget2_ = 20
                out13_: _dafny.Seq
                out14_: bool
                out15_: _dafny.Seq
                out13_, out14_, out15_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_12_closeBudget2_)
                generated = out13_
                insideConstrainedOut = out14_
                currentConstrainedOut = out15_
                d_1_steps_ = (d_1_steps_) + (d_12_closeBudget2_)
                if (d_1_steps_) > (maxSteps):
                    d_1_steps_ = maxSteps
        if (not(insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
            d_13_remaining_: int
            d_13_remaining_ = (maxSteps) - (d_1_steps_)
            if (d_13_remaining_) >= (5):
                out16_: _dafny.Seq
                out17_: bool
                out18_: _dafny.Seq
                out16_, out17_, out18_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                generated = out16_
                insideConstrainedOut = out17_
                currentConstrainedOut = out18_
                d_1_steps_ = (d_1_steps_) + (1)
                if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
                    d_14_fillBudget3_: int
                    d_14_fillBudget3_ = (maxSteps) - (d_1_steps_)
                    if (d_14_fillBudget3_) > (40):
                        d_14_fillBudget3_ = 40
                    d_15_stable3_: _dafny.Seq
                    d_15_stable3_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                    d_16_filled3_: _dafny.Seq
                    out19_: _dafny.Seq
                    out19_ = (d_0_helpers_).RegenerateUnitOnGroundingFailure(lm, parser, (prompt) + (d_15_stable3_), currentConstrainedOut, eosToken, d_14_fillBudget3_, 3, d_14_fillBudget3_)
                    d_16_filled3_ = out19_
                    generated = (d_15_stable3_) + (d_16_filled3_)
                    currentConstrainedOut = d_16_filled3_
                    d_1_steps_ = (d_1_steps_) + (d_14_fillBudget3_)
                    if (d_1_steps_) > (maxSteps):
                        d_1_steps_ = maxSteps
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_17_closeBudget_: int
            d_17_closeBudget_ = (maxSteps) - (d_1_steps_)
            out20_: _dafny.Seq
            out21_: bool
            out22_: _dafny.Seq
            out20_, out21_, out22_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_17_closeBudget_)
            generated = out20_
            insideConstrainedOut = out21_
            currentConstrainedOut = out22_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

