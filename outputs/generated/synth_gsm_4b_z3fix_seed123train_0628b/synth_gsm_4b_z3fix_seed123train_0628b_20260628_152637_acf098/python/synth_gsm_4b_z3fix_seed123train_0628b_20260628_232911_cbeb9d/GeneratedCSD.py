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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. Write the final answer as a single arithmetic expression inside << >>. Use only: variable names (no braces), integers, +, -, *, /, //, %, int(), parentheses. No {}, no **, no words inside << >>. Write << >> exactly ONCE at the end.")))
        d_2_unconstrainedBudget_: int
        d_2_unconstrainedBudget_ = _dafny.euclidian_division((maxSteps) * (80), 100)
        if (maxSteps) >= (60):
            if (d_2_unconstrainedBudget_) > ((maxSteps) - (60)):
                d_2_unconstrainedBudget_ = (maxSteps) - (60)
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
            d_4_remB_: int
            d_4_remB_ = (maxSteps) - (d_1_steps_)
            d_5_fillBudgetB_: int
            d_5_fillBudgetB_ = _dafny.euclidian_division(d_4_remB_, 2)
            if (d_5_fillBudgetB_) < (1):
                d_5_fillBudgetB_ = 1
            if (d_5_fillBudgetB_) > (d_4_remB_):
                d_5_fillBudgetB_ = d_4_remB_
            d_6_stableB_: _dafny.Seq
            d_6_stableB_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
            d_7_filledB_: _dafny.Seq
            out4_: _dafny.Seq
            out4_ = (d_0_helpers_).RegenerateUnitOnGroundingFailure(lm, parser, (prompt) + (d_6_stableB_), currentConstrainedOut, eosToken, d_5_fillBudgetB_, 3, d_5_fillBudgetB_)
            d_7_filledB_ = out4_
            generated = (d_6_stableB_) + (d_7_filledB_)
            currentConstrainedOut = d_7_filledB_
            d_1_steps_ = (d_1_steps_) + (d_5_fillBudgetB_)
            if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
                d_8_closeBudgetB_: int
                d_8_closeBudgetB_ = (maxSteps) - (d_1_steps_)
                out5_: _dafny.Seq
                out6_: bool
                out7_: _dafny.Seq
                out5_, out6_, out7_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_8_closeBudgetB_)
                generated = out5_
                insideConstrainedOut = out6_
                currentConstrainedOut = out7_
                d_1_steps_ = (d_1_steps_) + (d_8_closeBudgetB_)
        if (not(insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
            d_9_remC_: int
            d_9_remC_ = (maxSteps) - (d_1_steps_)
            if (d_9_remC_) >= (5):
                out8_: _dafny.Seq
                out9_: bool
                out10_: _dafny.Seq
                out8_, out9_, out10_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                generated = out8_
                insideConstrainedOut = out9_
                currentConstrainedOut = out10_
                d_1_steps_ = (d_1_steps_) + (1)
                if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
                    d_10_remC2_: int
                    d_10_remC2_ = (maxSteps) - (d_1_steps_)
                    d_11_fillBudgetC_: int
                    d_11_fillBudgetC_ = _dafny.euclidian_division(d_10_remC2_, 2)
                    if (d_11_fillBudgetC_) < (1):
                        d_11_fillBudgetC_ = 1
                    if (d_11_fillBudgetC_) > (d_10_remC2_):
                        d_11_fillBudgetC_ = d_10_remC2_
                    d_12_stableC_: _dafny.Seq
                    d_12_stableC_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                    d_13_filledC_: _dafny.Seq
                    out11_: _dafny.Seq
                    out11_ = (d_0_helpers_).RegenerateUnitOnGroundingFailure(lm, parser, (prompt) + (d_12_stableC_), currentConstrainedOut, eosToken, d_11_fillBudgetC_, 3, d_11_fillBudgetC_)
                    d_13_filledC_ = out11_
                    generated = (d_12_stableC_) + (d_13_filledC_)
                    currentConstrainedOut = d_13_filledC_
                    d_1_steps_ = (d_1_steps_) + (d_11_fillBudgetC_)
                    if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
                        d_14_closeBudgetC_: int
                        d_14_closeBudgetC_ = (maxSteps) - (d_1_steps_)
                        out12_: _dafny.Seq
                        out13_: bool
                        out14_: _dafny.Seq
                        out12_, out13_, out14_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_14_closeBudgetC_)
                        generated = out12_
                        insideConstrainedOut = out13_
                        currentConstrainedOut = out14_
                        d_1_steps_ = (d_1_steps_) + (d_14_closeBudgetC_)
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_15_closeBudgetD_: int
            d_15_closeBudgetD_ = (maxSteps) - (d_1_steps_)
            out15_: _dafny.Seq
            out16_: bool
            out17_: _dafny.Seq
            out15_, out16_, out17_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_15_closeBudgetD_)
            generated = out15_
            insideConstrainedOut = out16_
            currentConstrainedOut = out17_
            d_1_steps_ = (d_1_steps_) + (d_15_closeBudgetD_)
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

