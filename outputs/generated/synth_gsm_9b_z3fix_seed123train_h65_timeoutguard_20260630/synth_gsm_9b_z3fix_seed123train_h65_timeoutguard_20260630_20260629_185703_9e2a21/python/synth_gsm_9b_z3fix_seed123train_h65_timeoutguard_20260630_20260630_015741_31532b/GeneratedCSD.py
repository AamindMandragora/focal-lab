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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math problem step by step using the given variable names.\n\nFINAL ANSWER RULES:\n1. Place your ONE final answer inside << >> at the very end, like <<expression>>.\n2. Use variable names exactly as written (NO curly braces - write n1 not {n1}).\n3. OPERATORS AVAILABLE: + - * // % int()   [Note: // is integer/floor division]\n4. Use // (not /) for integer division: total_minutes // 60 for hours, total_items // n for groups.\n5. Use int() when multiplying a whole-number quantity by a fraction: int(price * frac), int(n * rate).\n6. Do NOT use ** (not supported). For powers, the problem will use specific numbers.\n7. Write the COMPLETE arithmetic expression - not just one variable.\n\nExamples of correct format:\n  'frac of n1 pieces plus n2 pieces' -> <<int(n1 * frac) + n2>>\n  'total minutes converted to hours' -> <<(n1 + n2) * t * d // 60>>\n  'price minus discount' -> <<p1 + (p1 + p2) + int(p3 * frac) + (p4 - d)>>\n  'first hour plus remaining hours at multiplier rate' -> <<c1 + (h - free - 1) * c1 * mult>>")))
        d_1_steps_: int
        d_1_steps_ = 0
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    d_2_next_: _dafny.Seq
                    out0_: _dafny.Seq
                    out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                    d_2_next_ = out0_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if (d_2_next_) == (eosToken):
                        raise _dafny.Break("0")
                    elif True:
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_2_next_]))
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_3_remainingA_: int
            d_3_remainingA_ = (maxSteps) - (d_1_steps_)
            d_4_closeBudgetA_: int
            if (d_3_remainingA_) < (60):
                d_4_closeBudgetA_ = d_3_remainingA_
            elif True:
                d_4_closeBudgetA_ = 60
            if (d_4_closeBudgetA_) > (0):
                d_5_cgA_: _dafny.Seq
                d_6_ciA_: bool
                d_7_ccA_: _dafny.Seq
                out1_: _dafny.Seq
                out2_: bool
                out3_: _dafny.Seq
                out1_, out2_, out3_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_4_closeBudgetA_)
                d_5_cgA_ = out1_
                d_6_ciA_ = out2_
                d_7_ccA_ = out3_
                generated = d_5_cgA_
                insideConstrainedOut = d_6_ciA_
                currentConstrainedOut = d_7_ccA_
                d_1_steps_ = (d_1_steps_) + (d_4_closeBudgetA_)
        d_8_genStr_: _dafny.Seq
        d_8_genStr_ = VerifiedDecoderAgent.CSDHelpers.PrefixToString(generated)
        d_9_openCount_: int
        d_9_openCount_ = VerifiedDecoderAgent.CSDHelpers.CountSubstring(d_8_genStr_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))
        if (((d_9_openCount_) == (0)) and (not(insideConstrainedOut))) and ((d_1_steps_) < (maxSteps)):
            d_10_remainingB_: int
            d_10_remainingB_ = (maxSteps) - (d_1_steps_)
            if (d_10_remainingB_) >= (10):
                d_11_ogB_: _dafny.Seq
                d_12_oiB_: bool
                d_13_ocB_: _dafny.Seq
                out4_: _dafny.Seq
                out5_: bool
                out6_: _dafny.Seq
                out4_, out5_, out6_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                d_11_ogB_ = out4_
                d_12_oiB_ = out5_
                d_13_ocB_ = out6_
                generated = d_11_ogB_
                insideConstrainedOut = d_12_oiB_
                currentConstrainedOut = d_13_ocB_
                d_1_steps_ = (d_1_steps_) + (1)
                if (d_1_steps_) < (maxSteps):
                    d_14_remainingB2_: int
                    d_14_remainingB2_ = (maxSteps) - (d_1_steps_)
                    d_15_closeBudgetB_: int
                    if (d_14_remainingB2_) < (80):
                        d_15_closeBudgetB_ = d_14_remainingB2_
                    elif True:
                        d_15_closeBudgetB_ = 80
                    if (d_15_closeBudgetB_) > (0):
                        d_16_cgB_: _dafny.Seq
                        d_17_ciB_: bool
                        d_18_ccB_: _dafny.Seq
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: _dafny.Seq
                        out7_, out8_, out9_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_15_closeBudgetB_)
                        d_16_cgB_ = out7_
                        d_17_ciB_ = out8_
                        d_18_ccB_ = out9_
                        generated = d_16_cgB_
                        insideConstrainedOut = d_17_ciB_
                        currentConstrainedOut = d_18_ccB_
                        d_1_steps_ = (d_1_steps_) + (d_15_closeBudgetB_)
        if (d_1_steps_) > (maxSteps):
            cost = maxSteps
        elif True:
            cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

