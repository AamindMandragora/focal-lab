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
        (d_0_helpers_).AppendTaskGuidance(lm, ((((_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve this math problem step by step. "))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "At the end, write the final answer as <<int(expr)>> using PLAIN variable names ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "(no curly braces — write n not {n}, write price not {price}). ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Use only operators: +, -, *, /, //, (, ), int(). ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Example: <<int(n * price - discount)>>"))))
        d_1_spanBudget_: int
        d_1_spanBudget_ = 50
        d_2_steps_: int
        d_2_steps_ = 0
        if insideConstrainedOut:
            if (d_2_steps_) < (maxSteps):
                d_3_remaining0_: int
                d_3_remaining0_ = (maxSteps) - (d_2_steps_)
                d_4_budget0_: int = int(0)
                if (d_3_remaining0_) < (d_1_spanBudget_):
                    d_4_budget0_ = d_3_remaining0_
                elif True:
                    d_4_budget0_ = d_1_spanBudget_
                d_5_cg0_: _dafny.Seq
                d_6_ci0_: bool
                d_7_cc0_: _dafny.Seq
                out0_: _dafny.Seq
                out1_: bool
                out2_: _dafny.Seq
                out0_, out1_, out2_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_4_budget0_)
                d_5_cg0_ = out0_
                d_6_ci0_ = out1_
                d_7_cc0_ = out2_
                generated = d_5_cg0_
                insideConstrainedOut = d_6_ci0_
                currentConstrainedOut = d_7_cc0_
                d_2_steps_ = (d_2_steps_) + (d_4_budget0_)
        elif True:
            d_8_phase1Budget_: int = int(0)
            if (maxSteps) > ((d_1_spanBudget_) + (1)):
                d_8_phase1Budget_ = ((maxSteps) - (d_1_spanBudget_)) - (1)
            elif True:
                d_8_phase1Budget_ = maxSteps
            with _dafny.label("1_0"):
                while (d_2_steps_) < (d_8_phase1Budget_):
                    with _dafny.c_label("1_0"):
                        d_9_next_: _dafny.Seq
                        out3_: _dafny.Seq
                        out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_9_next_ = out3_
                        d_2_steps_ = (d_2_steps_) + (1)
                        if (d_9_next_) == (eosToken):
                            raise _dafny.Break("1_0")
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_9_next_]))
                        pass
                pass
            d_10_generatedStr_: _dafny.Seq
            d_10_generatedStr_ = VerifiedDecoderAgent.CSDHelpers.PrefixToString(generated)
            d_11_closeCount_: int
            d_11_closeCount_ = VerifiedDecoderAgent.CSDHelpers.CountSubstring(d_10_generatedStr_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")))
            d_12_openCount_: int
            d_12_openCount_ = VerifiedDecoderAgent.CSDHelpers.CountSubstring(d_10_generatedStr_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))
            if ((d_11_closeCount_) == (0)) and ((d_2_steps_) < (maxSteps)):
                if (d_12_openCount_) > (0):
                    d_13_og_: _dafny.Seq
                    d_14_oi_: bool
                    d_15_oc_: _dafny.Seq
                    out4_: _dafny.Seq
                    out5_: bool
                    out6_: _dafny.Seq
                    out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                    d_13_og_ = out4_
                    d_14_oi_ = out5_
                    d_15_oc_ = out6_
                    generated = d_13_og_
                    insideConstrainedOut = d_14_oi_
                    currentConstrainedOut = d_15_oc_
                elif True:
                    d_16_og_: _dafny.Seq
                    d_17_oi_: bool
                    d_18_oc_: _dafny.Seq
                    out7_: _dafny.Seq
                    out8_: bool
                    out9_: _dafny.Seq
                    out7_, out8_, out9_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                    d_16_og_ = out7_
                    d_17_oi_ = out8_
                    d_18_oc_ = out9_
                    generated = d_16_og_
                    insideConstrainedOut = d_17_oi_
                    currentConstrainedOut = d_18_oc_
                    d_2_steps_ = (d_2_steps_) + (1)
                if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
                    d_19_remaining3_: int
                    d_19_remaining3_ = (maxSteps) - (d_2_steps_)
                    d_20_budget3_: int = int(0)
                    if (d_19_remaining3_) < (d_1_spanBudget_):
                        d_20_budget3_ = d_19_remaining3_
                    elif True:
                        d_20_budget3_ = d_1_spanBudget_
                    d_21_cg3_: _dafny.Seq
                    d_22_ci3_: bool
                    d_23_cc3_: _dafny.Seq
                    out10_: _dafny.Seq
                    out11_: bool
                    out12_: _dafny.Seq
                    out10_, out11_, out12_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_20_budget3_)
                    d_21_cg3_ = out10_
                    d_22_ci3_ = out11_
                    d_23_cc3_ = out12_
                    generated = d_21_cg3_
                    insideConstrainedOut = d_22_ci3_
                    currentConstrainedOut = d_23_cc3_
                    d_2_steps_ = (d_2_steps_) + (d_20_budget3_)
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

