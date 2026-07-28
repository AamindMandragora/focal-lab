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
        (d_0_helpers_).AppendTaskGuidance(lm, (((((((_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve this math problem step by step. Show all your work clearly. "))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "CRITICAL RULE: Do NOT write << or >> anywhere in your reasoning or intermediate calculations. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Only at the very last line of your response, write the final symbolic answer in this exact format: ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<int(expr)>> ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "where expr is a single arithmetic expression using only: ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "variable names from the problem (no curly braces {}), integers, +, -, *, /, //, %, (, ), int(). ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Always wrap integer results in int(). No LaTeX, no {braces}, no ** operator. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Keep the expression concise. Example: <<int(n * price - quantity * discount)>>"))))
        d_1_steps_: int
        d_1_steps_ = 0
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_2_ib_: int
            d_2_ib_ = 15
            if (d_2_ib_) > ((maxSteps) - (d_1_steps_)):
                d_2_ib_ = (maxSteps) - (d_1_steps_)
            d_3_cg0_: _dafny.Seq
            d_4_ci0_: bool
            d_5_cc0_: _dafny.Seq
            out0_: _dafny.Seq
            out1_: bool
            out2_: _dafny.Seq
            out0_, out1_, out2_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_2_ib_)
            d_3_cg0_ = out0_
            d_4_ci0_ = out1_
            d_5_cc0_ = out2_
            generated = d_3_cg0_
            insideConstrainedOut = d_4_ci0_
            currentConstrainedOut = d_5_cc0_
            d_1_steps_ = (d_1_steps_) + (d_2_ib_)
        d_6_pb_: int
        d_6_pb_ = _dafny.euclidian_division((maxSteps) * (87), 100)
        if ((maxSteps) > (0)) and ((d_6_pb_) >= (maxSteps)):
            d_6_pb_ = (maxSteps) - (1)
        with _dafny.label("0"):
            while ((d_1_steps_) < (d_6_pb_)) and (not(insideConstrainedOut)):
                with _dafny.c_label("0"):
                    d_7_next_: _dafny.Seq
                    out3_: _dafny.Seq
                    out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                    d_7_next_ = out3_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if (d_7_next_) == (eosToken):
                        raise _dafny.Break("0")
                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_7_next_]))
                    pass
            pass
        if (not(insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
            d_8_og_: _dafny.Seq
            d_9_oi_: bool
            d_10_oc_: _dafny.Seq
            out4_: _dafny.Seq
            out5_: bool
            out6_: _dafny.Seq
            out4_, out5_, out6_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_8_og_ = out4_
            d_9_oi_ = out5_
            d_10_oc_ = out6_
            generated = d_8_og_
            insideConstrainedOut = d_9_oi_
            currentConstrainedOut = d_10_oc_
            d_1_steps_ = (d_1_steps_) + (1)
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_11_rb_: int
            d_11_rb_ = (maxSteps) - (d_1_steps_)
            d_12_cg3_: _dafny.Seq
            d_13_ci3_: bool
            d_14_cc3_: _dafny.Seq
            out7_: _dafny.Seq
            out8_: bool
            out9_: _dafny.Seq
            out7_, out8_, out9_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_11_rb_)
            d_12_cg3_ = out7_
            d_13_ci3_ = out8_
            d_14_cc3_ = out9_
            generated = d_12_cg3_
            insideConstrainedOut = d_13_ci3_
            currentConstrainedOut = d_14_cc3_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

