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
        (d_0_helpers_).AppendTaskGuidance(lm, (((((((((((_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve this math problem step by step, showing all work in plain text. "))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "At the very end of your response, write the answer EXACTLY ONCE in this format: <<int(expr)>> ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "STRICT RULES for expr: ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "(1) NO curly braces {} anywhere — write n_1 not {n_1}, write initial_amount not {initial_amount}. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "(2) Use EXACTLY the variable names as written in the problem (preserve underscores: n_1 not n1, unit_price not unit price). ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "(3) Allowed symbols only: variable names, integers, +, -, *, /, //, %, (, ), int(). ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "(4) NO ** operator — you cannot write x**y. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "(5) NO LaTeX (no \\frac, no ^, no \\times). ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "(6) Always wrap the final expression in int(). ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "(7) Write ONLY ONE <<int(expr)>> block and put it at the very end. Do NOT write << anywhere else. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Correct example: <<int(n_1 * price - n_2 * discount)>> ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Wrong example: <<int({n_1} * {price} - {n_2} * {discount})>>"))))
        d_1_steps_: int
        d_1_steps_ = 0
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_2_budget0_: int
            d_2_budget0_ = 20
            if (d_2_budget0_) > ((maxSteps) - (d_1_steps_)):
                d_2_budget0_ = (maxSteps) - (d_1_steps_)
            d_3_cg0_: _dafny.Seq
            d_4_ci0_: bool
            d_5_cc0_: _dafny.Seq
            out0_: _dafny.Seq
            out1_: bool
            out2_: _dafny.Seq
            out0_, out1_, out2_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_2_budget0_)
            d_3_cg0_ = out0_
            d_4_ci0_ = out1_
            d_5_cc0_ = out2_
            generated = d_3_cg0_
            insideConstrainedOut = d_4_ci0_
            currentConstrainedOut = d_5_cc0_
            d_1_steps_ = (d_1_steps_) + (d_2_budget0_)
        with _dafny.label("0"):
            while ((d_1_steps_) < (maxSteps)) and (not(insideConstrainedOut)):
                with _dafny.c_label("0"):
                    d_6_next_: _dafny.Seq
                    out3_: _dafny.Seq
                    out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                    d_6_next_ = out3_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if (d_6_next_) == (eosToken):
                        raise _dafny.Break("0")
                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_6_next_]))
                    if VerifiedDecoderAgent.default__.RenderedEndsWith(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                        insideConstrainedOut = True
                        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_7_budget2_: int
            d_7_budget2_ = (maxSteps) - (d_1_steps_)
            d_8_cg2_: _dafny.Seq
            d_9_ci2_: bool
            d_10_cc2_: _dafny.Seq
            out4_: _dafny.Seq
            out5_: bool
            out6_: _dafny.Seq
            out4_, out5_, out6_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_7_budget2_)
            d_8_cg2_ = out4_
            d_9_ci2_ = out5_
            d_10_cc2_ = out6_
            generated = d_8_cg2_
            insideConstrainedOut = d_9_ci2_
            currentConstrainedOut = d_10_cc2_
            d_1_steps_ = maxSteps
        if (not(insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
            d_11_og_: _dafny.Seq
            d_12_oi_: bool
            d_13_oc_: _dafny.Seq
            out7_: _dafny.Seq
            out8_: bool
            out9_: _dafny.Seq
            out7_, out8_, out9_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_11_og_ = out7_
            d_12_oi_ = out8_
            d_13_oc_ = out9_
            generated = d_11_og_
            insideConstrainedOut = d_12_oi_
            currentConstrainedOut = d_13_oc_
            d_1_steps_ = (d_1_steps_) + (1)
            if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
                d_14_budget3_: int
                d_14_budget3_ = (maxSteps) - (d_1_steps_)
                d_15_cg3_: _dafny.Seq
                d_16_ci3_: bool
                d_17_cc3_: _dafny.Seq
                out10_: _dafny.Seq
                out11_: bool
                out12_: _dafny.Seq
                out10_, out11_, out12_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_14_budget3_)
                d_15_cg3_ = out10_
                d_16_ci3_ = out11_
                d_17_cc3_ = out12_
                generated = d_15_cg3_
                insideConstrainedOut = d_16_ci3_
                currentConstrainedOut = d_17_cc3_
                d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

