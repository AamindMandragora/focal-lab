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
        (d_0_helpers_).AppendTaskGuidance(lm, ((((((((((_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve this math problem step by step.\n"))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Write intermediate symbolic computations and the final answer inside << >> delimiters.\n")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "CRITICAL VARIABLE NAMING: In the problem, {name} is a placeholder meaning the variable\n")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "is called 'name'. Inside << >> spans, write variable names WITHOUT curly braces.\n")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "CORRECT: <<n * frac_1>>          WRONG: <<{n} * {frac_1}>>\n")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "CORRECT: <<end_time - start_time>> WRONG: <<{end_time} - {start_time}>>\n")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "CORRECT: <<n0 * (r + 1)>>        WRONG: <<{n0} * ({r} + 1)>>\n")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Allowed operators inside << >>: + - * / // ( ) only.\n")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Do NOT use ** for powers (write x * x instead of x**2).\n")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Use // for integer division when quantities must be whole numbers.\n")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "The LAST << >> span must be the complete final answer as a single arithmetic expression."))))
        d_1_steps_: int
        d_1_steps_ = 0
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_2_initBudget_: int
            if (maxSteps) >= (50):
                d_2_initBudget_ = 50
            elif True:
                d_2_initBudget_ = maxSteps
            d_3_cg_: _dafny.Seq
            d_4_ci_: bool
            d_5_cc_: _dafny.Seq
            out0_: _dafny.Seq
            out1_: bool
            out2_: _dafny.Seq
            out0_, out1_, out2_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_2_initBudget_)
            d_3_cg_ = out0_
            d_4_ci_ = out1_
            d_5_cc_ = out2_
            generated = d_3_cg_
            insideConstrainedOut = d_4_ci_
            currentConstrainedOut = d_5_cc_
            d_1_steps_ = (d_1_steps_) + (d_2_initBudget_)
            insideConstrainedOut = False
            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    d_6_next_: _dafny.Seq
                    out3_: _dafny.Seq
                    out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                    d_6_next_ = out3_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if (d_6_next_) == (eosToken):
                        raise _dafny.Break("0")
                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_6_next_]))
                    if (VerifiedDecoderAgent.default__.RenderedEndsWith(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))) and ((d_1_steps_) < (maxSteps)):
                        d_7_og_: _dafny.Seq
                        d_8_oi_: bool
                        d_9_oc_: _dafny.Seq
                        out4_: _dafny.Seq
                        out5_: bool
                        out6_: _dafny.Seq
                        out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                        d_7_og_ = out4_
                        d_8_oi_ = out5_
                        d_9_oc_ = out6_
                        generated = d_7_og_
                        insideConstrainedOut = d_8_oi_
                        currentConstrainedOut = d_9_oc_
                        d_10_remaining_: int
                        d_10_remaining_ = (maxSteps) - (d_1_steps_)
                        d_11_spanBudget_: int
                        if (d_10_remaining_) >= (50):
                            d_11_spanBudget_ = 50
                        elif True:
                            d_11_spanBudget_ = d_10_remaining_
                        d_12_cg_: _dafny.Seq
                        d_13_ci_: bool
                        d_14_cc_: _dafny.Seq
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: _dafny.Seq
                        out7_, out8_, out9_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_11_spanBudget_)
                        d_12_cg_ = out7_
                        d_13_ci_ = out8_
                        d_14_cc_ = out9_
                        generated = d_12_cg_
                        insideConstrainedOut = d_13_ci_
                        currentConstrainedOut = d_14_cc_
                        d_1_steps_ = (d_1_steps_) + (d_11_spanBudget_)
                        insideConstrainedOut = False
                        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

