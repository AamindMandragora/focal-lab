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
        (d_0_helpers_).AppendTaskGuidance(lm, ((((_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve this math problem step by step. "))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "At the very end, write your final answer exactly once as: <<int(expr)>> ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Rules for expr: variable names without curly braces, integers, +, -, *, /, //, %, (, ), int(). ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Always wrap in int(). No {braces}, no ** operator, no LaTeX. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Keep the expression short. Example: <<int(n * price - discount)>>"))))
        d_1_spanBudget_: int
        d_1_spanBudget_ = 50
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_hadConstrainedActivity_: bool
        d_3_hadConstrainedActivity_ = insideConstrained
        if not(insideConstrainedOut):
            d_4_phase1Budget_: int = int(0)
            if (maxSteps) > ((d_1_spanBudget_) + (1)):
                d_4_phase1Budget_ = ((maxSteps) - (d_1_spanBudget_)) - (1)
            elif True:
                d_4_phase1Budget_ = 0
            with _dafny.label("0_0"):
                while ((d_2_steps_) < (d_4_phase1Budget_)) and (not(insideConstrainedOut)):
                    with _dafny.c_label("0_0"):
                        d_5_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_5_next_ = out0_
                        d_2_steps_ = (d_2_steps_) + (1)
                        if (d_5_next_) == (eosToken):
                            raise _dafny.Break("0_0")
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_5_next_]))
                        if VerifiedDecoderAgent.default__.RenderedEndsWith(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                            insideConstrainedOut = True
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                        pass
                pass
            d_3_hadConstrainedActivity_ = (d_3_hadConstrainedActivity_) or (insideConstrainedOut)
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_6_remaining2_: int
            d_6_remaining2_ = (maxSteps) - (d_2_steps_)
            d_7_budget2_: int = int(0)
            if (d_6_remaining2_) < (d_1_spanBudget_):
                d_7_budget2_ = d_6_remaining2_
            elif True:
                d_7_budget2_ = d_1_spanBudget_
            d_8_cg2_: _dafny.Seq
            d_9_ci2_: bool
            d_10_cc2_: _dafny.Seq
            out1_: _dafny.Seq
            out2_: bool
            out3_: _dafny.Seq
            out1_, out2_, out3_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_7_budget2_)
            d_8_cg2_ = out1_
            d_9_ci2_ = out2_
            d_10_cc2_ = out3_
            generated = d_8_cg2_
            insideConstrainedOut = d_9_ci2_
            currentConstrainedOut = d_10_cc2_
            d_2_steps_ = (d_2_steps_) + (d_7_budget2_)
        if ((not(d_3_hadConstrainedActivity_)) and (not(insideConstrainedOut))) and ((d_2_steps_) < (maxSteps)):
            d_11_og_: _dafny.Seq
            d_12_oi_: bool
            d_13_oc_: _dafny.Seq
            out4_: _dafny.Seq
            out5_: bool
            out6_: _dafny.Seq
            out4_, out5_, out6_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_11_og_ = out4_
            d_12_oi_ = out5_
            d_13_oc_ = out6_
            generated = d_11_og_
            insideConstrainedOut = d_12_oi_
            currentConstrainedOut = d_13_oc_
            d_2_steps_ = (d_2_steps_) + (1)
            if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
                d_14_remaining3_: int
                d_14_remaining3_ = (maxSteps) - (d_2_steps_)
                d_15_budget3_: int = int(0)
                if (d_14_remaining3_) < (d_1_spanBudget_):
                    d_15_budget3_ = d_14_remaining3_
                elif True:
                    d_15_budget3_ = d_1_spanBudget_
                d_16_cg3_: _dafny.Seq
                d_17_ci3_: bool
                d_18_cc3_: _dafny.Seq
                out7_: _dafny.Seq
                out8_: bool
                out9_: _dafny.Seq
                out7_, out8_, out9_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_15_budget3_)
                d_16_cg3_ = out7_
                d_17_ci3_ = out8_
                d_18_cc3_ = out9_
                generated = d_16_cg3_
                insideConstrainedOut = d_17_ci3_
                currentConstrainedOut = d_18_cc3_
                d_2_steps_ = (d_2_steps_) + (d_15_budget3_)
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

