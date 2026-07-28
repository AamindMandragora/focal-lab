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
        d_1_guidance_: _dafny.Seq
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step. Write your reasoning in plain text. Then write EXACTLY ONE final arithmetic expression inside << >>. Use only variable names (no curly braces), +, -, *, /, //, (, ). Stop after closing >>. Do not open another << >> after the final answer."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        if (maxSteps) == (0):
            cost = 0
        elif True:
            d_2_steps_: int
            d_2_steps_ = 0
            d_3_prefixBudget_: int
            d_3_prefixBudget_ = _dafny.euclidian_division((maxSteps) * (2), 3)
            if (d_3_prefixBudget_) == (0):
                d_3_prefixBudget_ = 1
            if (d_3_prefixBudget_) >= (maxSteps):
                d_3_prefixBudget_ = (maxSteps) - (1)
            with _dafny.label("1_0"):
                while ((d_2_steps_) < (d_3_prefixBudget_)) and (not(insideConstrainedOut)):
                    with _dafny.c_label("1_0"):
                        d_4_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_4_next_ = out0_
                        d_2_steps_ = (d_2_steps_) + (1)
                        if (d_4_next_) == (eosToken):
                            raise _dafny.Break("1_0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_4_next_]))
                            if (d_4_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                d_5_g2_: _dafny.Seq
                                d_6_ic2_: bool
                                d_7_cc2_: _dafny.Seq
                                out1_: _dafny.Seq
                                out2_: bool
                                out3_: _dafny.Seq
                                out1_, out2_, out3_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_5_g2_ = out1_
                                d_6_ic2_ = out2_
                                d_7_cc2_ = out3_
                                generated = d_5_g2_
                                insideConstrainedOut = d_6_ic2_
                                currentConstrainedOut = d_7_cc2_
                        pass
                pass
            if (not(insideConstrainedOut)) and ((d_2_steps_) < (maxSteps)):
                d_8_g2_: _dafny.Seq
                d_9_ic2_: bool
                d_10_cc2_: _dafny.Seq
                out4_: _dafny.Seq
                out5_: bool
                out6_: _dafny.Seq
                out4_, out5_, out6_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                d_8_g2_ = out4_
                d_9_ic2_ = out5_
                d_10_cc2_ = out6_
                generated = d_8_g2_
                insideConstrainedOut = d_9_ic2_
                currentConstrainedOut = d_10_cc2_
                d_2_steps_ = (d_2_steps_) + (1)
            if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
                d_11_spanBudget_: int
                d_11_spanBudget_ = (maxSteps) - (d_2_steps_)
                if (d_11_spanBudget_) > (32):
                    d_11_spanBudget_ = 32
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
                d_2_steps_ = (d_2_steps_) + (d_11_spanBudget_)
            cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

