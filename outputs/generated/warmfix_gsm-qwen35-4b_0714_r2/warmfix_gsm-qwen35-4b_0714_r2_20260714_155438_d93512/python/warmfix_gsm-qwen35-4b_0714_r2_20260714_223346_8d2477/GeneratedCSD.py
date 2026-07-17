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
        if (maxSteps) == (0):
            cost = 0
        elif True:
            d_1_guidance_: _dafny.Seq
            d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. At the END write the final arithmetic expression inside << >>. Use ONLY plain variable names (n, x, k, etc.), numbers, and operators (+, -, *, /, //, %, (, ), int()). Do NOT use {}, [], LaTeX, or **. Example: <<n * price + base>>. The << >> span contains ONLY the expression, nothing else."))
            (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
            d_2_prefixBudget_: int
            d_2_prefixBudget_ = _dafny.euclidian_division((maxSteps) * (80), 100)
            if (d_2_prefixBudget_) >= (maxSteps):
                d_2_prefixBudget_ = (maxSteps) - (1)
            d_3_steps_: int
            d_3_steps_ = 0
            with _dafny.label("1_0"):
                while ((d_3_steps_) < (d_2_prefixBudget_)) and (not(insideConstrainedOut)):
                    with _dafny.c_label("1_0"):
                        d_4_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_4_next_ = out0_
                        d_3_steps_ = (d_3_steps_) + (1)
                        if (d_4_next_) == (eosToken):
                            raise _dafny.Break("1_0")
                        elif (d_4_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_4_next_]))
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
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_4_next_]))
                        pass
                pass
            if (not(insideConstrainedOut)) and ((d_3_steps_) < (maxSteps)):
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
                d_3_steps_ = (d_3_steps_) + (1)
            if (insideConstrainedOut) and ((d_3_steps_) < (maxSteps)):
                d_11_spanBudget_: int
                d_11_spanBudget_ = (maxSteps) - (d_3_steps_)
                d_12_g3_: _dafny.Seq
                d_13_ic3_: bool
                d_14_cc3_: _dafny.Seq
                out7_: _dafny.Seq
                out8_: bool
                out9_: _dafny.Seq
                out7_, out8_, out9_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_11_spanBudget_)
                d_12_g3_ = out7_
                d_13_ic3_ = out8_
                d_14_cc3_ = out9_
                generated = d_12_g3_
                insideConstrainedOut = d_13_ic3_
                currentConstrainedOut = d_14_cc3_
                d_3_steps_ = maxSteps
            cost = d_3_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

