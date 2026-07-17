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
            pass
        elif True:
            d_1_guidance_: _dafny.Seq
            d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve this problem step by step using plain text. Work through ALL arithmetic and reasoning completely. At the very end of your solution, state the final symbolic arithmetic expression using only: variable names, numbers, +, -, *, /, //, %, (, ), int(). Do NOT use curly braces, LaTeX, or markdown. Just plain arithmetic. For example: int(n * (m + k) / 2)"))
            (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
            d_2_reservedForConstrained_: int
            d_2_reservedForConstrained_ = 20
            if (d_2_reservedForConstrained_) >= (maxSteps):
                d_2_reservedForConstrained_ = _dafny.euclidian_division(maxSteps, 2)
                if (d_2_reservedForConstrained_) == (0):
                    d_2_reservedForConstrained_ = 0
            d_3_freeBudget_: int
            d_3_freeBudget_ = (maxSteps) - (d_2_reservedForConstrained_)
            if ((d_3_freeBudget_) == (0)) and ((maxSteps) > (1)):
                d_3_freeBudget_ = (maxSteps) - (1)
                d_2_reservedForConstrained_ = 1
            d_4_steps_: int
            d_4_steps_ = 0
            d_5_hitEos_: bool
            d_5_hitEos_ = False
            while ((d_4_steps_) < (d_3_freeBudget_)) and (not(d_5_hitEos_)):
                d_6_next_: _dafny.Seq
                out0_: _dafny.Seq
                out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                d_6_next_ = out0_
                d_4_steps_ = (d_4_steps_) + (1)
                if (d_6_next_) == (eosToken):
                    d_5_hitEos_ = True
                elif True:
                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_6_next_]))
            if ((not(d_5_hitEos_)) and (not(insideConstrainedOut))) and ((d_4_steps_) < (maxSteps)):
                d_7_openG_: _dafny.Seq
                d_8_openI_: bool
                d_9_openC_: _dafny.Seq
                out1_: _dafny.Seq
                out2_: bool
                out3_: _dafny.Seq
                out1_, out2_, out3_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                d_7_openG_ = out1_
                d_8_openI_ = out2_
                d_9_openC_ = out3_
                generated = d_7_openG_
                insideConstrainedOut = d_8_openI_
                currentConstrainedOut = d_9_openC_
                d_4_steps_ = (d_4_steps_) + (1)
                if (d_4_steps_) < (maxSteps):
                    d_10_closeBudget_: int
                    d_10_closeBudget_ = (maxSteps) - (d_4_steps_)
                    d_11_cg_: _dafny.Seq
                    d_12_ci_: bool
                    d_13_cc_: _dafny.Seq
                    out4_: _dafny.Seq
                    out5_: bool
                    out6_: _dafny.Seq
                    out4_, out5_, out6_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_10_closeBudget_)
                    d_11_cg_ = out4_
                    d_12_ci_ = out5_
                    d_13_cc_ = out6_
                    generated = d_11_cg_
                    insideConstrainedOut = d_12_ci_
                    currentConstrainedOut = d_13_cc_
                    d_4_steps_ = maxSteps
            elif ((not(d_5_hitEos_)) and (insideConstrainedOut)) and ((d_4_steps_) < (maxSteps)):
                d_14_closeBudget_: int
                d_14_closeBudget_ = (maxSteps) - (d_4_steps_)
                d_15_cg_: _dafny.Seq
                d_16_ci_: bool
                d_17_cc_: _dafny.Seq
                out7_: _dafny.Seq
                out8_: bool
                out9_: _dafny.Seq
                out7_, out8_, out9_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_14_closeBudget_)
                d_15_cg_ = out7_
                d_16_ci_ = out8_
                d_17_cc_ = out9_
                generated = d_15_cg_
                insideConstrainedOut = d_16_ci_
                currentConstrainedOut = d_17_cc_
                d_4_steps_ = maxSteps
            cost = d_4_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

