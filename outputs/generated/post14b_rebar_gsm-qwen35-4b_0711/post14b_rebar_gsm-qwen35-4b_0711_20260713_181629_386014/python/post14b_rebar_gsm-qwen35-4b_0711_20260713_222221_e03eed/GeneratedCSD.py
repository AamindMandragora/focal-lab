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
            d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the problem step by step using plain text reasoning. At the very end, write your final arithmetic expression inside << >> exactly once. The expression inside << >> must be a COMPLETE arithmetic expression using the variable names from the problem (e.g., <<n * (m + k)>>). Use only: variable names, numbers, +, -, *, /, //, %, (, ), int(). Do NOT open << >> anywhere except for the single final answer. Make sure to include ALL variables needed for the complete answer inside the single << >> span."))
            (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
            d_2_freeSteps_: int
            d_2_freeSteps_ = _dafny.euclidian_division((maxSteps) * (9), 10)
            if (d_2_freeSteps_) == (0):
                d_2_freeSteps_ = 1
            if (d_2_freeSteps_) >= (maxSteps):
                d_2_freeSteps_ = (maxSteps) - (1)
            d_3_steps_: int
            d_3_steps_ = 0
            with _dafny.label("1_0"):
                while (d_3_steps_) < (d_2_freeSteps_):
                    with _dafny.c_label("1_0"):
                        d_4_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_4_next_ = out0_
                        d_3_steps_ = (d_3_steps_) + (1)
                        if (d_4_next_) == (eosToken):
                            raise _dafny.Break("1_0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_4_next_]))
                            if (d_4_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                if not(insideConstrainedOut):
                                    insideConstrainedOut = True
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                            elif (d_4_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))):
                                if insideConstrainedOut:
                                    insideConstrainedOut = False
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                        pass
                pass
            if (insideConstrainedOut) and ((d_3_steps_) < (maxSteps)):
                d_5_closeBudget_: int
                d_5_closeBudget_ = (maxSteps) - (d_3_steps_)
                d_6_cg_: _dafny.Seq
                d_7_ci_: bool
                d_8_cc_: _dafny.Seq
                out1_: _dafny.Seq
                out2_: bool
                out3_: _dafny.Seq
                out1_, out2_, out3_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_5_closeBudget_)
                d_6_cg_ = out1_
                d_7_ci_ = out2_
                d_8_cc_ = out3_
                generated = d_6_cg_
                insideConstrainedOut = d_7_ci_
                currentConstrainedOut = d_8_cc_
                d_3_steps_ = maxSteps
            elif (not(insideConstrainedOut)) and ((d_3_steps_) < (maxSteps)):
                d_9_openG_: _dafny.Seq
                d_10_openI_: bool
                d_11_openC_: _dafny.Seq
                out4_: _dafny.Seq
                out5_: bool
                out6_: _dafny.Seq
                out4_, out5_, out6_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                d_9_openG_ = out4_
                d_10_openI_ = out5_
                d_11_openC_ = out6_
                generated = d_9_openG_
                insideConstrainedOut = d_10_openI_
                currentConstrainedOut = d_11_openC_
                d_3_steps_ = (d_3_steps_) + (1)
                if (d_3_steps_) < (maxSteps):
                    d_12_closeBudget2_: int
                    d_12_closeBudget2_ = (maxSteps) - (d_3_steps_)
                    d_13_cg2_: _dafny.Seq
                    d_14_ci2_: bool
                    d_15_cc2_: _dafny.Seq
                    out7_: _dafny.Seq
                    out8_: bool
                    out9_: _dafny.Seq
                    out7_, out8_, out9_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_12_closeBudget2_)
                    d_13_cg2_ = out7_
                    d_14_ci2_ = out8_
                    d_15_cc2_ = out9_
                    generated = d_13_cg2_
                    insideConstrainedOut = d_14_ci2_
                    currentConstrainedOut = d_15_cc2_
                    d_3_steps_ = maxSteps
            cost = d_3_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

