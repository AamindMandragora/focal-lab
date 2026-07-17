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
            d_2_freeBudget_: int
            d_2_freeBudget_ = _dafny.euclidian_division((maxSteps) * (11), 20)
            if (d_2_freeBudget_) == (0):
                d_2_freeBudget_ = 1
            if (d_2_freeBudget_) >= (maxSteps):
                d_2_freeBudget_ = (maxSteps) - (1)
            d_3_steps_: int
            d_3_steps_ = 0
            with _dafny.label("1_0"):
                while (d_3_steps_) < (d_2_freeBudget_):
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
                    d_12_constrainedSteps_: int
                    d_12_constrainedSteps_ = 0
                    d_13_spanBudget_: int
                    d_13_spanBudget_ = (maxSteps) - (d_3_steps_)
                    with _dafny.label("1_4_0_0_0"):
                        while ((d_12_constrainedSteps_) < (d_13_spanBudget_)) and (insideConstrainedOut):
                            with _dafny.c_label("1_4_0_0_0"):
                                if (parser).IsCompletePrefix(currentConstrainedOut):
                                    d_14_closedG_: _dafny.Seq
                                    d_15_closedI_: bool
                                    d_16_closedC_: _dafny.Seq
                                    out7_: _dafny.Seq
                                    out8_: bool
                                    out9_: _dafny.Seq
                                    out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_14_closedG_ = out7_
                                    d_15_closedI_ = out8_
                                    d_16_closedC_ = out9_
                                    generated = d_14_closedG_
                                    insideConstrainedOut = d_15_closedI_
                                    currentConstrainedOut = d_16_closedC_
                                    d_12_constrainedSteps_ = (d_12_constrainedSteps_) + (1)
                                    raise _dafny.Break("1_4_0_0_0")
                                elif True:
                                    d_17_constrainedPrompt_: _dafny.Seq
                                    d_17_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                    d_18_next_: _dafny.Seq
                                    out10_: _dafny.Seq
                                    out10_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_17_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                                    d_18_next_ = out10_
                                    d_12_constrainedSteps_ = (d_12_constrainedSteps_) + (1)
                                    if (d_18_next_) == (eosToken):
                                        raise _dafny.Break("1_4_0_0_0")
                                    elif True:
                                        d_19_appendedG_: _dafny.Seq
                                        d_20_appendedI_: bool
                                        d_21_appendedC_: _dafny.Seq
                                        out11_: _dafny.Seq
                                        out12_: bool
                                        out13_: _dafny.Seq
                                        out11_, out12_, out13_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_18_next_)
                                        d_19_appendedG_ = out11_
                                        d_20_appendedI_ = out12_
                                        d_21_appendedC_ = out13_
                                        generated = d_19_appendedG_
                                        insideConstrainedOut = d_20_appendedI_
                                        currentConstrainedOut = d_21_appendedC_
                                pass
                        pass
                    d_3_steps_ = (d_3_steps_) + (d_12_constrainedSteps_)
                    if (insideConstrainedOut) and ((d_3_steps_) < (maxSteps)):
                        d_22_finalBudget_: int
                        d_22_finalBudget_ = (maxSteps) - (d_3_steps_)
                        d_23_cg2_: _dafny.Seq
                        d_24_ci2_: bool
                        d_25_cc2_: _dafny.Seq
                        out14_: _dafny.Seq
                        out15_: bool
                        out16_: _dafny.Seq
                        out14_, out15_, out16_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_22_finalBudget_)
                        d_23_cg2_ = out14_
                        d_24_ci2_ = out15_
                        d_25_cc2_ = out16_
                        generated = d_23_cg2_
                        insideConstrainedOut = d_24_ci2_
                        currentConstrainedOut = d_25_cc2_
                        d_3_steps_ = maxSteps
            cost = d_3_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

