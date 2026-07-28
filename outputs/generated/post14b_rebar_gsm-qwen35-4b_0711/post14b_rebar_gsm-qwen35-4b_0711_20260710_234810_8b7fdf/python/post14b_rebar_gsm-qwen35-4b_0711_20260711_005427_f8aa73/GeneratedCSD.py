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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step. Show your reasoning in plain text. For the final answer, write an arithmetic expression using only variable names (no curly braces), digits, and operators (+, -, *, /, //, (, )) inside << >>. Example: <<a + b * c>>. Do not use {variable} syntax inside << >>."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        if (maxSteps) == (0):
            cost = 0
        elif True:
            d_2_steps_: int
            d_2_steps_ = 0
            d_3_reasoningBudget_: int
            d_3_reasoningBudget_ = _dafny.euclidian_division((maxSteps) * (3), 4)
            if (d_3_reasoningBudget_) == (0):
                d_3_reasoningBudget_ = 1
            if (d_3_reasoningBudget_) >= (maxSteps):
                d_3_reasoningBudget_ = (maxSteps) - (1)
            with _dafny.label("1_0"):
                while ((d_2_steps_) < (d_3_reasoningBudget_)) and (not(insideConstrainedOut)):
                    with _dafny.c_label("1_0"):
                        d_4_chunkBudget_: int
                        d_4_chunkBudget_ = (d_3_reasoningBudget_) - (d_2_steps_)
                        if (d_4_chunkBudget_) > (32):
                            d_4_chunkBudget_ = 32
                        d_5_genOut_: _dafny.Seq
                        d_6_stoppedOnOpen_: bool
                        d_7_stoppedOnEos_: bool
                        d_8_stepsUsed_: int
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: bool
                        out3_: int
                        out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_4_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_5_genOut_ = out0_
                        d_6_stoppedOnOpen_ = out1_
                        d_7_stoppedOnEos_ = out2_
                        d_8_stepsUsed_ = out3_
                        d_2_steps_ = (d_2_steps_) + (d_8_stepsUsed_)
                        generated = d_5_genOut_
                        if d_7_stoppedOnEos_:
                            raise _dafny.Break("1_0")
                        elif d_6_stoppedOnOpen_:
                            d_9_g2_: _dafny.Seq
                            d_10_ic2_: bool
                            d_11_cc2_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                            d_9_g2_ = out4_
                            d_10_ic2_ = out5_
                            d_11_cc2_ = out6_
                            generated = d_9_g2_
                            insideConstrainedOut = d_10_ic2_
                            currentConstrainedOut = d_11_cc2_
                            raise _dafny.Break("1_0")
                        pass
                pass
            if (not(insideConstrainedOut)) and ((d_2_steps_) < (maxSteps)):
                d_12_g2_: _dafny.Seq
                d_13_ic2_: bool
                d_14_cc2_: _dafny.Seq
                out7_: _dafny.Seq
                out8_: bool
                out9_: _dafny.Seq
                out7_, out8_, out9_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                d_12_g2_ = out7_
                d_13_ic2_ = out8_
                d_14_cc2_ = out9_
                generated = d_12_g2_
                insideConstrainedOut = d_13_ic2_
                currentConstrainedOut = d_14_cc2_
                d_2_steps_ = (d_2_steps_) + (1)
            if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
                d_15_closeBudget_: int
                d_15_closeBudget_ = (maxSteps) - (d_2_steps_)
                d_16_cg_: _dafny.Seq
                d_17_ci_: bool
                d_18_cc_: _dafny.Seq
                out10_: _dafny.Seq
                out11_: bool
                out12_: _dafny.Seq
                out10_, out11_, out12_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_15_closeBudget_)
                d_16_cg_ = out10_
                d_17_ci_ = out11_
                d_18_cc_ = out12_
                generated = d_16_cg_
                insideConstrainedOut = d_17_ci_
                currentConstrainedOut = d_18_cc_
                d_2_steps_ = (d_2_steps_) + (d_15_closeBudget_)
            if (not(insideConstrainedOut)) and ((d_2_steps_) < (maxSteps)):
                d_19_remainingBudget_: int
                d_19_remainingBudget_ = (maxSteps) - (d_2_steps_)
                d_20_innerSteps_: int
                d_20_innerSteps_ = 0
                with _dafny.label("1_5_0"):
                    while (d_20_innerSteps_) < (d_19_remainingBudget_):
                        with _dafny.c_label("1_5_0"):
                            d_21_g2_: _dafny.Seq
                            d_22_ic2_: bool
                            d_23_cc2_: _dafny.Seq
                            d_24_done_: bool
                            out13_: _dafny.Seq
                            out14_: bool
                            out15_: _dafny.Seq
                            out16_: bool
                            out13_, out14_, out15_, out16_ = (d_0_helpers_).ManagedStep(lm, parser, prompt, generated, insideConstrainedOut, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                            d_21_g2_ = out13_
                            d_22_ic2_ = out14_
                            d_23_cc2_ = out15_
                            d_24_done_ = out16_
                            generated = d_21_g2_
                            insideConstrainedOut = d_22_ic2_
                            currentConstrainedOut = d_23_cc2_
                            d_20_innerSteps_ = (d_20_innerSteps_) + (1)
                            if d_24_done_:
                                raise _dafny.Break("1_5_0")
                            pass
                    pass
                d_2_steps_ = (d_2_steps_) + (d_20_innerSteps_)
            cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

