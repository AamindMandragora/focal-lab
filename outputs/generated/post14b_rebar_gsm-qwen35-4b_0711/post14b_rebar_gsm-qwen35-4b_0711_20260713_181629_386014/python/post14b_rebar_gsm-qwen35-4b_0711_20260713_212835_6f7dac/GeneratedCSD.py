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
            d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Think through the problem step by step in plain text. Do NOT use << or >> anywhere in your reasoning. Only after you have finished ALL reasoning, write your final arithmetic expression inside << >> exactly once at the very end. The expression must use only: variable names, numbers, +, -, *, /, //, %, (, ), int(). No LaTeX, no $, no {}, no **, no backticks. Open << exactly once for the final answer only."))
            (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
            d_2_steps_: int
            d_2_steps_ = 0
            d_3_prefixBudget_: int
            d_3_prefixBudget_ = _dafny.euclidian_division((maxSteps) * (13), 20)
            if (d_3_prefixBudget_) >= (maxSteps):
                d_3_prefixBudget_ = (maxSteps) - (1)
            with _dafny.label("1_0"):
                while (d_2_steps_) < (d_3_prefixBudget_):
                    with _dafny.c_label("1_0"):
                        if insideConstrainedOut:
                            d_4_closeBudget_: int
                            d_4_closeBudget_ = (d_3_prefixBudget_) - (d_2_steps_)
                            if (d_4_closeBudget_) > (10):
                                d_4_closeBudget_ = 10
                            if (d_4_closeBudget_) == (0):
                                raise _dafny.Break("1_0")
                            d_5_cg_: _dafny.Seq
                            d_6_ci_: bool
                            d_7_cc_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_4_closeBudget_)
                            d_5_cg_ = out0_
                            d_6_ci_ = out1_
                            d_7_cc_ = out2_
                            generated = d_5_cg_
                            insideConstrainedOut = d_6_ci_
                            currentConstrainedOut = d_7_cc_
                            d_2_steps_ = (d_2_steps_) + (d_4_closeBudget_)
                        elif True:
                            d_8_chunkBudget_: int
                            d_8_chunkBudget_ = (d_3_prefixBudget_) - (d_2_steps_)
                            if (d_8_chunkBudget_) > (30):
                                d_8_chunkBudget_ = 30
                            if (d_8_chunkBudget_) == (0):
                                raise _dafny.Break("1_0")
                            d_9_gOut_: _dafny.Seq
                            d_10_stoppedOnOpen_: bool
                            d_11_stoppedOnEos_: bool
                            d_12_stepsUsed_: int
                            out3_: _dafny.Seq
                            out4_: bool
                            out5_: bool
                            out6_: int
                            out3_, out4_, out5_, out6_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_8_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_9_gOut_ = out3_
                            d_10_stoppedOnOpen_ = out4_
                            d_11_stoppedOnEos_ = out5_
                            d_12_stepsUsed_ = out6_
                            generated = d_9_gOut_
                            d_2_steps_ = (d_2_steps_) + (d_12_stepsUsed_)
                            if d_11_stoppedOnEos_:
                                raise _dafny.Break("1_0")
                            if d_10_stoppedOnOpen_:
                                d_13_g2_: _dafny.Seq
                                d_14_ic2_: bool
                                d_15_cc2_: _dafny.Seq
                                out7_: _dafny.Seq
                                out8_: bool
                                out9_: _dafny.Seq
                                out7_, out8_, out9_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_13_g2_ = out7_
                                d_14_ic2_ = out8_
                                d_15_cc2_ = out9_
                                generated = d_13_g2_
                                insideConstrainedOut = d_14_ic2_
                                currentConstrainedOut = d_15_cc2_
                                d_16_intBudget_: int
                                d_16_intBudget_ = (d_3_prefixBudget_) - (d_2_steps_)
                                if (d_16_intBudget_) > (8):
                                    d_16_intBudget_ = 8
                                if (d_16_intBudget_) > (0):
                                    d_17_cg2_: _dafny.Seq
                                    d_18_ci2_: bool
                                    d_19_cc3_: _dafny.Seq
                                    out10_: _dafny.Seq
                                    out11_: bool
                                    out12_: _dafny.Seq
                                    out10_, out11_, out12_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_16_intBudget_)
                                    d_17_cg2_ = out10_
                                    d_18_ci2_ = out11_
                                    d_19_cc3_ = out12_
                                    generated = d_17_cg2_
                                    insideConstrainedOut = d_18_ci2_
                                    currentConstrainedOut = d_19_cc3_
                                    d_2_steps_ = (d_2_steps_) + (d_16_intBudget_)
                        pass
                pass
            if (d_2_steps_) < (maxSteps):
                d_20_remainBudget_: int
                d_20_remainBudget_ = (maxSteps) - (d_2_steps_)
                d_21_prefixBudget2_: int
                d_21_prefixBudget2_ = 5
                if (d_21_prefixBudget2_) > (d_20_remainBudget_):
                    d_21_prefixBudget2_ = d_20_remainBudget_
                d_22_gFinal_: _dafny.Seq
                d_23_icFinal_: bool
                d_24_ccFinal_: _dafny.Seq
                out13_: _dafny.Seq
                out14_: bool
                out15_: _dafny.Seq
                out13_, out14_, out15_ = (d_0_helpers_).GenerateWithPrefixAndManagedSpan(lm, parser, prompt, generated, insideConstrainedOut, currentConstrainedOut, d_20_remainBudget_, d_21_prefixBudget2_, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                d_22_gFinal_ = out13_
                d_23_icFinal_ = out14_
                d_24_ccFinal_ = out15_
                generated = d_22_gFinal_
                insideConstrainedOut = d_23_icFinal_
                currentConstrainedOut = d_24_ccFinal_
                d_2_steps_ = maxSteps
            cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

