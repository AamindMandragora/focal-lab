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
        d_1_steps_: int
        d_1_steps_ = 0
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math problem step by step. Write the final answer as a single arithmetic expression inside << >>. Use only: variable names (no braces), integers, +, -, *, /, //, %, int(), and parentheses. No {braces}, no **, no text inside << >>. Write << >> exactly ONCE at the very end. After closing >> do NOT write anything else.")))
        d_2_spanDone_: bool
        d_2_spanDone_ = False
        d_3_unconstrainedBudget_: int
        d_3_unconstrainedBudget_ = _dafny.euclidian_division((maxSteps) * (88), 100)
        if ((maxSteps) >= (60)) and ((d_3_unconstrainedBudget_) > ((maxSteps) - (60))):
            d_3_unconstrainedBudget_ = (maxSteps) - (60)
        if (d_3_unconstrainedBudget_) > (maxSteps):
            d_3_unconstrainedBudget_ = maxSteps
        with _dafny.label("0"):
            while ((d_1_steps_) < (d_3_unconstrainedBudget_)) and (not(d_2_spanDone_)):
                with _dafny.c_label("0"):
                    d_4_chunkBudget_: int
                    d_4_chunkBudget_ = (d_3_unconstrainedBudget_) - (d_1_steps_)
                    if (d_4_chunkBudget_) > (50):
                        d_4_chunkBudget_ = 50
                    if ((d_1_steps_) + (d_4_chunkBudget_)) > (d_3_unconstrainedBudget_):
                        d_4_chunkBudget_ = (d_3_unconstrainedBudget_) - (d_1_steps_)
                    if (d_4_chunkBudget_) == (0):
                        raise _dafny.Break("0")
                    d_5_chunkGenerated_: _dafny.Seq
                    d_6_stoppedOnOpenSpan_: bool
                    d_7_stoppedOnEos_: bool
                    d_8_chunkSteps_: int
                    out0_: _dafny.Seq
                    out1_: bool
                    out2_: bool
                    out3_: int
                    out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_4_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                    d_5_chunkGenerated_ = out0_
                    d_6_stoppedOnOpenSpan_ = out1_
                    d_7_stoppedOnEos_ = out2_
                    d_8_chunkSteps_ = out3_
                    generated = d_5_chunkGenerated_
                    d_1_steps_ = (d_1_steps_) + (d_8_chunkSteps_)
                    if (d_1_steps_) > (maxSteps):
                        d_1_steps_ = maxSteps
                    if d_7_stoppedOnEos_:
                        raise _dafny.Break("0")
                    if d_6_stoppedOnOpenSpan_:
                        out4_: _dafny.Seq
                        out5_: bool
                        out6_: _dafny.Seq
                        out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                        generated = out4_
                        insideConstrainedOut = out5_
                        currentConstrainedOut = out6_
                        if (d_1_steps_) < (maxSteps):
                            d_9_closeBudget1_: int
                            d_9_closeBudget1_ = (maxSteps) - (d_1_steps_)
                            if (d_9_closeBudget1_) > (80):
                                d_9_closeBudget1_ = 80
                            out7_: _dafny.Seq
                            out8_: bool
                            out9_: _dafny.Seq
                            out7_, out8_, out9_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_9_closeBudget1_)
                            generated = out7_
                            insideConstrainedOut = out8_
                            currentConstrainedOut = out9_
                            d_1_steps_ = (d_1_steps_) + (d_9_closeBudget1_)
                            if (d_1_steps_) > (maxSteps):
                                d_1_steps_ = maxSteps
                        if not(insideConstrainedOut):
                            d_2_spanDone_ = True
                            raise _dafny.Break("0")
                        raise _dafny.Break("0")
                    if (d_1_steps_) >= (d_3_unconstrainedBudget_):
                        raise _dafny.Break("0")
                    pass
            pass
        if ((not(d_2_spanDone_)) and (not(insideConstrainedOut))) and ((d_1_steps_) < (maxSteps)):
            d_10_remaining_: int
            d_10_remaining_ = (maxSteps) - (d_1_steps_)
            if (d_10_remaining_) >= (5):
                out10_: _dafny.Seq
                out11_: bool
                out12_: _dafny.Seq
                out10_, out11_, out12_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                generated = out10_
                insideConstrainedOut = out11_
                currentConstrainedOut = out12_
                d_1_steps_ = (d_1_steps_) + (1)
                if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
                    d_11_closeBudget2_: int
                    d_11_closeBudget2_ = (maxSteps) - (d_1_steps_)
                    if (d_11_closeBudget2_) > (70):
                        d_11_closeBudget2_ = 70
                    out13_: _dafny.Seq
                    out14_: bool
                    out15_: _dafny.Seq
                    out13_, out14_, out15_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_11_closeBudget2_)
                    generated = out13_
                    insideConstrainedOut = out14_
                    currentConstrainedOut = out15_
                    d_1_steps_ = (d_1_steps_) + (d_11_closeBudget2_)
                    if (d_1_steps_) > (maxSteps):
                        d_1_steps_ = maxSteps
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_12_closeBudget_: int
            d_12_closeBudget_ = (maxSteps) - (d_1_steps_)
            out16_: _dafny.Seq
            out17_: bool
            out18_: _dafny.Seq
            out16_, out17_, out18_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_12_closeBudget_)
            generated = out16_
            insideConstrainedOut = out17_
            currentConstrainedOut = out18_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

