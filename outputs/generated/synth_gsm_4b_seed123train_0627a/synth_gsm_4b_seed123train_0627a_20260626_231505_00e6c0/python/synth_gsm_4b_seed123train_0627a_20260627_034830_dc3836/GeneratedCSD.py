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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. Wrap each intermediate calculation and the final answer in << >>. Inside << >> write ONLY symbolic expressions using variable names without braces, numbers, +, -, *, /, (, ). Example: <<n * price + extra>>. Keep expressions concise. Always close << with >>."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_effectiveMax_: int
        if (maxSteps) > (320):
            d_3_effectiveMax_ = 320
        elif True:
            d_3_effectiveMax_ = maxSteps
        with _dafny.label("0"):
            while (d_2_steps_) < (d_3_effectiveMax_):
                with _dafny.c_label("0"):
                    if (d_2_steps_) >= (d_3_effectiveMax_):
                        raise _dafny.Break("0")
                    if not(insideConstrainedOut):
                        d_4_remaining_: int
                        d_4_remaining_ = (d_3_effectiveMax_) - (d_2_steps_)
                        d_5_chunkBudget_: int = int(0)
                        if (d_4_remaining_) <= (15):
                            d_5_chunkBudget_ = d_4_remaining_
                        elif True:
                            d_6_available_: int
                            d_6_available_ = (d_4_remaining_) - (15)
                            if (d_6_available_) > (200):
                                d_5_chunkBudget_ = 200
                            elif True:
                                d_5_chunkBudget_ = d_6_available_
                        if (d_5_chunkBudget_) == (0):
                            raise _dafny.Break("0")
                        d_7_chunkGenerated_: _dafny.Seq
                        d_8_stoppedOnOpenSpan_: bool
                        d_9_stoppedOnEos_: bool
                        d_10_stepsUsed_: int
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: bool
                        out3_: int
                        out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_5_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_7_chunkGenerated_ = out0_
                        d_8_stoppedOnOpenSpan_ = out1_
                        d_9_stoppedOnEos_ = out2_
                        d_10_stepsUsed_ = out3_
                        d_2_steps_ = (d_2_steps_) + (d_10_stepsUsed_)
                        generated = d_7_chunkGenerated_
                        if d_8_stoppedOnOpenSpan_:
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                            generated = out4_
                            insideConstrainedOut = out5_
                            currentConstrainedOut = out6_
                        elif d_9_stoppedOnEos_:
                            raise _dafny.Break("0")
                    elif True:
                        d_11_remaining_: int
                        d_11_remaining_ = (d_3_effectiveMax_) - (d_2_steps_)
                        if (d_11_remaining_) == (0):
                            raise _dafny.Break("0")
                        d_12_spanBudget_: int
                        if (d_11_remaining_) > (45):
                            d_12_spanBudget_ = 45
                        elif True:
                            d_12_spanBudget_ = d_11_remaining_
                        d_13_cg_: _dafny.Seq
                        d_14_ci_: bool
                        d_15_cc_: _dafny.Seq
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: _dafny.Seq
                        out7_, out8_, out9_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_12_spanBudget_)
                        d_13_cg_ = out7_
                        d_14_ci_ = out8_
                        d_15_cc_ = out9_
                        generated = d_13_cg_
                        insideConstrainedOut = d_14_ci_
                        currentConstrainedOut = d_15_cc_
                        d_2_steps_ = (d_2_steps_) + (d_12_spanBudget_)
                    pass
            pass
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

