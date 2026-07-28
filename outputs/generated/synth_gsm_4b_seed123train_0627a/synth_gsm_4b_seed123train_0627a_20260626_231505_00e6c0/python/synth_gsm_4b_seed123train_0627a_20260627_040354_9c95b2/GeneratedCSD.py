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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "For each calculation step and the final answer, use << expr >> with only symbolic expressions inside (no curly braces, no words)."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        with _dafny.label("0"):
            while (d_2_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_3_remaining_: int
                        d_3_remaining_ = (maxSteps) - (d_2_steps_)
                        d_4_reserveForSpan_: int
                        d_4_reserveForSpan_ = 41
                        d_5_chunkBudget_: int = int(0)
                        if (d_3_remaining_) <= (d_4_reserveForSpan_):
                            d_5_chunkBudget_ = d_3_remaining_
                        elif True:
                            d_5_chunkBudget_ = (d_3_remaining_) - (d_4_reserveForSpan_)
                        d_6_chunkGenerated_: _dafny.Seq
                        d_7_stoppedOnOpenSpan_: bool
                        d_8_stoppedOnEos_: bool
                        d_9_stepsUsed_: int
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: bool
                        out3_: int
                        out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_5_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_6_chunkGenerated_ = out0_
                        d_7_stoppedOnOpenSpan_ = out1_
                        d_8_stoppedOnEos_ = out2_
                        d_9_stepsUsed_ = out3_
                        d_2_steps_ = (d_2_steps_) + (d_9_stepsUsed_)
                        generated = d_6_chunkGenerated_
                        if d_7_stoppedOnOpenSpan_:
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                            generated = out4_
                            insideConstrainedOut = out5_
                            currentConstrainedOut = out6_
                        elif d_8_stoppedOnEos_:
                            d_10_remaining2_: int
                            d_10_remaining2_ = (maxSteps) - (d_2_steps_)
                            if (d_10_remaining2_) >= (2):
                                out7_: _dafny.Seq
                                out8_: bool
                                out9_: _dafny.Seq
                                out7_, out8_, out9_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                generated = out7_
                                insideConstrainedOut = out8_
                                currentConstrainedOut = out9_
                                d_2_steps_ = (d_2_steps_) + (1)
                            elif True:
                                raise _dafny.Break("0")
                        elif True:
                            d_11_remaining2_: int
                            d_11_remaining2_ = (maxSteps) - (d_2_steps_)
                            if (d_11_remaining2_) >= (1):
                                out10_: _dafny.Seq
                                out11_: bool
                                out12_: _dafny.Seq
                                out10_, out11_, out12_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                generated = out10_
                                insideConstrainedOut = out11_
                                currentConstrainedOut = out12_
                                d_2_steps_ = (d_2_steps_) + (1)
                            elif True:
                                raise _dafny.Break("0")
                    elif True:
                        d_12_remaining_: int
                        d_12_remaining_ = (maxSteps) - (d_2_steps_)
                        if (d_12_remaining_) == (0):
                            raise _dafny.Break("0")
                        d_13_spanCap_: int
                        d_13_spanCap_ = 50
                        d_14_closeBudget_: int
                        if (d_12_remaining_) > (d_13_spanCap_):
                            d_14_closeBudget_ = d_13_spanCap_
                        elif True:
                            d_14_closeBudget_ = d_12_remaining_
                        d_15_cg_: _dafny.Seq
                        d_16_ci_: bool
                        d_17_cc_: _dafny.Seq
                        out13_: _dafny.Seq
                        out14_: bool
                        out15_: _dafny.Seq
                        out13_, out14_, out15_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_14_closeBudget_)
                        d_15_cg_ = out13_
                        d_16_ci_ = out14_
                        d_17_cc_ = out15_
                        generated = d_15_cg_
                        insideConstrainedOut = d_16_ci_
                        currentConstrainedOut = d_17_cc_
                        d_2_steps_ = (d_2_steps_) + (d_14_closeBudget_)
                    pass
            pass
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

