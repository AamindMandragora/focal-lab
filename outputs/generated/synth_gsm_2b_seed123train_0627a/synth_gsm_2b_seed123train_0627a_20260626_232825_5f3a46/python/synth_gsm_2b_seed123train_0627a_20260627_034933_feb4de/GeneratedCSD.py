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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. Then write ONLY the final arithmetic expression as <<expr>> using variable names, +, -, *, /. Keep the expression SHORT (under 8 tokens). Do NOT write reasoning inside << >>. Example: <<n * price + tax>>."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_hardFreeCap_: int
        d_3_hardFreeCap_ = 200
        d_4_minCloseReserve_: int
        d_4_minCloseReserve_ = 30
        d_5_freeChunkBudget_: int = int(0)
        if (maxSteps) > (d_4_minCloseReserve_):
            d_6_available_: int
            d_6_available_ = (maxSteps) - (d_4_minCloseReserve_)
            if (d_6_available_) < (d_3_hardFreeCap_):
                d_5_freeChunkBudget_ = d_6_available_
            elif True:
                d_5_freeChunkBudget_ = d_3_hardFreeCap_
        elif True:
            d_5_freeChunkBudget_ = 0
        if (not(insideConstrainedOut)) and ((d_2_steps_) < (d_5_freeChunkBudget_)):
            d_7_chunkBudget_: int
            d_7_chunkBudget_ = (d_5_freeChunkBudget_) - (d_2_steps_)
            d_8_cg_: _dafny.Seq
            d_9_stoppedOnOpenSpan_: bool
            d_10_stoppedOnEos_: bool
            d_11_stepsUsed_: int
            out0_: _dafny.Seq
            out1_: bool
            out2_: bool
            out3_: int
            out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_7_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
            d_8_cg_ = out0_
            d_9_stoppedOnOpenSpan_ = out1_
            d_10_stoppedOnEos_ = out2_
            d_11_stepsUsed_ = out3_
            generated = d_8_cg_
            d_2_steps_ = (d_2_steps_) + (d_11_stepsUsed_)
            if d_10_stoppedOnEos_:
                cost = d_2_steps_
                return generated, insideConstrainedOut, currentConstrainedOut, cost
            if d_9_stoppedOnOpenSpan_:
                out4_: _dafny.Seq
                out5_: bool
                out6_: _dafny.Seq
                out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                generated = out4_
                insideConstrainedOut = out5_
                currentConstrainedOut = out6_
        if (not(insideConstrainedOut)) and ((d_2_steps_) < (maxSteps)):
            out7_: _dafny.Seq
            out8_: bool
            out9_: _dafny.Seq
            out7_, out8_, out9_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            generated = out7_
            insideConstrainedOut = out8_
            currentConstrainedOut = out9_
            d_2_steps_ = (d_2_steps_) + (1)
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_12_remaining_: int
            d_12_remaining_ = (maxSteps) - (d_2_steps_)
            d_13_tightCap_: int
            d_13_tightCap_ = 25
            d_14_closeBudget_: int
            if (d_12_remaining_) < (d_13_tightCap_):
                d_14_closeBudget_ = d_12_remaining_
            elif True:
                d_14_closeBudget_ = d_13_tightCap_
            out10_: _dafny.Seq
            out11_: bool
            out12_: _dafny.Seq
            out10_, out11_, out12_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_14_closeBudget_)
            generated = out10_
            insideConstrainedOut = out11_
            currentConstrainedOut = out12_
            d_2_steps_ = (d_2_steps_) + (d_14_closeBudget_)
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

