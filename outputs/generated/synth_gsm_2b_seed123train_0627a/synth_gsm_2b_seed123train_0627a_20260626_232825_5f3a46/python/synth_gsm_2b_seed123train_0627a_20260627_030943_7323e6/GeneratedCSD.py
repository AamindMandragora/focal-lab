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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. Write your final answer as <<expression>> with simple arithmetic using variable names, +, -, *, /. Keep the expression concise. Example: <<n * price + tax>>."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_freeChunkBudget_: int
        if (maxSteps) > (500):
            d_3_freeChunkBudget_ = (maxSteps) - (500)
        elif True:
            if (maxSteps) > (1):
                d_3_freeChunkBudget_ = _dafny.euclidian_division(maxSteps, 2)
            elif True:
                d_3_freeChunkBudget_ = maxSteps
        if (not(insideConstrainedOut)) and ((d_2_steps_) < (d_3_freeChunkBudget_)):
            d_4_chunkBudget_: int
            d_4_chunkBudget_ = (d_3_freeChunkBudget_) - (d_2_steps_)
            d_5_cg_: _dafny.Seq
            d_6_stoppedOnOpenSpan_: bool
            d_7_stoppedOnEos_: bool
            d_8_stepsUsed_: int
            out0_: _dafny.Seq
            out1_: bool
            out2_: bool
            out3_: int
            out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_4_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
            d_5_cg_ = out0_
            d_6_stoppedOnOpenSpan_ = out1_
            d_7_stoppedOnEos_ = out2_
            d_8_stepsUsed_ = out3_
            generated = d_5_cg_
            d_2_steps_ = (d_2_steps_) + (d_8_stepsUsed_)
            if d_7_stoppedOnEos_:
                cost = d_2_steps_
                return generated, insideConstrainedOut, currentConstrainedOut, cost
            if d_6_stoppedOnOpenSpan_:
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
            d_9_closeReserve_: int
            d_9_closeReserve_ = 100
            d_10_remaining_: int
            d_10_remaining_ = (maxSteps) - (d_2_steps_)
            d_11_fillBudget_: int
            if (d_10_remaining_) > (d_9_closeReserve_):
                d_11_fillBudget_ = (d_10_remaining_) - (d_9_closeReserve_)
            elif True:
                d_11_fillBudget_ = 0
            if (d_11_fillBudget_) > (0):
                d_12_stablePrefix_: _dafny.Seq
                d_12_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                d_13_constrainedPrompt_: _dafny.Seq
                d_13_constrainedPrompt_ = (prompt) + (d_12_stablePrefix_)
                d_14_rolloutGen_: _dafny.Seq
                d_15_rolloutSteps_: int
                d_16_rolloutEos_: bool
                out10_: _dafny.Seq
                out11_: int
                out12_: bool
                out10_, out11_, out12_ = (d_0_helpers_).RolloutConstrainedWithPenalties(lm, parser, d_13_constrainedPrompt_, currentConstrainedOut, d_11_fillBudget_, currentConstrainedOut, _dafny.BigRational('3e0'), eosToken)
                d_14_rolloutGen_ = out10_
                d_15_rolloutSteps_ = out11_
                d_16_rolloutEos_ = out12_
                generated = (d_12_stablePrefix_) + (d_14_rolloutGen_)
                currentConstrainedOut = d_14_rolloutGen_
                d_2_steps_ = (d_2_steps_) + (d_15_rolloutSteps_)
                if d_16_rolloutEos_:
                    if (d_2_steps_) < (maxSteps):
                        d_17_closeBudget_: int
                        d_17_closeBudget_ = (maxSteps) - (d_2_steps_)
                        out13_: _dafny.Seq
                        out14_: bool
                        out15_: _dafny.Seq
                        out13_, out14_, out15_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_17_closeBudget_)
                        generated = out13_
                        insideConstrainedOut = out14_
                        currentConstrainedOut = out15_
                        d_2_steps_ = maxSteps
                    cost = d_2_steps_
                    return generated, insideConstrainedOut, currentConstrainedOut, cost
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_18_closeBudget_: int
            d_18_closeBudget_ = (maxSteps) - (d_2_steps_)
            out16_: _dafny.Seq
            out17_: bool
            out18_: _dafny.Seq
            out16_, out17_, out18_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_18_closeBudget_)
            generated = out16_
            insideConstrainedOut = out17_
            currentConstrainedOut = out18_
            d_2_steps_ = maxSteps
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

