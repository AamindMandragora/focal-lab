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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve this math problem step by step, then write ONLY the final answer expression inside << >> at the very end. The expression inside << >> must use only variable names, numbers, +, -, *, /, //, %, (, ) - no LaTeX, no curly braces, no text. Example: <<n * price + extra>>. Keep the expression concise and correct."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_freeCapSteps_: int
        d_3_freeCapSteps_ = 350
        with _dafny.label("0"):
            while (((d_2_steps_) < (maxSteps)) and ((d_2_steps_) < (d_3_freeCapSteps_))) and (not(insideConstrainedOut)):
                with _dafny.c_label("0"):
                    d_4_remaining_: int
                    d_4_remaining_ = (maxSteps) - (d_2_steps_)
                    d_5_chunkSize_: int
                    d_5_chunkSize_ = 20
                    if (d_4_remaining_) < (d_5_chunkSize_):
                        d_5_chunkSize_ = d_4_remaining_
                    if (d_5_chunkSize_) == (0):
                        raise _dafny.Break("0")
                    d_6_generatedOut_: _dafny.Seq
                    d_7_stoppedOnOpenSpan_: bool
                    d_8_stoppedOnEos_: bool
                    d_9_chunkUsed_: int
                    out0_: _dafny.Seq
                    out1_: bool
                    out2_: bool
                    out3_: int
                    out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_5_chunkSize_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                    d_6_generatedOut_ = out0_
                    d_7_stoppedOnOpenSpan_ = out1_
                    d_8_stoppedOnEos_ = out2_
                    d_9_chunkUsed_ = out3_
                    generated = d_6_generatedOut_
                    d_2_steps_ = (d_2_steps_) + (d_9_chunkUsed_)
                    if d_7_stoppedOnOpenSpan_:
                        out4_: _dafny.Seq
                        out5_: bool
                        out6_: _dafny.Seq
                        out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                        generated = out4_
                        insideConstrainedOut = out5_
                        currentConstrainedOut = out6_
                    elif d_8_stoppedOnEos_:
                        raise _dafny.Break("0")
                    pass
            pass
        if (not(insideConstrainedOut)) and ((d_2_steps_) < (maxSteps)):
            out7_: _dafny.Seq
            out8_: bool
            out9_: _dafny.Seq
            out7_, out8_, out9_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            generated = out7_
            insideConstrainedOut = out8_
            currentConstrainedOut = out9_
            d_2_steps_ = (d_2_steps_) + (1)
        d_10_closeReserve_: int
        d_10_closeReserve_ = 50
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_11_available_: int
            d_11_available_ = (maxSteps) - (d_2_steps_)
            d_12_rolloutBudget_: int
            d_12_rolloutBudget_ = 0
            if (d_11_available_) > (d_10_closeReserve_):
                d_12_rolloutBudget_ = (d_11_available_) - (d_10_closeReserve_)
            if (d_12_rolloutBudget_) > (80):
                d_12_rolloutBudget_ = 80
            if (d_12_rolloutBudget_) >= (1):
                d_13_stable_: _dafny.Seq
                d_13_stable_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                d_14_constrainedPrompt_: _dafny.Seq
                d_14_constrainedPrompt_ = (prompt) + (d_13_stable_)
                d_15_penalties_: _dafny.Seq
                d_15_penalties_ = _dafny.SeqWithoutIsStrInference([])
                d_16_rolloutGen_: _dafny.Seq
                d_17_rolloutSteps_: int
                d_18_rolloutEos_: bool
                out10_: _dafny.Seq
                out11_: int
                out12_: bool
                out10_, out11_, out12_ = (d_0_helpers_).RolloutConstrainedWithPenalties(lm, parser, d_14_constrainedPrompt_, currentConstrainedOut, d_12_rolloutBudget_, d_15_penalties_, _dafny.BigRational('0e0'), eosToken)
                d_16_rolloutGen_ = out10_
                d_17_rolloutSteps_ = out11_
                d_18_rolloutEos_ = out12_
                generated = (d_13_stable_) + (d_16_rolloutGen_)
                currentConstrainedOut = d_16_rolloutGen_
                d_2_steps_ = (d_2_steps_) + (d_17_rolloutSteps_)
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_19_closeBudget_: int
            d_19_closeBudget_ = (maxSteps) - (d_2_steps_)
            out13_: _dafny.Seq
            out14_: bool
            out15_: _dafny.Seq
            out13_, out14_, out15_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_19_closeBudget_)
            generated = out13_
            insideConstrainedOut = out14_
            currentConstrainedOut = out15_
            d_2_steps_ = maxSteps
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

