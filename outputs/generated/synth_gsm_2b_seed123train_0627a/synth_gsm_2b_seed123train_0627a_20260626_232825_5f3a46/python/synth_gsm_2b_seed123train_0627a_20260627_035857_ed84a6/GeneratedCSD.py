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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. At the end, write your final arithmetic expression as <<expression>> using only variable names and +, -, *, /. Example: <<n * price + tax>>. Keep the expression concise."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_reserveForConstrained_: int
        d_3_reserveForConstrained_ = 100
        d_4_freeChunkBudget_: int = int(0)
        if (maxSteps) > (d_3_reserveForConstrained_):
            d_4_freeChunkBudget_ = (maxSteps) - (d_3_reserveForConstrained_)
        elif True:
            d_4_freeChunkBudget_ = 0
        if (not(insideConstrainedOut)) and ((d_4_freeChunkBudget_) > (0)):
            d_5_chunkBudget_: int
            d_5_chunkBudget_ = d_4_freeChunkBudget_
            d_6_cg_: _dafny.Seq
            d_7_stoppedOnOpenSpan_: bool
            d_8_stoppedOnEos_: bool
            d_9_stepsUsed_: int
            out0_: _dafny.Seq
            out1_: bool
            out2_: bool
            out3_: int
            out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_5_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
            d_6_cg_ = out0_
            d_7_stoppedOnOpenSpan_ = out1_
            d_8_stoppedOnEos_ = out2_
            d_9_stepsUsed_ = out3_
            generated = d_6_cg_
            d_2_steps_ = (d_2_steps_) + (d_9_stepsUsed_)
            if d_8_stoppedOnEos_:
                cost = d_2_steps_
                return generated, insideConstrainedOut, currentConstrainedOut, cost
            if d_7_stoppedOnOpenSpan_:
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
        d_10_constrainedBudget_: int = int(0)
        if (d_2_steps_) < (maxSteps):
            d_11_remaining_: int
            d_11_remaining_ = (maxSteps) - (d_2_steps_)
            if (d_11_remaining_) < (100):
                d_10_constrainedBudget_ = d_11_remaining_
            elif True:
                d_10_constrainedBudget_ = 100
        elif True:
            d_10_constrainedBudget_ = 0
        d_12_constrainedSteps_: int
        d_12_constrainedSteps_ = 0
        with _dafny.label("0"):
            while (insideConstrainedOut) and ((d_12_constrainedSteps_) < (d_10_constrainedBudget_)):
                with _dafny.c_label("0"):
                    d_13_cg_: _dafny.Seq
                    d_14_ci_: bool
                    d_15_cc_: _dafny.Seq
                    d_16_closed_: bool
                    out10_: _dafny.Seq
                    out11_: bool
                    out12_: _dafny.Seq
                    out13_: bool
                    out10_, out11_, out12_, out13_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                    d_13_cg_ = out10_
                    d_14_ci_ = out11_
                    d_15_cc_ = out12_
                    d_16_closed_ = out13_
                    if d_16_closed_:
                        generated = d_13_cg_
                        insideConstrainedOut = d_14_ci_
                        currentConstrainedOut = d_15_cc_
                        d_12_constrainedSteps_ = (d_12_constrainedSteps_) + (1)
                    elif True:
                        d_17_constrainedPrompt_: _dafny.Seq
                        d_17_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_18_next_: _dafny.Seq
                        out14_: _dafny.Seq
                        out14_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, d_17_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), eosToken)
                        d_18_next_ = out14_
                        d_12_constrainedSteps_ = (d_12_constrainedSteps_) + (1)
                        if (d_18_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_19_ag_: _dafny.Seq
                            d_20_ai_: bool
                            d_21_ac_: _dafny.Seq
                            out15_: _dafny.Seq
                            out16_: bool
                            out17_: _dafny.Seq
                            out15_, out16_, out17_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_18_next_)
                            d_19_ag_ = out15_
                            d_20_ai_ = out16_
                            d_21_ac_ = out17_
                            generated = d_19_ag_
                            insideConstrainedOut = d_20_ai_
                            currentConstrainedOut = d_21_ac_
                    pass
            pass
        d_2_steps_ = (d_2_steps_) + (d_12_constrainedSteps_)
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_22_finalBudget_: int
            d_22_finalBudget_ = (maxSteps) - (d_2_steps_)
            out18_: _dafny.Seq
            out19_: bool
            out20_: _dafny.Seq
            out18_, out19_, out20_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_22_finalBudget_)
            generated = out18_
            insideConstrainedOut = out19_
            currentConstrainedOut = out20_
            d_2_steps_ = maxSteps
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

