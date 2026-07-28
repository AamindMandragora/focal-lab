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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Output exactly one line: SQL: <<SELECT ...>> using only the exact table names and column names from the schema. The query must be complete and correct. Use lowercase keywords."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        if (not(insideConstrainedOut)) and ((d_2_steps_) < (maxSteps)):
            d_3_chunkBudget_: int
            if (10) <= ((maxSteps) - (d_2_steps_)):
                d_3_chunkBudget_ = 10
            elif True:
                d_3_chunkBudget_ = (maxSteps) - (d_2_steps_)
            if (d_3_chunkBudget_) >= (1):
                d_4_cg_: _dafny.Seq
                d_5_stoppedOnOpenSpan_: bool
                d_6_stoppedOnEos_: bool
                d_7_stepsUsed_: int
                out0_: _dafny.Seq
                out1_: bool
                out2_: bool
                out3_: int
                out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_3_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                d_4_cg_ = out0_
                d_5_stoppedOnOpenSpan_ = out1_
                d_6_stoppedOnEos_ = out2_
                d_7_stepsUsed_ = out3_
                d_2_steps_ = (d_2_steps_) + (d_7_stepsUsed_)
                generated = d_4_cg_
                if d_6_stoppedOnEos_:
                    cost = d_2_steps_
                    return generated, insideConstrainedOut, currentConstrainedOut, cost
                if d_5_stoppedOnOpenSpan_:
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
        d_8_closeReserve_: int
        if (maxSteps) >= (20):
            d_8_closeReserve_ = 20
        elif (maxSteps) >= (5):
            d_8_closeReserve_ = 5
        elif True:
            d_8_closeReserve_ = maxSteps
        d_9_loopBound_: int
        if (maxSteps) > (d_8_closeReserve_):
            d_9_loopBound_ = (maxSteps) - (d_8_closeReserve_)
        elif True:
            d_9_loopBound_ = 0
        with _dafny.label("0"):
            while (insideConstrainedOut) and ((d_2_steps_) < (d_9_loopBound_)):
                with _dafny.c_label("0"):
                    d_10_cg_: _dafny.Seq
                    d_11_ci_: bool
                    d_12_cc_: _dafny.Seq
                    d_13_closed_: bool
                    out10_: _dafny.Seq
                    out11_: bool
                    out12_: _dafny.Seq
                    out13_: bool
                    out10_, out11_, out12_, out13_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                    d_10_cg_ = out10_
                    d_11_ci_ = out11_
                    d_12_cc_ = out12_
                    d_13_closed_ = out13_
                    d_2_steps_ = (d_2_steps_) + (1)
                    if d_13_closed_:
                        generated = d_10_cg_
                        insideConstrainedOut = d_11_ci_
                        currentConstrainedOut = d_12_cc_
                    elif True:
                        d_14_stable_: _dafny.Seq
                        d_14_stable_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_15_constrainedPrompt_: _dafny.Seq
                        d_15_constrainedPrompt_ = (prompt) + (d_14_stable_)
                        d_16_next_: _dafny.Seq
                        out14_: _dafny.Seq
                        out14_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_15_constrainedPrompt_, currentConstrainedOut, eosToken)
                        d_16_next_ = out14_
                        if (d_16_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_17_appendedGenerated_: _dafny.Seq
                            d_18_appendedInside_: bool
                            d_19_appendedCurrent_: _dafny.Seq
                            out15_: _dafny.Seq
                            out16_: bool
                            out17_: _dafny.Seq
                            out15_, out16_, out17_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_16_next_)
                            d_17_appendedGenerated_ = out15_
                            d_18_appendedInside_ = out16_
                            d_19_appendedCurrent_ = out17_
                            generated = d_17_appendedGenerated_
                            insideConstrainedOut = d_18_appendedInside_
                            currentConstrainedOut = d_19_appendedCurrent_
                    pass
            pass
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_20_closeBudget_: int
            d_20_closeBudget_ = (maxSteps) - (d_2_steps_)
            d_21_cg_: _dafny.Seq
            d_22_ci_: bool
            d_23_cc_: _dafny.Seq
            out18_: _dafny.Seq
            out19_: bool
            out20_: _dafny.Seq
            out18_, out19_, out20_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_20_closeBudget_)
            d_21_cg_ = out18_
            d_22_ci_ = out19_
            d_23_cc_ = out20_
            generated = d_21_cg_
            insideConstrainedOut = d_22_ci_
            currentConstrainedOut = d_23_cc_
            d_2_steps_ = maxSteps
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

