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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Output format: SQL: <<SELECT query>>. Write the most direct SQL SELECT statement. Use exact table and column names from the schema. Keep the query simple and focused on what is asked. No subqueries unless necessary. No explanation."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        if (not(insideConstrainedOut)) and ((d_2_steps_) < (maxSteps)):
            d_3_chunkBudget_: int
            d_3_chunkBudget_ = 20
            d_4_remaining_: int
            d_4_remaining_ = (maxSteps) - (d_2_steps_)
            d_5_actualBudget_: int
            if (d_3_chunkBudget_) <= (d_4_remaining_):
                d_5_actualBudget_ = d_3_chunkBudget_
            elif True:
                d_5_actualBudget_ = d_4_remaining_
            if (d_5_actualBudget_) >= (1):
                d_6_cg_: _dafny.Seq
                d_7_stoppedOnOpenSpan_: bool
                d_8_stoppedOnEos_: bool
                d_9_stepsUsed_: int
                out0_: _dafny.Seq
                out1_: bool
                out2_: bool
                out3_: int
                out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_5_actualBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                d_6_cg_ = out0_
                d_7_stoppedOnOpenSpan_ = out1_
                d_8_stoppedOnEos_ = out2_
                d_9_stepsUsed_ = out3_
                d_2_steps_ = (d_2_steps_) + (d_9_stepsUsed_)
                generated = d_6_cg_
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
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_10_remaining_: int
            d_10_remaining_ = (maxSteps) - (d_2_steps_)
            d_11_closeReserve_: int
            if (d_10_remaining_) >= (30):
                d_11_closeReserve_ = 30
            elif (d_10_remaining_) >= (5):
                d_11_closeReserve_ = 5
            elif True:
                d_11_closeReserve_ = 1
            d_12_genBudget_: int
            if (d_10_remaining_) > (d_11_closeReserve_):
                d_12_genBudget_ = (d_10_remaining_) - (d_11_closeReserve_)
            elif True:
                d_12_genBudget_ = 0
            if (d_12_genBudget_) >= (1):
                d_13_stable_: _dafny.Seq
                d_13_stable_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                d_14_constrainedPrompt_: _dafny.Seq
                d_14_constrainedPrompt_ = (prompt) + (d_13_stable_)
                d_15_constrainedResult_: _dafny.Seq
                d_16_terminatedByEos_: bool
                out10_: _dafny.Seq
                out11_: bool
                out10_, out11_ = (d_0_helpers_).ConstrainedGeneration(lm, parser, d_14_constrainedPrompt_, d_12_genBudget_, eosToken)
                d_15_constrainedResult_ = out10_
                d_16_terminatedByEos_ = out11_
                generated = (d_13_stable_) + (d_15_constrainedResult_)
                currentConstrainedOut = d_15_constrainedResult_
                d_2_steps_ = (d_2_steps_) + (d_12_genBudget_)
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_17_closeBudget_: int
            d_17_closeBudget_ = (maxSteps) - (d_2_steps_)
            d_18_cg_: _dafny.Seq
            d_19_ci_: bool
            d_20_cc_: _dafny.Seq
            out12_: _dafny.Seq
            out13_: bool
            out14_: _dafny.Seq
            out12_, out13_, out14_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_17_closeBudget_)
            d_18_cg_ = out12_
            d_19_ci_ = out13_
            d_20_cc_ = out14_
            generated = d_18_cg_
            insideConstrainedOut = d_19_ci_
            currentConstrainedOut = d_20_cc_
            d_2_steps_ = maxSteps
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

