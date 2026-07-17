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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "You must output exactly: SQL: <<your SQL query here>> where the SQL query is a single valid SQL statement using SELECT. Use the schema tables and columns provided. No explanation, no markdown, just SQL: <<query>>."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_prefixBudget_: int
        if (maxSteps) >= (20):
            d_3_prefixBudget_ = 20
        elif True:
            d_3_prefixBudget_ = maxSteps
        if (not(insideConstrainedOut)) and ((d_2_steps_) < (maxSteps)):
            d_4_chunkBudget_: int
            if (d_3_prefixBudget_) <= ((maxSteps) - (d_2_steps_)):
                d_4_chunkBudget_ = d_3_prefixBudget_
            elif True:
                d_4_chunkBudget_ = (maxSteps) - (d_2_steps_)
            if (d_4_chunkBudget_) >= (1):
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
                d_2_steps_ = (d_2_steps_) + (d_8_stepsUsed_)
                generated = d_5_cg_
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
            d_9_rem_: int
            d_9_rem_ = (maxSteps) - (d_2_steps_)
            d_10_closeReserve_: int
            if (d_9_rem_) >= (30):
                d_10_closeReserve_ = 30
            elif (d_9_rem_) >= (10):
                d_10_closeReserve_ = 10
            elif True:
                d_10_closeReserve_ = d_9_rem_
            d_11_fillBudget_: int
            if (d_9_rem_) > (d_10_closeReserve_):
                d_11_fillBudget_ = (d_9_rem_) - (d_10_closeReserve_)
            elif True:
                d_11_fillBudget_ = 0
            if (d_11_fillBudget_) >= (1):
                d_12_stable_: _dafny.Seq
                d_12_stable_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                d_13_constrainedPrompt_: _dafny.Seq
                d_13_constrainedPrompt_ = (prompt) + (d_12_stable_)
                d_14_filled_: _dafny.Seq
                out10_: _dafny.Seq
                out10_ = (d_0_helpers_).RegenerateUnitOnGroundingFailure(lm, parser, d_13_constrainedPrompt_, currentConstrainedOut, eosToken, d_11_fillBudget_, 3, (10 if (d_11_fillBudget_) >= (10) else d_11_fillBudget_))
                d_14_filled_ = out10_
                generated = (d_12_stable_) + (d_14_filled_)
                currentConstrainedOut = d_14_filled_
                d_2_steps_ = (d_2_steps_) + (d_11_fillBudget_)
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_15_closeBudget_: int
            d_15_closeBudget_ = (maxSteps) - (d_2_steps_)
            d_16_cg_: _dafny.Seq
            d_17_ci_: bool
            d_18_cc_: _dafny.Seq
            out11_: _dafny.Seq
            out12_: bool
            out13_: _dafny.Seq
            out11_, out12_, out13_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_15_closeBudget_)
            d_16_cg_ = out11_
            d_17_ci_ = out12_
            d_18_cc_ = out13_
            generated = d_16_cg_
            insideConstrainedOut = d_17_ci_
            currentConstrainedOut = d_18_cc_
            d_2_steps_ = maxSteps
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

