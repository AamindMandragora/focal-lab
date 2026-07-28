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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Output format: SQL: <<query>>. Write the simplest correct SQL query. Do NOT use table aliases. Do NOT use AS aliases for columns. Use exact column names from the schema. For max/min use ORDER BY col DESC LIMIT 1. Use direct WHERE clauses, not subqueries when possible.")))
        d_1_steps_: int
        d_1_steps_ = 0
        if (not(insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
            d_2_chunkBudget_: int
            d_2_chunkBudget_ = 8
            if (d_2_chunkBudget_) > ((maxSteps) - (d_1_steps_)):
                d_2_chunkBudget_ = (maxSteps) - (d_1_steps_)
            if (d_2_chunkBudget_) >= (1):
                d_3_cg_: _dafny.Seq
                d_4_stoppedOnOpen_: bool
                d_5_stoppedOnEos_: bool
                d_6_stepsUsed_: int
                out0_: _dafny.Seq
                out1_: bool
                out2_: bool
                out3_: int
                out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_2_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                d_3_cg_ = out0_
                d_4_stoppedOnOpen_ = out1_
                d_5_stoppedOnEos_ = out2_
                d_6_stepsUsed_ = out3_
                generated = d_3_cg_
                d_1_steps_ = (d_1_steps_) + (d_6_stepsUsed_)
                if d_5_stoppedOnEos_:
                    cost = d_1_steps_
                    return generated, insideConstrainedOut, currentConstrainedOut, cost
                if d_4_stoppedOnOpen_:
                    out4_: _dafny.Seq
                    out5_: bool
                    out6_: _dafny.Seq
                    out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                    generated = out4_
                    insideConstrainedOut = out5_
                    currentConstrainedOut = out6_
        if (not(insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
            d_7_og_: _dafny.Seq
            d_8_oi_: bool
            d_9_oc_: _dafny.Seq
            out7_: _dafny.Seq
            out8_: bool
            out9_: _dafny.Seq
            out7_, out8_, out9_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_7_og_ = out7_
            d_8_oi_ = out8_
            d_9_oc_ = out9_
            generated = d_7_og_
            insideConstrainedOut = d_8_oi_
            currentConstrainedOut = d_9_oc_
            d_1_steps_ = (d_1_steps_) + (1)
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_10_rem_: int
            d_10_rem_ = (maxSteps) - (d_1_steps_)
            d_11_closeReserve_: int
            d_11_closeReserve_ = _dafny.euclidian_division(d_10_rem_, 4)
            if ((d_11_closeReserve_) < (20)) and ((d_10_rem_) >= (25)):
                d_11_closeReserve_ = 20
            if (d_11_closeReserve_) > (50):
                d_11_closeReserve_ = 50
            if (d_11_closeReserve_) >= (d_10_rem_):
                if (d_10_rem_) > (1):
                    d_11_closeReserve_ = (d_10_rem_) - (1)
                elif True:
                    d_11_closeReserve_ = 0
            d_12_fillBudget_: int
            d_12_fillBudget_ = (d_10_rem_) - (d_11_closeReserve_)
            if (d_12_fillBudget_) >= (1):
                d_13_stable_: _dafny.Seq
                d_13_stable_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                d_14_filled_: _dafny.Seq
                out10_: _dafny.Seq
                out10_ = (d_0_helpers_).RegenerateUnitOnGroundingFailure(lm, parser, (prompt) + (d_13_stable_), currentConstrainedOut, eosToken, d_12_fillBudget_, 4, d_12_fillBudget_)
                d_14_filled_ = out10_
                generated = (d_13_stable_) + (d_14_filled_)
                currentConstrainedOut = d_14_filled_
                d_1_steps_ = (d_1_steps_) + (d_12_fillBudget_)
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_15_closeBudget_: int
            d_15_closeBudget_ = (maxSteps) - (d_1_steps_)
            out11_: _dafny.Seq
            out12_: bool
            out13_: _dafny.Seq
            out11_, out12_, out13_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_15_closeBudget_)
            generated = out11_
            insideConstrainedOut = out12_
            currentConstrainedOut = out13_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

