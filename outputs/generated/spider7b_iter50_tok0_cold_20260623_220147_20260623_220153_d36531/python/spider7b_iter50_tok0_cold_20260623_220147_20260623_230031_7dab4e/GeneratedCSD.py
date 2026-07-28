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
        d_1_steps_: int
        d_1_steps_ = 0
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate a complete SQL query. Use lowercase SQL keywords (select, from, join, where, group by, order by, having). Always include FROM clause with full table names. Use table.column format for column references. Use COUNT(*) not COUNT(column). Use JOIN with ON condition. Write the full query without truncation.")))
        if not(insideConstrainedOut):
            out0_: _dafny.Seq
            out1_: bool
            out2_: _dafny.Seq
            out0_, out1_, out2_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
            generated = out0_
            insideConstrainedOut = out1_
            currentConstrainedOut = out2_
        if (d_1_steps_) < (maxSteps):
            d_2_rem_: int
            d_2_rem_ = (maxSteps) - (d_1_steps_)
            d_3_fillBudget_: int
            d_3_fillBudget_ = _dafny.euclidian_division((d_2_rem_) * (4), 5)
            if (d_3_fillBudget_) >= (1):
                d_4_stable_: _dafny.Seq
                d_4_stable_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                d_5_penalties_: _dafny.Seq
                d_5_penalties_ = _dafny.SeqWithoutIsStrInference([])
                d_6_rolloutGen_: _dafny.Seq
                d_7_rolloutSteps_: int
                d_8_rolloutEos_: bool
                out3_: _dafny.Seq
                out4_: int
                out5_: bool
                out3_, out4_, out5_ = (d_0_helpers_).RolloutConstrainedWithPenalties(lm, parser, (prompt) + (d_4_stable_), currentConstrainedOut, d_3_fillBudget_, d_5_penalties_, _dafny.BigRational('0e0'), eosToken)
                d_6_rolloutGen_ = out3_
                d_7_rolloutSteps_ = out4_
                d_8_rolloutEos_ = out5_
                generated = (d_4_stable_) + (d_6_rolloutGen_)
                currentConstrainedOut = d_6_rolloutGen_
                d_1_steps_ = (d_1_steps_) + (d_3_fillBudget_)
        if (d_1_steps_) < (maxSteps):
            d_9_closeBudget_: int
            d_9_closeBudget_ = (maxSteps) - (d_1_steps_)
            out6_: _dafny.Seq
            out7_: bool
            out8_: _dafny.Seq
            out6_, out7_, out8_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_9_closeBudget_)
            generated = out6_
            insideConstrainedOut = out7_
            currentConstrainedOut = out8_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

