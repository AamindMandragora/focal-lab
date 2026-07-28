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
        if not(insideConstrainedOut):
            out0_: _dafny.Seq
            out1_: bool
            out2_: _dafny.Seq
            out0_, out1_, out2_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
            generated = out0_
            insideConstrainedOut = out1_
            currentConstrainedOut = out2_
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Write the simplest possible SQL query. Use only tables and columns from the schema. Do NOT use aliases (AS). Do NOT use short table aliases like s1, s2, t1, t2, e1, e2. Do NOT use UNION. Use INTERSECT when finding rows satisfying two separate conditions. Do NOT use subqueries. Do NOT prefix column names with table names unless joining multiple tables. Output only the SQL query.")))
        if (d_1_steps_) < (maxSteps):
            d_2_rem_: int
            d_2_rem_ = (maxSteps) - (d_1_steps_)
            d_3_fillBudget_: int
            d_3_fillBudget_ = _dafny.euclidian_division((d_2_rem_) * (2), 3)
            if (d_3_fillBudget_) >= (1):
                d_4_stable_: _dafny.Seq
                d_4_stable_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                d_5_rollbackBudget_: int
                d_5_rollbackBudget_ = 15
                d_6_groundedCurrent_: _dafny.Seq
                out3_: _dafny.Seq
                out3_ = (d_0_helpers_).RegenerateUnitOnGroundingFailure(lm, parser, (prompt) + (d_4_stable_), currentConstrainedOut, eosToken, d_3_fillBudget_, 5, d_5_rollbackBudget_)
                d_6_groundedCurrent_ = out3_
                generated = (d_4_stable_) + (d_6_groundedCurrent_)
                currentConstrainedOut = d_6_groundedCurrent_
                d_1_steps_ = (d_1_steps_) + (d_3_fillBudget_)
        if (d_1_steps_) < (maxSteps):
            d_7_closeBudget_: int
            d_7_closeBudget_ = (maxSteps) - (d_1_steps_)
            out4_: _dafny.Seq
            out5_: bool
            out6_: _dafny.Seq
            out4_, out5_, out6_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_7_closeBudget_)
            generated = out4_
            insideConstrainedOut = out5_
            currentConstrainedOut = out6_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

