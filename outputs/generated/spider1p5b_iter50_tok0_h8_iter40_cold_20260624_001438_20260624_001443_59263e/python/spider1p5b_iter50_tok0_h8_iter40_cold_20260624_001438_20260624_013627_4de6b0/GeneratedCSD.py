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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "You are generating a SQL query for Spider benchmark. CRITICAL RULES: (1) Write ONLY the SQL query, nothing else. (2) Use ONLY lowercase SQL: 'select', 'from', 'where', 'join', 'on', 'group by', 'having', 'order by', 'limit', 'count', 'max', 'min', 'avg', 'sum'. (3) NEVER use aliases: wrong='SELECT d.age FROM dogs d', right='select age from dogs'. (4) NEVER use 'AS' keyword. (5) Add spaces inside parentheses: write 'count ( * )' not 'count(*)'. (6) Use simple queries when possible: avoid unnecessary JOINs. (7) Example: 'What is the max age of dogs?' -> 'select max ( age ) from dogs'. (8) Example: 'How many employees?' -> 'select count ( * ) from employee'. (9) Use exact column/table names from the provided schema.")))
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
            d_3_fillBudget_ = _dafny.euclidian_division(d_2_rem_, 3)
            if (d_3_fillBudget_) >= (1):
                d_4_stableLen_: int
                d_4_stableLen_ = (len(generated)) - (len(currentConstrainedOut))
                d_5_stable_: _dafny.Seq
                d_5_stable_ = _dafny.SeqWithoutIsStrInference((generated)[:d_4_stableLen_:])
                d_6_flatTokens_: _dafny.Seq
                out3_: _dafny.Seq
                out3_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
                d_6_flatTokens_ = out3_
                d_7_allowedUnits_: _dafny.Seq
                d_7_allowedUnits_ = d_6_flatTokens_
                d_8_unitBudget_: int
                if (d_3_fillBudget_) < (20):
                    d_8_unitBudget_ = d_3_fillBudget_
                elif True:
                    d_8_unitBudget_ = 20
                d_9_filled_: _dafny.Seq
                out4_: _dafny.Seq
                out4_ = (d_0_helpers_).RegenerateUnitOnCheckFailure(lm, parser, (prompt) + (d_5_stable_), currentConstrainedOut, eosToken, d_8_unitBudget_, 3, d_3_fillBudget_, d_7_allowedUnits_)
                d_9_filled_ = out4_
                generated = (d_5_stable_) + (d_9_filled_)
                currentConstrainedOut = d_9_filled_
                d_1_steps_ = (d_1_steps_) + (d_3_fillBudget_)
        if (d_1_steps_) < (maxSteps):
            d_10_closeBudget_: int
            d_10_closeBudget_ = (maxSteps) - (d_1_steps_)
            out5_: _dafny.Seq
            out6_: bool
            out7_: _dafny.Seq
            out5_, out6_, out7_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_10_closeBudget_)
            generated = out5_
            insideConstrainedOut = out6_
            currentConstrainedOut = out7_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

