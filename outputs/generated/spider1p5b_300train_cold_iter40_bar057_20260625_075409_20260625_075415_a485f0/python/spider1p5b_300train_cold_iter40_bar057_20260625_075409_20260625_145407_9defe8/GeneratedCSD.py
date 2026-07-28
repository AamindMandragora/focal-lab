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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SPIDER SQL FORMAT RULES - follow exactly: lowercase only, spaces inside all parentheses, no table aliases ever. CORRECT: select max ( age ) from dogs | select count ( * ) from airlines | select avg ( salary ) from employee | select min ( population ) from city where country_code = 'CHN' | select tv_channel.country from tv_channel join cartoon on tv_channel.id = cartoon.channel where cartoon.written_by = 'Todd Casey' | select count ( * ) from documents join templates on documents.template_id = templates.template_id where templates.template_type_code = 'PPT' | select name from teacher where teacher_id not in ( select teacher_id from course_arrange ) | select city , count ( * ) from station group by city order by count ( * ) desc limit 1. WRONG (never do this): SELECT MAX(d.age) FROM dogs d | SELECT COUNT(DISTINCT x) FROM t | SELECT T1.col FROM t1 T1 JOIN t2 T2 ON T1.id = T2.id | SELECT col FROM t WHERE id IN (SELECT id FROM t WHERE ...) AND id IN (SELECT id FROM t WHERE ...)")))
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
            d_3_fillBudget_ = _dafny.euclidian_division(d_2_rem_, 2)
            if (d_3_fillBudget_) == (0):
                d_3_fillBudget_ = d_2_rem_
            if (d_3_fillBudget_) > (d_2_rem_):
                d_3_fillBudget_ = d_2_rem_
            d_4_stable_: _dafny.Seq
            d_4_stable_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
            d_5_filled_: _dafny.Seq
            out3_: _dafny.Seq
            out3_ = (d_0_helpers_).RegenerateUnitOnGroundingFailure(lm, parser, (prompt) + (d_4_stable_), currentConstrainedOut, eosToken, d_3_fillBudget_, 3, d_3_fillBudget_)
            d_5_filled_ = out3_
            generated = (d_4_stable_) + (d_5_filled_)
            currentConstrainedOut = d_5_filled_
            d_1_steps_ = (d_1_steps_) + (d_3_fillBudget_)
        if (d_1_steps_) < (maxSteps):
            d_6_closeBudget_: int
            d_6_closeBudget_ = (maxSteps) - (d_1_steps_)
            out4_: _dafny.Seq
            out5_: bool
            out6_: _dafny.Seq
            out4_, out5_, out6_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_6_closeBudget_)
            generated = out4_
            insideConstrainedOut = out5_
            currentConstrainedOut = out6_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

