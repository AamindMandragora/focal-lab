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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate exactly one SQL query answering the question. Critical rules: (1) For LIST or SHOW questions without conditions, use SELECT cols FROM table ORDER BY col - do NOT add WHERE. (2) For questions about entities satisfying TWO separate conditions from different records, use INTERSECT. (3) For questions with minimum count conditions on groups, use GROUP BY col HAVING COUNT(*) >= N. (4) Use NOT IN (SELECT col FROM table WHERE condition) for exclusion. (5) Write the simplest correct SQL - avoid adding JOINs, WHERE clauses, or subqueries unless the question explicitly requires them. Output SQL directly.")))
        d_2_freeLimit_: int
        d_2_freeLimit_ = 0
        if (maxSteps) >= (4):
            d_2_freeLimit_ = _dafny.euclidian_division((maxSteps) * (3), 4)
            if (d_2_freeLimit_) > ((maxSteps) - (2)):
                d_2_freeLimit_ = (maxSteps) - (2)
        with _dafny.label("0"):
            while ((d_1_steps_) < (d_2_freeLimit_)) and (not(insideConstrainedOut)):
                with _dafny.c_label("0"):
                    d_3_next_: _dafny.Seq
                    out0_: _dafny.Seq
                    out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                    d_3_next_ = out0_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if (d_3_next_) == (eosToken):
                        raise _dafny.Break("0")
                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_3_next_]))
                    d_4_glen_: int
                    d_4_glen_ = len(generated)
                    if (d_4_glen_) >= ((len(generatedPrefix)) + (4)):
                        if (_dafny.SeqWithoutIsStrInference((generated)[(d_4_glen_) - (2):d_4_glen_:])) == (_dafny.SeqWithoutIsStrInference((generated)[(d_4_glen_) - (4):(d_4_glen_) - (2):])):
                            raise _dafny.Break("0")
                    if (d_4_glen_) >= ((len(generatedPrefix)) + (6)):
                        if (_dafny.SeqWithoutIsStrInference((generated)[(d_4_glen_) - (3):d_4_glen_:])) == (_dafny.SeqWithoutIsStrInference((generated)[(d_4_glen_) - (6):(d_4_glen_) - (3):])):
                            raise _dafny.Break("0")
                    if (d_4_glen_) >= ((len(generatedPrefix)) + (8)):
                        if (_dafny.SeqWithoutIsStrInference((generated)[(d_4_glen_) - (4):d_4_glen_:])) == (_dafny.SeqWithoutIsStrInference((generated)[(d_4_glen_) - (8):(d_4_glen_) - (4):])):
                            raise _dafny.Break("0")
                    pass
            pass
        if not(insideConstrainedOut):
            out1_: _dafny.Seq
            out2_: bool
            out3_: _dafny.Seq
            out1_, out2_, out3_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
            generated = out1_
            insideConstrainedOut = out2_
            currentConstrainedOut = out3_
        if (d_1_steps_) < (maxSteps):
            d_5_closeBudget_: int
            d_5_closeBudget_ = (maxSteps) - (d_1_steps_)
            out4_: _dafny.Seq
            out5_: bool
            out6_: _dafny.Seq
            out4_, out5_, out6_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_5_closeBudget_)
            generated = out4_
            insideConstrainedOut = out5_
            currentConstrainedOut = out6_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

