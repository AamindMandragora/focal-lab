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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate accurate SQL for Spider benchmark. Critical rules: (1) NEVER use table aliases or AS keyword. Write 'table.column' not 't.column', 'FROM table' not 'FROM table t'. (2) Map question keywords to SQL: 'average/avg' -> AVG(col), 'total/sum of' -> SUM(col), 'number of/count/how many' -> COUNT(*), 'maximum/highest/most' -> MAX(col) or ORDER BY col DESC LIMIT 1, 'minimum/lowest/least' -> MIN(col) or ORDER BY col ASC LIMIT 1. (3) For 'X that have more than one Y' or 'X that have at least N Y': use JOIN + GROUP BY + HAVING COUNT(*) > N. Example: 'singers with more than one song' -> SELECT singer.name FROM singer JOIN song ON singer.singer_id = song.singer_id GROUP BY singer.singer_id HAVING COUNT(*) > 1. (4) For 'X used for greatest number of Y' or 'most used X': use JOIN + GROUP BY + ORDER BY COUNT(*) DESC LIMIT 1. (5) For 'when both A and B': use simple INTERSECT. Example: 'semester when both Master and Bachelor enrolled' -> SELECT semester_id FROM table1 WHERE condition1 INTERSECT SELECT semester_id FROM table2 WHERE condition2. Use INTERSECT, not complex multi-join queries. (6) ALWAYS add GROUP BY when question uses 'for each' or 'per'. ALWAYS add HAVING when comparing counts with thresholds. (7) For ordering: 'ascending/alphabetical' -> ORDER BY col ASC; 'descending/reversed/largest first' -> ORDER BY col DESC. (8) SELECT only the specific columns the question asks for. (9) Use exact table and column names from the schema. (10) For multi-table queries: use JOIN...ON. For set operations: use INTERSECT or EXCEPT.")))
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
            d_3_fillBudget1_: int
            d_3_fillBudget1_ = _dafny.euclidian_division((d_2_rem_) * (4), 5)
            if (d_3_fillBudget1_) >= (1):
                d_4_stable_: _dafny.Seq
                d_4_stable_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                d_5_filled_: _dafny.Seq
                out3_: _dafny.Seq
                out3_ = (d_0_helpers_).RegenerateUnitOnGroundingFailure(lm, parser, (prompt) + (d_4_stable_), currentConstrainedOut, eosToken, d_3_fillBudget1_, 3, 5)
                d_5_filled_ = out3_
                generated = (d_4_stable_) + (d_5_filled_)
                currentConstrainedOut = d_5_filled_
                d_1_steps_ = (d_1_steps_) + (d_3_fillBudget1_)
        if (d_1_steps_) < (maxSteps):
            d_6_rem2_: int
            d_6_rem2_ = (maxSteps) - (d_1_steps_)
            d_7_fillBudget2_: int
            d_7_fillBudget2_ = _dafny.euclidian_division(d_6_rem2_, 2)
            d_8_loopSteps_: int
            d_8_loopSteps_ = 0
            with _dafny.label("2_0"):
                while (d_8_loopSteps_) < (d_7_fillBudget2_):
                    with _dafny.c_label("2_0"):
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            raise _dafny.Break("2_0")
                        d_9_stable2_: _dafny.Seq
                        d_9_stable2_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_10_constrainedPrompt2_: _dafny.Seq
                        d_10_constrainedPrompt2_ = (prompt) + (d_9_stable2_)
                        d_11_nextTemp_: _dafny.Seq
                        out4_: _dafny.Seq
                        out4_ = (d_0_helpers_).SafeTemperatureConstrainedStep(lm, parser, d_10_constrainedPrompt2_, currentConstrainedOut, _dafny.BigRational('8e-1'), eosToken)
                        d_11_nextTemp_ = out4_
                        d_8_loopSteps_ = (d_8_loopSteps_) + (1)
                        if (d_11_nextTemp_) == (eosToken):
                            raise _dafny.Break("2_0")
                        d_12_ng_: _dafny.Seq
                        d_13_ni_: bool
                        d_14_nc_: _dafny.Seq
                        out5_: _dafny.Seq
                        out6_: bool
                        out7_: _dafny.Seq
                        out5_, out6_, out7_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_11_nextTemp_)
                        d_12_ng_ = out5_
                        d_13_ni_ = out6_
                        d_14_nc_ = out7_
                        generated = d_12_ng_
                        insideConstrainedOut = d_13_ni_
                        currentConstrainedOut = d_14_nc_
                        pass
                pass
            d_1_steps_ = (d_1_steps_) + (d_7_fillBudget2_)
        if (d_1_steps_) < (maxSteps):
            d_15_closeBudget_: int
            d_15_closeBudget_ = (maxSteps) - (d_1_steps_)
            out8_: _dafny.Seq
            out9_: bool
            out10_: _dafny.Seq
            out8_, out9_, out10_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_15_closeBudget_)
            generated = out8_
            insideConstrainedOut = out9_
            currentConstrainedOut = out10_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

