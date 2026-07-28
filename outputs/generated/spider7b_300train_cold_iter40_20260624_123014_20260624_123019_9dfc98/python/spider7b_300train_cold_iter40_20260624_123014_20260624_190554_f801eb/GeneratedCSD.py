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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate accurate SQL for Spider benchmark. Critical rules: (1) NEVER use table aliases or AS keyword. Write 'table.column' not 't.column', 'FROM table' not 'FROM table t'. (2) Map question keywords: 'average/avg' -> AVG(col), 'total/sum' -> SUM(col), 'how many/count/number of' -> COUNT(*), 'maximum/highest/most' -> MAX(col) or ORDER BY col DESC LIMIT 1, 'minimum/lowest/fewest/least' -> MIN(col) or ORDER BY col ASC LIMIT 1. (3) 'highest rank' means best performance = lowest rank NUMBER = MIN(rank_col). 'Rank 1' is the best. (4) For 'both X and Y' or 'visited both ... and ...': use INTERSECT between two SELECT statements. Do NOT use complex multi-table joins when INTERSECT is simpler. (5) For 'greatest/largest percentage for each group': use GROUP BY + MAX(percentage). (6) For 'for each country' or 'per country': GROUP BY country_col. (7) SELECT only the specific columns the question asks for. Do NOT select all columns. (8) Prefer JOIN over subqueries. Use JOIN...ON for multi-table queries. (9) For ordering: 'ascending/alphabetical' -> ORDER BY col ASC; 'descending/reversed' -> ORDER BY col DESC. (10) Use exact table and column names from the schema provided.")))
        if not(insideConstrainedOut):
            out0_: _dafny.Seq
            out1_: bool
            out2_: _dafny.Seq
            out0_, out1_, out2_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
            generated = out0_
            insideConstrainedOut = out1_
            currentConstrainedOut = out2_
        d_2_allowedUnits_: _dafny.Seq
        out3_: _dafny.Seq
        out3_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
        d_2_allowedUnits_ = out3_
        if (d_1_steps_) < (maxSteps):
            d_3_rem_: int
            d_3_rem_ = (maxSteps) - (d_1_steps_)
            d_4_fillBudget2_: int
            d_4_fillBudget2_ = _dafny.euclidian_division((d_3_rem_) * (3), 5)
            if (d_4_fillBudget2_) >= (1):
                d_5_stable_: _dafny.Seq
                d_5_stable_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                d_6_filled_: _dafny.Seq
                out4_: _dafny.Seq
                out4_ = (d_0_helpers_).RegenerateUnitOnCheckFailure(lm, parser, (prompt) + (d_5_stable_), currentConstrainedOut, eosToken, d_4_fillBudget2_, 4, 8, d_2_allowedUnits_)
                d_6_filled_ = out4_
                generated = (d_5_stable_) + (d_6_filled_)
                currentConstrainedOut = d_6_filled_
                d_1_steps_ = (d_1_steps_) + (d_4_fillBudget2_)
        if (d_1_steps_) < (maxSteps):
            d_7_rem2_: int
            d_7_rem2_ = (maxSteps) - (d_1_steps_)
            d_8_fillBudget3_: int
            d_8_fillBudget3_ = _dafny.euclidian_division(d_7_rem2_, 2)
            d_9_loopSteps_: int
            d_9_loopSteps_ = 0
            with _dafny.label("2_0"):
                while (d_9_loopSteps_) < (d_8_fillBudget3_):
                    with _dafny.c_label("2_0"):
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            raise _dafny.Break("2_0")
                        d_10_stable2_: _dafny.Seq
                        d_10_stable2_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_11_constrainedPrompt2_: _dafny.Seq
                        d_11_constrainedPrompt2_ = (prompt) + (d_10_stable2_)
                        d_12_nextTemp_: _dafny.Seq
                        out5_: _dafny.Seq
                        out5_ = (d_0_helpers_).SafeTemperatureConstrainedStep(lm, parser, d_11_constrainedPrompt2_, currentConstrainedOut, _dafny.BigRational('8e-1'), eosToken)
                        d_12_nextTemp_ = out5_
                        d_9_loopSteps_ = (d_9_loopSteps_) + (1)
                        if (d_12_nextTemp_) == (eosToken):
                            raise _dafny.Break("2_0")
                        d_13_ng_: _dafny.Seq
                        d_14_ni_: bool
                        d_15_nc_: _dafny.Seq
                        out6_: _dafny.Seq
                        out7_: bool
                        out8_: _dafny.Seq
                        out6_, out7_, out8_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_12_nextTemp_)
                        d_13_ng_ = out6_
                        d_14_ni_ = out7_
                        d_15_nc_ = out8_
                        generated = d_13_ng_
                        insideConstrainedOut = d_14_ni_
                        currentConstrainedOut = d_15_nc_
                        pass
                pass
            d_1_steps_ = (d_1_steps_) + (d_8_fillBudget3_)
        if (d_1_steps_) < (maxSteps):
            d_16_closeBudget_: int
            d_16_closeBudget_ = (maxSteps) - (d_1_steps_)
            out9_: _dafny.Seq
            out10_: bool
            out11_: _dafny.Seq
            out9_, out10_, out11_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_16_closeBudget_)
            generated = out9_
            insideConstrainedOut = out10_
            currentConstrainedOut = out11_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

