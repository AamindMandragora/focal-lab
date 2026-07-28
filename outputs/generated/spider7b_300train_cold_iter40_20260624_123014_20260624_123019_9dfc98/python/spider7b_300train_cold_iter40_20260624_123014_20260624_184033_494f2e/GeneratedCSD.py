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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate correct SQL for the Spider benchmark. CRITICAL RULES (follow exactly): (1) ABSOLUTELY NEVER use table aliases. NEVER write 'FROM table t' or 't.column'. ALWAYS write the full table name: 'FROM visitor' and 'visitor.name', never 'FROM visitor v' or 'v.name'. (2) NEVER use the AS keyword for column aliases. Write COUNT(*) not 'COUNT(*) AS cnt'. (3) Aggregation: 'average' -> AVG(col), 'total/sum' -> SUM(col), 'how many/count/number of' -> COUNT(*), 'highest/maximum/most' -> MAX(col) or ORDER BY col DESC LIMIT 1, 'lowest/minimum/least' -> MIN(col) or ORDER BY col ASC LIMIT 1. (4) For 'X for each Y': use GROUP BY Y with aggregate on X. For 'greatest/largest X for each Y': SELECT Y, MAX(X) FROM table GROUP BY Y. (5) For 'visited both A and B': use INTERSECT between two SELECT statements. (6) SELECT ONLY the exact columns the question asks for. Never select all columns or all column combinations. (7) For multi-table queries: use explicit JOIN...ON syntax. (8) Use exact table and column names from the schema. No invented names. (9) For 'in order': add ORDER BY. For 'reversed/descending': ORDER BY col DESC. (10) Always write complete valid SQL.")))
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
                        out4_ = (d_0_helpers_).SafeTemperatureConstrainedStep(lm, parser, d_10_constrainedPrompt2_, currentConstrainedOut, _dafny.BigRational('6e-1'), eosToken)
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

