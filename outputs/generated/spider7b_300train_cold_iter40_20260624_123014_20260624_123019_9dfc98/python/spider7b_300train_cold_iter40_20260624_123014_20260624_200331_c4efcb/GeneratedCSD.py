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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate a single correct SQL query. STRICT RULES: (1) NEVER use AS keyword anywhere - not for tables, not for columns, not for anything. Just write 'SELECT column FROM table'. (2) When the question uses 'both X and Y', 'X and also Y', or similar 'and' patterns across rows: use INTERSECT. Write two SELECT statements joined by INTERSECT. Never use self-join for this. (3) SELECT only the exact column(s) the question asks for. Do not select extra columns. If question asks 'what is the name' then write 'SELECT name'. (4) When filtering involves multiple tables (e.g., students who have pets of type X), use JOIN...ON to connect the tables, then use WHERE for conditions. (5) SQL function mapping - use these EXACTLY: 'average'->AVG(col), 'total/sum'->SUM(col), 'number/count/how many'->COUNT(*), 'maximum/highest/most'->MAX(col), 'minimum/lowest/least'->MIN(col). (6) Use exact table and column names from the schema. Use WHERE for filtering. Use ORDER BY for sorting. Use GROUP BY when counting 'for each'.")))
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
                out3_ = (d_0_helpers_).RegenerateUnitOnGroundingFailure(lm, parser, (prompt) + (d_4_stable_), currentConstrainedOut, eosToken, d_3_fillBudget1_, 3, 10)
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

