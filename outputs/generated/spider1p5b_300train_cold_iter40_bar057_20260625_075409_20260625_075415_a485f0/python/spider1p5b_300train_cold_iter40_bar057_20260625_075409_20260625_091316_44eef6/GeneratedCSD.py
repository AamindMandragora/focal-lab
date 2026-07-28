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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate a single valid SQL query that answers the question using only the provided schema. Format: SQL: <query>. Use correct table and column names. For aggregations use COUNT(*), AVG(col), SUM(col), MAX(col), MIN(col). For filtering use WHERE. For multi-table queries use JOIN ON. For grouping use GROUP BY. For sorting use ORDER BY. Output only the SQL query.")))
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
            d_3_fillBudget_ = _dafny.euclidian_division((d_2_rem_) * (3), 5)
            if (d_3_fillBudget_) == (0):
                d_3_fillBudget_ = 1
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
        d_6_phase2Budget_: int
        d_6_phase2Budget_ = 0
        if (maxSteps) >= (5):
            d_6_phase2Budget_ = _dafny.euclidian_division(maxSteps, 5)
        d_7_phase2Steps_: int
        d_7_phase2Steps_ = 0
        with _dafny.label("0"):
            while ((((d_1_steps_) < (maxSteps)) and ((d_7_phase2Steps_) < (d_6_phase2Budget_))) and (insideConstrainedOut)) and (not((parser).IsCompletePrefix(currentConstrainedOut))):
                with _dafny.c_label("0"):
                    d_8_stable2_: _dafny.Seq
                    d_8_stable2_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                    d_9_constrainedPrompt_: _dafny.Seq
                    d_9_constrainedPrompt_ = (prompt) + (d_8_stable2_)
                    d_10_next_: _dafny.Seq
                    out4_: _dafny.Seq
                    out4_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_9_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                    d_10_next_ = out4_
                    d_1_steps_ = (d_1_steps_) + (1)
                    d_7_phase2Steps_ = (d_7_phase2Steps_) + (1)
                    if (d_10_next_) == (eosToken):
                        raise _dafny.Break("0")
                    elif True:
                        d_11_ag_: _dafny.Seq
                        d_12_ai_: bool
                        d_13_ac_: _dafny.Seq
                        out5_: _dafny.Seq
                        out6_: bool
                        out7_: _dafny.Seq
                        out5_, out6_, out7_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_10_next_)
                        d_11_ag_ = out5_
                        d_12_ai_ = out6_
                        d_13_ac_ = out7_
                        generated = d_11_ag_
                        insideConstrainedOut = d_12_ai_
                        currentConstrainedOut = d_13_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_14_closeBudget_: int
            d_14_closeBudget_ = (maxSteps) - (d_1_steps_)
            out8_: _dafny.Seq
            out9_: bool
            out10_: _dafny.Seq
            out8_, out9_, out10_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_14_closeBudget_)
            generated = out8_
            insideConstrainedOut = out9_
            currentConstrainedOut = out10_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

