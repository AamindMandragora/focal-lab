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
        (d_0_helpers_).AppendTaskGuidance(lm, (((((_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Write a single SQL SELECT statement answering the question. Important rules: "))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "1. Use JOIN...ON to combine multiple tables (do NOT use nested IN subqueries for joins). ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "2. Use HAVING clause for filtering after GROUP BY (not WHERE with aggregate functions). ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "3. Use ORDER BY col ASC/DESC for sorting results. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "4. Avoid deeply nested subqueries. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "5. Write the complete query in one statement."))))
        d_2_freeLimit_: int
        d_2_freeLimit_ = _dafny.euclidian_division((maxSteps) * (3), 4)
        if ((d_2_freeLimit_) < (2)) and ((maxSteps) >= (2)):
            d_2_freeLimit_ = 2
        if ((d_2_freeLimit_) > ((maxSteps) - (2))) and ((maxSteps) >= (2)):
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
                    if (len(generated)) >= ((len(generatedPrefix)) + (8)):
                        d_4_glen_: int
                        d_4_glen_ = len(generated)
                        if (_dafny.SeqWithoutIsStrInference((generated)[(d_4_glen_) - (4):d_4_glen_:])) == (_dafny.SeqWithoutIsStrInference((generated)[(d_4_glen_) - (8):(d_4_glen_) - (4):])):
                            raise _dafny.Break("0")
                    d_5_parenCount_: int
                    out1_: int
                    out1_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "(")))
                    d_5_parenCount_ = out1_
                    if (d_5_parenCount_) > (12):
                        raise _dafny.Break("0")
                    d_6_selectCount_: int
                    out2_: int
                    out2_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SELECT")))
                    d_6_selectCount_ = out2_
                    if (d_6_selectCount_) < (1):
                        out3_: int
                        out3_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "select")))
                        d_6_selectCount_ = out3_
                    if (d_6_selectCount_) > (4):
                        raise _dafny.Break("0")
                    pass
            pass
        if not(insideConstrainedOut):
            out4_: _dafny.Seq
            out5_: bool
            out6_: _dafny.Seq
            out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
            generated = out4_
            insideConstrainedOut = out5_
            currentConstrainedOut = out6_
        if (d_1_steps_) < (maxSteps):
            d_7_rem_: int
            d_7_rem_ = (maxSteps) - (d_1_steps_)
            d_8_fillBudget_: int
            d_8_fillBudget_ = _dafny.euclidian_division((d_7_rem_) * (3), 4)
            if (d_8_fillBudget_) < (1):
                d_8_fillBudget_ = 1
            if ((d_7_rem_) >= (2)) and ((d_8_fillBudget_) > ((d_7_rem_) - (1))):
                d_8_fillBudget_ = (d_7_rem_) - (1)
            if (d_8_fillBudget_) > ((maxSteps) - (d_1_steps_)):
                d_8_fillBudget_ = (maxSteps) - (d_1_steps_)
            d_9_stable_: _dafny.Seq
            d_9_stable_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
            d_10_filled_: _dafny.Seq
            out7_: _dafny.Seq
            out7_ = (d_0_helpers_).RegenerateUnitOnGroundingFailure(lm, parser, (prompt) + (d_9_stable_), currentConstrainedOut, eosToken, d_8_fillBudget_, 3, d_8_fillBudget_)
            d_10_filled_ = out7_
            generated = (d_9_stable_) + (d_10_filled_)
            currentConstrainedOut = d_10_filled_
            d_1_steps_ = (d_1_steps_) + (d_8_fillBudget_)
        if (d_1_steps_) < (maxSteps):
            d_11_closeBudget_: int
            d_11_closeBudget_ = (maxSteps) - (d_1_steps_)
            out8_: _dafny.Seq
            out9_: bool
            out10_: _dafny.Seq
            out8_, out9_, out10_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_11_closeBudget_)
            generated = out8_
            insideConstrainedOut = out9_
            currentConstrainedOut = out10_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

