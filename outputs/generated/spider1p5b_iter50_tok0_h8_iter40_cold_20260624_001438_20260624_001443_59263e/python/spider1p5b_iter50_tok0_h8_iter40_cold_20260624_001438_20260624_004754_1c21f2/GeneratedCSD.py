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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate a single valid SQL query matching the question. Use exact table and column names from the schema. Do NOT use AS aliases for columns or tables. Do NOT use short table aliases (like c, m, s1). Write full table names in JOIN conditions. Use INTERSECT not UNION for 'both' queries. Use simple direct queries.")))
        if not(insideConstrainedOut):
            out0_: _dafny.Seq
            out1_: bool
            out2_: _dafny.Seq
            out0_, out1_, out2_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
            generated = out0_
            insideConstrainedOut = out1_
            currentConstrainedOut = out2_
        if (d_1_steps_) < (maxSteps):
            d_2_rem1_: int
            d_2_rem1_ = (maxSteps) - (d_1_steps_)
            d_3_fillBudget1_: int
            d_3_fillBudget1_ = _dafny.euclidian_division(d_2_rem1_, 2)
            if (d_3_fillBudget1_) >= (1):
                d_4_stableLen1_: int
                d_4_stableLen1_ = (len(generated)) - (len(currentConstrainedOut))
                d_5_stable1_: _dafny.Seq
                d_5_stable1_ = _dafny.SeqWithoutIsStrInference((generated)[:d_4_stableLen1_:])
                d_6_unitBudget1_: int
                if (d_3_fillBudget1_) < (20):
                    d_6_unitBudget1_ = d_3_fillBudget1_
                elif True:
                    d_6_unitBudget1_ = 20
                d_7_filled1_: _dafny.Seq
                out3_: _dafny.Seq
                out3_ = (d_0_helpers_).RegenerateUnitOnGroundingFailure(lm, parser, (prompt) + (d_5_stable1_), currentConstrainedOut, eosToken, d_6_unitBudget1_, 5, d_3_fillBudget1_)
                d_7_filled1_ = out3_
                generated = (d_5_stable1_) + (d_7_filled1_)
                currentConstrainedOut = d_7_filled1_
                d_1_steps_ = (d_1_steps_) + (d_3_fillBudget1_)
        if (d_1_steps_) < (maxSteps):
            d_8_rem2_: int
            d_8_rem2_ = (maxSteps) - (d_1_steps_)
            d_9_fillBudget2_: int
            d_9_fillBudget2_ = _dafny.euclidian_division(d_8_rem2_, 2)
            if (d_9_fillBudget2_) >= (1):
                d_10_stableLen2_: int
                d_10_stableLen2_ = (len(generated)) - (len(currentConstrainedOut))
                d_11_stable2_: _dafny.Seq
                d_11_stable2_ = _dafny.SeqWithoutIsStrInference((generated)[:d_10_stableLen2_:])
                d_12_unitBudget2_: int
                if (d_9_fillBudget2_) < (20):
                    d_12_unitBudget2_ = d_9_fillBudget2_
                elif True:
                    d_12_unitBudget2_ = 20
                d_13_filled2_: _dafny.Seq
                out4_: _dafny.Seq
                out4_ = (d_0_helpers_).RegenerateUnitOnGroundingFailure(lm, parser, (prompt) + (d_11_stable2_), currentConstrainedOut, eosToken, d_12_unitBudget2_, 3, d_9_fillBudget2_)
                d_13_filled2_ = out4_
                generated = (d_11_stable2_) + (d_13_filled2_)
                currentConstrainedOut = d_13_filled2_
                d_1_steps_ = (d_1_steps_) + (d_9_fillBudget2_)
        if (d_1_steps_) < (maxSteps):
            d_14_closeBudget_: int
            d_14_closeBudget_ = (maxSteps) - (d_1_steps_)
            out5_: _dafny.Seq
            out6_: bool
            out7_: _dafny.Seq
            out5_, out6_, out7_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_14_closeBudget_)
            generated = out5_
            insideConstrainedOut = out6_
            currentConstrainedOut = out7_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

