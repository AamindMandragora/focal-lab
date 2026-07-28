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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Write a complete and accurate SQL query. Include all JOIN conditions to connect related tables. Include all WHERE filters. Use subqueries for aggregate comparisons. Do not use table aliases.")))
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
            d_3_fillBudget1_ = _dafny.euclidian_division(d_2_rem_, 2)
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
            d_7_fillBudget2_ = _dafny.euclidian_division((d_6_rem2_) * (3), 5)
            if (d_7_fillBudget2_) >= (1):
                d_8_stable2_: _dafny.Seq
                d_8_stable2_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                d_9_constrainedPrompt2_: _dafny.Seq
                d_9_constrainedPrompt2_ = (prompt) + (d_8_stable2_)
                d_10_rolloutGen_: _dafny.Seq
                d_11_rolloutSteps_: int
                d_12_rolloutEos_: bool
                out4_: _dafny.Seq
                out5_: int
                out6_: bool
                out4_, out5_, out6_ = (d_0_helpers_).RolloutConstrainedWithPenalties(lm, parser, d_9_constrainedPrompt2_, currentConstrainedOut, d_7_fillBudget2_, generated, _dafny.BigRational('2e0'), eosToken)
                d_10_rolloutGen_ = out4_
                d_11_rolloutSteps_ = out5_
                d_12_rolloutEos_ = out6_
                generated = (d_8_stable2_) + (d_10_rolloutGen_)
                currentConstrainedOut = d_10_rolloutGen_
                d_1_steps_ = (d_1_steps_) + (d_7_fillBudget2_)
                if d_12_rolloutEos_:
                    cost = d_1_steps_
                    return generated, insideConstrainedOut, currentConstrainedOut, cost
        if (d_1_steps_) < (maxSteps):
            d_13_closeBudget_: int
            d_13_closeBudget_ = (maxSteps) - (d_1_steps_)
            out7_: _dafny.Seq
            out8_: bool
            out9_: _dafny.Seq
            out7_, out8_, out9_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_13_closeBudget_)
            generated = out7_
            insideConstrainedOut = out8_
            currentConstrainedOut = out9_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

