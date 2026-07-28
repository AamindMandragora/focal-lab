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
        d_1_guidance_: _dafny.Seq
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate a single complete SQL query using only the exact table names and column names from the schema provided. Use proper JOINs. Do not repeat clauses."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        if not(insideConstrainedOut):
            out0_: _dafny.Seq
            out1_: bool
            out2_: _dafny.Seq
            out0_, out1_, out2_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
            generated = out0_
            insideConstrainedOut = out1_
            currentConstrainedOut = out2_
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_3_rem_: int
            d_3_rem_ = (maxSteps) - (d_2_steps_)
            d_4_closeReserve_: int
            if (d_3_rem_) >= (60):
                d_4_closeReserve_ = 30
            elif (d_3_rem_) >= (20):
                d_4_closeReserve_ = 10
            elif (d_3_rem_) >= (4):
                d_4_closeReserve_ = 2
            elif True:
                d_4_closeReserve_ = 0
            d_5_fillBudget_: int
            if (d_3_rem_) > (d_4_closeReserve_):
                d_5_fillBudget_ = (d_3_rem_) - (d_4_closeReserve_)
            elif True:
                d_5_fillBudget_ = 0
            if (d_5_fillBudget_) >= (1):
                d_6_stable_: _dafny.Seq
                d_6_stable_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                d_7_constrainedPrompt_: _dafny.Seq
                d_7_constrainedPrompt_ = (prompt) + (d_6_stable_)
                d_8_maxStepsPerUnit_: int
                if (d_5_fillBudget_) >= (20):
                    d_8_maxStepsPerUnit_ = 20
                elif True:
                    d_8_maxStepsPerUnit_ = d_5_fillBudget_
                d_9_maxRollbackBudget_: int
                if (d_5_fillBudget_) >= (10):
                    d_9_maxRollbackBudget_ = 10
                elif True:
                    d_9_maxRollbackBudget_ = d_5_fillBudget_
                d_10_filled_: _dafny.Seq
                out3_: _dafny.Seq
                out3_ = (d_0_helpers_).RegenerateUnitOnGroundingFailure(lm, parser, d_7_constrainedPrompt_, currentConstrainedOut, eosToken, d_8_maxStepsPerUnit_, 3, d_9_maxRollbackBudget_)
                d_10_filled_ = out3_
                generated = (d_6_stable_) + (d_10_filled_)
                currentConstrainedOut = d_10_filled_
                d_2_steps_ = (d_2_steps_) + (d_5_fillBudget_)
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_11_closeBudget_: int
            d_11_closeBudget_ = (maxSteps) - (d_2_steps_)
            d_12_cg_: _dafny.Seq
            d_13_ci_: bool
            d_14_cc_: _dafny.Seq
            out4_: _dafny.Seq
            out5_: bool
            out6_: _dafny.Seq
            out4_, out5_, out6_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_11_closeBudget_)
            d_12_cg_ = out4_
            d_13_ci_ = out5_
            d_14_cc_ = out6_
            generated = d_12_cg_
            insideConstrainedOut = d_13_ci_
            currentConstrainedOut = d_14_cc_
            d_2_steps_ = maxSteps
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

