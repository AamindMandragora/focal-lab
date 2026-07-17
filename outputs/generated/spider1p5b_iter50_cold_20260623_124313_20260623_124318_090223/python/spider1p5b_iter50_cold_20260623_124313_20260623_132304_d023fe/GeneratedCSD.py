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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate a single valid SQL query answering the question using the schema. Use correct table and column names from the schema. Output only the SQL query.")))
        d_1_steps_: int
        d_1_steps_ = 0
        if (not(insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
            d_2_og_: _dafny.Seq
            d_3_oi_: bool
            d_4_oc_: _dafny.Seq
            out0_: _dafny.Seq
            out1_: bool
            out2_: _dafny.Seq
            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_2_og_ = out0_
            d_3_oi_ = out1_
            d_4_oc_ = out2_
            generated = d_2_og_
            insideConstrainedOut = d_3_oi_
            currentConstrainedOut = d_4_oc_
            d_1_steps_ = (d_1_steps_) + (1)
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_5_groundBudget_: int
            d_5_groundBudget_ = (maxSteps) - (d_1_steps_)
            if (d_5_groundBudget_) > (1):
                d_6_maxSymSteps_: int
                d_6_maxSymSteps_ = (d_5_groundBudget_) - (1)
                d_7_maxStepsPerUnit_: int
                d_7_maxStepsPerUnit_ = 20
                d_8_maxRetries_: int
                d_8_maxRetries_ = 3
                d_9_maxRollbackBudget_: int
                d_9_maxRollbackBudget_ = 10
                d_10_unitBudget_: int
                d_10_unitBudget_ = ((d_8_maxRetries_) + (1)) * (d_7_maxStepsPerUnit_)
                if (d_10_unitBudget_) > (d_6_maxSymSteps_):
                    d_10_unitBudget_ = d_6_maxSymSteps_
                    d_7_maxStepsPerUnit_ = _dafny.euclidian_division(d_10_unitBudget_, (d_8_maxRetries_) + (1))
                    if (d_7_maxStepsPerUnit_) < (1):
                        d_7_maxStepsPerUnit_ = 1
                d_11_constrainedPrompt_: _dafny.Seq
                d_11_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                d_12_resultConstrained_: _dafny.Seq
                out3_: _dafny.Seq
                out3_ = (d_0_helpers_).RegenerateUnitOnGroundingFailure(lm, parser, d_11_constrainedPrompt_, currentConstrainedOut, eosToken, d_7_maxStepsPerUnit_, d_8_maxRetries_, d_9_maxRollbackBudget_)
                d_12_resultConstrained_ = out3_
                d_13_stablePrefix_: _dafny.Seq
                d_13_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                generated = (d_13_stablePrefix_) + (d_12_resultConstrained_)
                currentConstrainedOut = d_12_resultConstrained_
                d_14_stepsUsed_: int
                d_14_stepsUsed_ = ((d_8_maxRetries_) + (1)) * (d_7_maxStepsPerUnit_)
                if (d_14_stepsUsed_) > (d_6_maxSymSteps_):
                    d_14_stepsUsed_ = d_6_maxSymSteps_
                d_1_steps_ = (d_1_steps_) + (d_14_stepsUsed_)
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_15_closeBudget_: int
            d_15_closeBudget_ = (maxSteps) - (d_1_steps_)
            d_16_cg_: _dafny.Seq
            d_17_ci_: bool
            d_18_cc_: _dafny.Seq
            out4_: _dafny.Seq
            out5_: bool
            out6_: _dafny.Seq
            out4_, out5_, out6_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_15_closeBudget_)
            d_16_cg_ = out4_
            d_17_ci_ = out5_
            d_18_cc_ = out6_
            generated = d_16_cg_
            insideConstrainedOut = d_17_ci_
            currentConstrainedOut = d_18_cc_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

