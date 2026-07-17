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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate a single valid SQL query answering the question using only the provided schema. Output format: SQL: YOUR QUERY. Use correct SQL syntax with proper JOIN, WHERE, GROUP BY, ORDER BY clauses as needed."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        with _dafny.label("0"):
            while ((d_2_steps_) < (maxSteps)) and (not(insideConstrainedOut)):
                with _dafny.c_label("0"):
                    d_3_next_: _dafny.Seq
                    out0_: _dafny.Seq
                    out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                    d_3_next_ = out0_
                    d_2_steps_ = (d_2_steps_) + (1)
                    if (d_3_next_) == (eosToken):
                        raise _dafny.Break("0")
                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_3_next_]))
                    if (d_3_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                        insideConstrainedOut = True
                        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    pass
            pass
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_4_remainingBudget_: int
            d_4_remainingBudget_ = (maxSteps) - (d_2_steps_)
            d_5_closeReserve_: int
            if (d_4_remainingBudget_) > (2):
                d_5_closeReserve_ = 2
            elif True:
                d_5_closeReserve_ = d_4_remainingBudget_
            d_6_genBudget_: int
            d_6_genBudget_ = (d_4_remainingBudget_) - (d_5_closeReserve_)
            if (d_6_genBudget_) > (0):
                d_7_maxStepsPerUnit_: int
                if (d_6_genBudget_) > (20):
                    d_7_maxStepsPerUnit_ = 20
                elif True:
                    d_7_maxStepsPerUnit_ = d_6_genBudget_
                d_8_maxRetries_: int
                d_8_maxRetries_ = 3
                d_9_maxRollbackBudget_: int
                if (d_6_genBudget_) > (10):
                    d_9_maxRollbackBudget_ = 10
                elif True:
                    d_9_maxRollbackBudget_ = d_6_genBudget_
                d_10_constrainedPrompt_: _dafny.Seq
                d_10_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                d_11_resultConstrained_: _dafny.Seq
                out1_: _dafny.Seq
                out1_ = (d_0_helpers_).RegenerateUnitOnGroundingFailure(lm, parser, d_10_constrainedPrompt_, currentConstrainedOut, eosToken, d_7_maxStepsPerUnit_, d_8_maxRetries_, d_9_maxRollbackBudget_)
                d_11_resultConstrained_ = out1_
                d_12_stepsUsed_: int
                d_12_stepsUsed_ = ((d_8_maxRetries_) + (1)) * (d_7_maxStepsPerUnit_)
                if (d_12_stepsUsed_) > (d_6_genBudget_):
                    d_12_stepsUsed_ = d_6_genBudget_
                d_13_stableLen_: int
                d_13_stableLen_ = (len(generated)) - (len(currentConstrainedOut))
                d_14_stablePrefix_: _dafny.Seq
                d_14_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:d_13_stableLen_:])
                generated = (d_14_stablePrefix_) + (d_11_resultConstrained_)
                currentConstrainedOut = d_11_resultConstrained_
                d_2_steps_ = (d_2_steps_) + (d_12_stepsUsed_)
            if ((d_2_steps_) < (maxSteps)) and (insideConstrainedOut):
                d_15_closeBudget_: int
                d_15_closeBudget_ = (maxSteps) - (d_2_steps_)
                d_16_cg_: _dafny.Seq
                d_17_ci_: bool
                d_18_cc_: _dafny.Seq
                out2_: _dafny.Seq
                out3_: bool
                out4_: _dafny.Seq
                out2_, out3_, out4_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_15_closeBudget_)
                d_16_cg_ = out2_
                d_17_ci_ = out3_
                d_18_cc_ = out4_
                generated = d_16_cg_
                insideConstrainedOut = d_17_ci_
                currentConstrainedOut = d_18_cc_
                d_2_steps_ = maxSteps
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

