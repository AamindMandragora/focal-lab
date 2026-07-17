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
        d_0_helpers_: VerifiedDecoderAgent.CSDHelpers
        nw0_ = VerifiedDecoderAgent.CSDHelpers()
        nw0_.ctor__()
        d_0_helpers_ = nw0_
        generated = generatedPrefix
        insideConstrainedOut = insideConstrained
        currentConstrainedOut = currentConstrained
        cost = 0
        d_1_guidance_: _dafny.Seq
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate a single valid SQL query. Output format: SQL: <<your SQL query here>>. Use only tables and columns from the provided schema. Do not include explanation or markdown."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_maxStepsPerUnit_: int
        d_2_maxStepsPerUnit_ = 20
        d_3_maxRetries_: int
        d_3_maxRetries_ = 3
        d_4_maxRollbackBudget_: int
        d_4_maxRollbackBudget_ = 15
        d_5_groundBound_: int
        d_5_groundBound_ = ((d_3_maxRetries_) + (1)) * (d_2_maxStepsPerUnit_)
        d_6_steps_: int
        d_6_steps_ = 0
        with _dafny.label("0"):
            while ((d_6_steps_) < (maxSteps)) and (not(insideConstrainedOut)):
                with _dafny.c_label("0"):
                    d_7_next_: _dafny.Seq
                    out0_: _dafny.Seq
                    out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                    d_7_next_ = out0_
                    d_6_steps_ = (d_6_steps_) + (1)
                    if (d_7_next_) == (eosToken):
                        raise _dafny.Break("0")
                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_7_next_]))
                    if (d_7_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                        insideConstrainedOut = True
                        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    pass
            pass
        if (insideConstrainedOut) and (((d_6_steps_) + (d_5_groundBound_)) <= (maxSteps)):
            d_8_constrainedPrompt_: _dafny.Seq
            d_8_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
            d_9_resultConstrained_: _dafny.Seq
            out1_: _dafny.Seq
            out1_ = (d_0_helpers_).RegenerateUnitOnGroundingFailure(lm, parser, d_8_constrainedPrompt_, currentConstrainedOut, eosToken, d_2_maxStepsPerUnit_, d_3_maxRetries_, d_4_maxRollbackBudget_)
            d_9_resultConstrained_ = out1_
            d_10_stableLen_: int
            d_10_stableLen_ = (len(generated)) - (len(currentConstrainedOut))
            d_11_stablePrefix_: _dafny.Seq
            d_11_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:d_10_stableLen_:])
            generated = (d_11_stablePrefix_) + (d_9_resultConstrained_)
            currentConstrainedOut = d_9_resultConstrained_
            d_6_steps_ = (d_6_steps_) + (d_5_groundBound_)
        if (insideConstrainedOut) and ((d_6_steps_) < (maxSteps)):
            d_12_remain_: int
            d_12_remain_ = (maxSteps) - (d_6_steps_)
            d_13_cg_: _dafny.Seq
            d_14_ci_: bool
            d_15_cc_: _dafny.Seq
            out2_: _dafny.Seq
            out3_: bool
            out4_: _dafny.Seq
            out2_, out3_, out4_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_12_remain_)
            d_13_cg_ = out2_
            d_14_ci_ = out3_
            d_15_cc_ = out4_
            generated = d_13_cg_
            insideConstrainedOut = d_14_ci_
            currentConstrainedOut = d_15_cc_
            d_6_steps_ = maxSteps
        cost = d_6_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

