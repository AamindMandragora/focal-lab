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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate exactly one SQL query. Output: SQL: <<your SQL query here>>. Use only schema tables and columns. No markdown, no explanation."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_minUnconstrainedSteps_: int
        d_3_minUnconstrainedSteps_ = 8
        d_4_unconstrainedCount_: int
        d_4_unconstrainedCount_ = 0
        with _dafny.label("0"):
            while (((d_2_steps_) < (maxSteps)) and (not(insideConstrainedOut))) and ((d_4_unconstrainedCount_) < (d_3_minUnconstrainedSteps_)):
                with _dafny.c_label("0"):
                    d_5_next_: _dafny.Seq
                    out0_: _dafny.Seq
                    out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                    d_5_next_ = out0_
                    d_2_steps_ = (d_2_steps_) + (1)
                    d_4_unconstrainedCount_ = (d_4_unconstrainedCount_) + (1)
                    if (d_5_next_) == (eosToken):
                        raise _dafny.Break("0")
                    elif True:
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_5_next_]))
                        if (d_5_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                            insideConstrainedOut = True
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    pass
            pass
        with _dafny.label("1"):
            while ((d_2_steps_) < (maxSteps)) and (not(insideConstrainedOut)):
                with _dafny.c_label("1"):
                    d_6_next_: _dafny.Seq
                    out1_: _dafny.Seq
                    out1_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                    d_6_next_ = out1_
                    d_2_steps_ = (d_2_steps_) + (1)
                    if (d_6_next_) == (eosToken):
                        raise _dafny.Break("1")
                    elif True:
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_6_next_]))
                        if (d_6_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                            insideConstrainedOut = True
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    pass
            pass
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_7_reserveForClose_: int
            d_7_reserveForClose_ = 30
            d_8_innerBudget_: int
            if ((maxSteps) - (d_2_steps_)) > (d_7_reserveForClose_):
                d_8_innerBudget_ = ((maxSteps) - (d_2_steps_)) - (d_7_reserveForClose_)
            elif True:
                d_8_innerBudget_ = 0
            if (d_8_innerBudget_) > (0):
                d_9_maxStepsPerUnit_: int
                d_9_maxStepsPerUnit_ = 20
                d_10_maxRetries_: int
                d_10_maxRetries_ = 3
                d_11_maxRollbackBudget_: int
                d_11_maxRollbackBudget_ = 10
                d_12_constrainedPrompt_: _dafny.Seq
                d_12_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                d_13_innerStepsUsed_: int
                d_13_innerStepsUsed_ = 0
                d_14_maxInnerCalls_: int
                d_14_maxInnerCalls_ = _dafny.euclidian_division(d_8_innerBudget_, (((d_10_maxRetries_) + (1)) * (d_9_maxStepsPerUnit_) if (((d_10_maxRetries_) + (1)) * (d_9_maxStepsPerUnit_)) > (0) else 1))
                if (d_14_maxInnerCalls_) < (1):
                    d_14_maxInnerCalls_ = 1
                d_15_callsDone_: int
                d_15_callsDone_ = 0
                while (((d_15_callsDone_) < (d_14_maxInnerCalls_)) and (((d_2_steps_) + (d_13_innerStepsUsed_)) < ((maxSteps) - (d_7_reserveForClose_)))) and (insideConstrainedOut):
                    d_16_stepsBeforeCall_: int
                    d_16_stepsBeforeCall_ = cost
                    d_17_resultConstrained_: _dafny.Seq
                    out2_: _dafny.Seq
                    out2_ = (d_0_helpers_).RegenerateUnitOnGroundingFailure(lm, parser, d_12_constrainedPrompt_, currentConstrainedOut, eosToken, d_9_maxStepsPerUnit_, d_10_maxRetries_, d_11_maxRollbackBudget_)
                    d_17_resultConstrained_ = out2_
                    d_18_stepsAfterCall_: int
                    d_18_stepsAfterCall_ = cost
                    d_19_usedThisCall_: int
                    if (d_18_stepsAfterCall_) > (d_16_stepsBeforeCall_):
                        d_19_usedThisCall_ = (d_18_stepsAfterCall_) - (d_16_stepsBeforeCall_)
                    elif True:
                        d_19_usedThisCall_ = 0
                    if (d_17_resultConstrained_) == (currentConstrainedOut):
                        d_15_callsDone_ = d_14_maxInnerCalls_
                    elif True:
                        d_20_stablePrefix_: _dafny.Seq
                        d_20_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        if ((len(d_20_stablePrefix_)) + (len(d_17_resultConstrained_))) <= ((len(generated)) + (d_19_usedThisCall_)):
                            generated = (d_20_stablePrefix_) + (d_17_resultConstrained_)
                            currentConstrainedOut = d_17_resultConstrained_
                        d_13_innerStepsUsed_ = (d_13_innerStepsUsed_) + (d_19_usedThisCall_)
                        d_15_callsDone_ = (d_15_callsDone_) + (1)
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            d_15_callsDone_ = d_14_maxInnerCalls_
                d_2_steps_ = (d_2_steps_) + (d_13_innerStepsUsed_)
            if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
                d_21_closeBudget_: int
                d_21_closeBudget_ = (maxSteps) - (d_2_steps_)
                d_22_cg_: _dafny.Seq
                d_23_ci_: bool
                d_24_cc_: _dafny.Seq
                out3_: _dafny.Seq
                out4_: bool
                out5_: _dafny.Seq
                out3_, out4_, out5_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_21_closeBudget_)
                d_22_cg_ = out3_
                d_23_ci_ = out4_
                d_24_cc_ = out5_
                generated = d_22_cg_
                insideConstrainedOut = d_23_ci_
                currentConstrainedOut = d_24_cc_
                d_2_steps_ = maxSteps
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

