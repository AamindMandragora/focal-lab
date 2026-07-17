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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate exactly one complete SQL query. Use JOINs when data from multiple tables is needed. Use GROUP BY with aggregates. Use INTERSECT/UNION for set operations. Output only the SQL query, no explanation.")))
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
            d_3_fillBudget_ = _dafny.euclidian_division((d_2_rem_) * (3), 4)
            if (d_3_fillBudget_) >= (1):
                d_4_stable_: _dafny.Seq
                d_4_stable_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                d_5_constrainedPrompt_: _dafny.Seq
                d_5_constrainedPrompt_ = (prompt) + (d_4_stable_)
                d_6_filled_: _dafny.Seq
                d_7_terminatedByEos_: bool
                out3_: _dafny.Seq
                out4_: bool
                out3_, out4_ = (d_0_helpers_).ConstrainedGeneration(lm, parser, d_5_constrainedPrompt_, d_3_fillBudget_, eosToken)
                d_6_filled_ = out3_
                d_7_terminatedByEos_ = out4_
                if (len(currentConstrainedOut)) == (0):
                    generated = (d_4_stable_) + (d_6_filled_)
                    currentConstrainedOut = d_6_filled_
                elif True:
                    d_8_filled2_: _dafny.Seq
                    out5_: _dafny.Seq
                    out5_ = (d_0_helpers_).RegenerateUnitOnGroundingFailure(lm, parser, d_5_constrainedPrompt_, currentConstrainedOut, eosToken, d_3_fillBudget_, 3, d_3_fillBudget_)
                    d_8_filled2_ = out5_
                    generated = (d_4_stable_) + (d_8_filled2_)
                    currentConstrainedOut = d_8_filled2_
                d_1_steps_ = (d_1_steps_) + (d_3_fillBudget_)
        with _dafny.label("0"):
            while ((d_1_steps_) < (maxSteps)) and (insideConstrainedOut):
                with _dafny.c_label("0"):
                    if ((d_1_steps_) + (3)) >= (maxSteps):
                        raise _dafny.Break("0")
                    d_9_cg_: _dafny.Seq
                    d_10_ci_: bool
                    d_11_cc_: _dafny.Seq
                    d_12_closed_: bool
                    out6_: _dafny.Seq
                    out7_: bool
                    out8_: _dafny.Seq
                    out9_: bool
                    out6_, out7_, out8_, out9_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                    d_9_cg_ = out6_
                    d_10_ci_ = out7_
                    d_11_cc_ = out8_
                    d_12_closed_ = out9_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if d_12_closed_:
                        generated = d_9_cg_
                        insideConstrainedOut = d_10_ci_
                        currentConstrainedOut = d_11_cc_
                    elif True:
                        d_13_constrainedPrompt_: _dafny.Seq
                        d_13_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_14_next_: _dafny.Seq
                        out10_: _dafny.Seq
                        out10_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_13_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                        d_14_next_ = out10_
                        if (d_14_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_15_appendedGenerated_: _dafny.Seq
                            d_16_appendedInside_: bool
                            d_17_appendedCurrent_: _dafny.Seq
                            out11_: _dafny.Seq
                            out12_: bool
                            out13_: _dafny.Seq
                            out11_, out12_, out13_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_14_next_)
                            d_15_appendedGenerated_ = out11_
                            d_16_appendedInside_ = out12_
                            d_17_appendedCurrent_ = out13_
                            generated = d_15_appendedGenerated_
                            insideConstrainedOut = d_16_appendedInside_
                            currentConstrainedOut = d_17_appendedCurrent_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_18_closeBudget_: int
            d_18_closeBudget_ = (maxSteps) - (d_1_steps_)
            out14_: _dafny.Seq
            out15_: bool
            out16_: _dafny.Seq
            out14_, out15_, out16_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_18_closeBudget_)
            generated = out14_
            insideConstrainedOut = out15_
            currentConstrainedOut = out16_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

