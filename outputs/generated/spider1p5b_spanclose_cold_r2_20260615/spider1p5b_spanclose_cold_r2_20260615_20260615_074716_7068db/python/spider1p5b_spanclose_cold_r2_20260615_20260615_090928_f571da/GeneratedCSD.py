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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Output exactly: SQL: <<SELECT ... FROM ...>> where the content between << and >> is a single valid SQL query using only the tables and columns in the provided schema. No explanations, no markdown, no semicolons inside the span.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_freeGenLimit_: int
        d_2_freeGenLimit_ = 8
        if (d_2_freeGenLimit_) > (maxSteps):
            d_2_freeGenLimit_ = maxSteps
        with _dafny.label("0"):
            while ((d_1_steps_) < (d_2_freeGenLimit_)) and (not(insideConstrainedOut)):
                with _dafny.c_label("0"):
                    d_3_next_: _dafny.Seq
                    out0_: _dafny.Seq
                    out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                    d_3_next_ = out0_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if (d_3_next_) == (eosToken):
                        raise _dafny.Break("0")
                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_3_next_]))
                    if (d_3_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                        d_4_eg_: _dafny.Seq
                        d_5_ei_: bool
                        d_6_ec_: _dafny.Seq
                        out1_: _dafny.Seq
                        out2_: bool
                        out3_: _dafny.Seq
                        out1_, out2_, out3_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                        d_4_eg_ = out1_
                        d_5_ei_ = out2_
                        d_6_ec_ = out3_
                        generated = d_4_eg_
                        insideConstrainedOut = d_5_ei_
                        currentConstrainedOut = d_6_ec_
                    pass
            pass
        if (not(insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
            d_7_og_: _dafny.Seq
            d_8_oi_: bool
            d_9_oc_: _dafny.Seq
            out4_: _dafny.Seq
            out5_: bool
            out6_: _dafny.Seq
            out4_, out5_, out6_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_7_og_ = out4_
            d_8_oi_ = out5_
            d_9_oc_ = out6_
            generated = d_7_og_
            insideConstrainedOut = d_8_oi_
            currentConstrainedOut = d_9_oc_
            d_1_steps_ = (d_1_steps_) + (1)
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_10_reserveBudget_: int
            d_10_reserveBudget_ = 1
            d_11_constrainedBudget_: int
            d_11_constrainedBudget_ = 0
            if ((maxSteps) - (d_1_steps_)) > (d_10_reserveBudget_):
                d_11_constrainedBudget_ = ((maxSteps) - (d_1_steps_)) - (d_10_reserveBudget_)
            d_12_maxStepsPerUnit_: int
            d_12_maxStepsPerUnit_ = 40
            d_13_maxRetries_: int
            d_13_maxRetries_ = 2
            d_14_maxRollbackBudget_: int
            d_14_maxRollbackBudget_ = 10
            d_15_unitCost_: int
            d_15_unitCost_ = ((d_13_maxRetries_) + (1)) * (d_12_maxStepsPerUnit_)
            if ((d_11_constrainedBudget_) >= (d_15_unitCost_)) and (((d_1_steps_) + (d_15_unitCost_)) <= (maxSteps)):
                d_16_resultConstrained_: _dafny.Seq
                out7_: _dafny.Seq
                out7_ = (d_0_helpers_).RegenerateUnitOnGroundingFailure(lm, parser, prompt, currentConstrainedOut, eosToken, d_12_maxStepsPerUnit_, d_13_maxRetries_, d_14_maxRollbackBudget_)
                d_16_resultConstrained_ = out7_
                d_17_prefixLen_: int
                d_17_prefixLen_ = (len(generated)) - (len(currentConstrainedOut))
                generated = (_dafny.SeqWithoutIsStrInference((generated)[:d_17_prefixLen_:])) + (d_16_resultConstrained_)
                currentConstrainedOut = d_16_resultConstrained_
                d_1_steps_ = (d_1_steps_) + (d_15_unitCost_)
            elif True:
                with _dafny.label("3_2_0"):
                    while ((d_1_steps_) < (maxSteps)) and (insideConstrainedOut):
                        with _dafny.c_label("3_2_0"):
                            d_18_cg_: _dafny.Seq
                            d_19_ci_: bool
                            d_20_cc_: _dafny.Seq
                            d_21_closed_: bool
                            out8_: _dafny.Seq
                            out9_: bool
                            out10_: _dafny.Seq
                            out11_: bool
                            out8_, out9_, out10_, out11_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                            d_18_cg_ = out8_
                            d_19_ci_ = out9_
                            d_20_cc_ = out10_
                            d_21_closed_ = out11_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if d_21_closed_:
                                generated = d_18_cg_
                                insideConstrainedOut = d_19_ci_
                                currentConstrainedOut = d_20_cc_
                                raise _dafny.Break("3_2_0")
                            elif True:
                                d_22_constrainedPrompt_: _dafny.Seq
                                d_22_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_23_next_: _dafny.Seq
                                out12_: _dafny.Seq
                                out12_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_22_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                                d_23_next_ = out12_
                                if (d_23_next_) == (eosToken):
                                    raise _dafny.Break("3_2_0")
                                elif True:
                                    d_24_ag_: _dafny.Seq
                                    d_25_ai_: bool
                                    d_26_ac_: _dafny.Seq
                                    out13_: _dafny.Seq
                                    out14_: bool
                                    out15_: _dafny.Seq
                                    out13_, out14_, out15_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_23_next_)
                                    d_24_ag_ = out13_
                                    d_25_ai_ = out14_
                                    d_26_ac_ = out15_
                                    generated = d_24_ag_
                                    insideConstrainedOut = d_25_ai_
                                    currentConstrainedOut = d_26_ac_
                            pass
                    pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_27_closeBudget_: int
            d_27_closeBudget_ = (maxSteps) - (d_1_steps_)
            d_28_cg_: _dafny.Seq
            d_29_ci_: bool
            d_30_cc_: _dafny.Seq
            out16_: _dafny.Seq
            out17_: bool
            out18_: _dafny.Seq
            out16_, out17_, out18_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_27_closeBudget_)
            d_28_cg_ = out16_
            d_29_ci_ = out17_
            d_30_cc_ = out18_
            generated = d_28_cg_
            insideConstrainedOut = d_29_ci_
            currentConstrainedOut = d_30_cc_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

