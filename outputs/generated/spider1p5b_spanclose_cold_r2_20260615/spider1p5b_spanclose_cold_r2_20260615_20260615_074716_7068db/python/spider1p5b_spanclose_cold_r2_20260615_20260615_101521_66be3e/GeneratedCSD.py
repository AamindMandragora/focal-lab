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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate a valid SQL query. Output format: SQL: <<query>>. Use only table and column names from the schema.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_freeLimit_: int
        d_2_freeLimit_ = 8
        if (d_2_freeLimit_) > (maxSteps):
            d_2_freeLimit_ = maxSteps
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
        d_10_phase2Limit_: int = int(0)
        if (maxSteps) >= (50):
            d_10_phase2Limit_ = (maxSteps) - (50)
        elif True:
            d_10_phase2Limit_ = maxSteps
        with _dafny.label("1"):
            while ((d_1_steps_) < (d_10_phase2Limit_)) and (insideConstrainedOut):
                with _dafny.c_label("1"):
                    d_11_cg_: _dafny.Seq
                    d_12_ci_: bool
                    d_13_cc_: _dafny.Seq
                    d_14_closed_: bool
                    out7_: _dafny.Seq
                    out8_: bool
                    out9_: _dafny.Seq
                    out10_: bool
                    out7_, out8_, out9_, out10_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                    d_11_cg_ = out7_
                    d_12_ci_ = out8_
                    d_13_cc_ = out9_
                    d_14_closed_ = out10_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if d_14_closed_:
                        generated = d_11_cg_
                        insideConstrainedOut = d_12_ci_
                        currentConstrainedOut = d_13_cc_
                        raise _dafny.Break("1")
                    elif True:
                        d_15_constrainedPrompt_: _dafny.Seq
                        d_15_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_16_next_: _dafny.Seq
                        out11_: _dafny.Seq
                        out11_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_15_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                        d_16_next_ = out11_
                        if (d_16_next_) == (eosToken):
                            raise _dafny.Break("1")
                        elif True:
                            d_17_ag_: _dafny.Seq
                            d_18_ai_: bool
                            d_19_ac_: _dafny.Seq
                            out12_: _dafny.Seq
                            out13_: bool
                            out14_: _dafny.Seq
                            out12_, out13_, out14_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_16_next_)
                            d_17_ag_ = out12_
                            d_18_ai_ = out13_
                            d_19_ac_ = out14_
                            generated = d_17_ag_
                            insideConstrainedOut = d_18_ai_
                            currentConstrainedOut = d_19_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_20_closeBudget_: int
            d_20_closeBudget_ = (maxSteps) - (d_1_steps_)
            d_21_cg_: _dafny.Seq
            d_22_ci_: bool
            d_23_cc_: _dafny.Seq
            out15_: _dafny.Seq
            out16_: bool
            out17_: _dafny.Seq
            out15_, out16_, out17_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_20_closeBudget_)
            d_21_cg_ = out15_
            d_22_ci_ = out16_
            d_23_cc_ = out17_
            generated = d_21_cg_
            insideConstrainedOut = d_22_ci_
            currentConstrainedOut = d_23_cc_
            d_1_steps_ = maxSteps
        if ((d_1_steps_) == (0)) and ((maxSteps) > (0)):
            d_24_next_: _dafny.Seq
            out18_: _dafny.Seq
            out18_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
            d_24_next_ = out18_
            d_1_steps_ = (d_1_steps_) + (1)
            if (d_24_next_) != (eosToken):
                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_24_next_]))
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

