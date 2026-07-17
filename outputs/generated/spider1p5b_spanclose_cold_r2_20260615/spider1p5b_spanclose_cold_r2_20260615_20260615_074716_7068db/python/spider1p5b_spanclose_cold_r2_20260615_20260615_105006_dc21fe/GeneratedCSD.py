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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate a valid SQL query. Output format: SQL: <<your SQL query here>>. Use only table and column names from the schema.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_freeLimit_: int
        d_2_freeLimit_ = 4
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
                        insideConstrainedOut = True
                        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    pass
            pass
        if (not(insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
            d_4_og_: _dafny.Seq
            d_5_oi_: bool
            d_6_oc_: _dafny.Seq
            out1_: _dafny.Seq
            out2_: bool
            out3_: _dafny.Seq
            out1_, out2_, out3_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_4_og_ = out1_
            d_5_oi_ = out2_
            d_6_oc_ = out3_
            generated = d_4_og_
            insideConstrainedOut = d_5_oi_
            currentConstrainedOut = d_6_oc_
            d_1_steps_ = (d_1_steps_) + (1)
        d_7_phase2Reserve_: int
        d_7_phase2Reserve_ = 200
        if (maxSteps) < (400):
            d_7_phase2Reserve_ = _dafny.euclidian_division(maxSteps, 2)
        if (d_7_phase2Reserve_) > (maxSteps):
            d_7_phase2Reserve_ = maxSteps
        d_8_phase2Limit_: int = int(0)
        if (maxSteps) >= (d_7_phase2Reserve_):
            d_8_phase2Limit_ = (maxSteps) - (d_7_phase2Reserve_)
        elif True:
            d_8_phase2Limit_ = 0
        with _dafny.label("1"):
            while ((d_1_steps_) < (d_8_phase2Limit_)) and (insideConstrainedOut):
                with _dafny.c_label("1"):
                    d_9_cg_: _dafny.Seq
                    d_10_ci_: bool
                    d_11_cc_: _dafny.Seq
                    d_12_closed_: bool
                    out4_: _dafny.Seq
                    out5_: bool
                    out6_: _dafny.Seq
                    out7_: bool
                    out4_, out5_, out6_, out7_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                    d_9_cg_ = out4_
                    d_10_ci_ = out5_
                    d_11_cc_ = out6_
                    d_12_closed_ = out7_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if d_12_closed_:
                        generated = d_9_cg_
                        insideConstrainedOut = d_10_ci_
                        currentConstrainedOut = d_11_cc_
                        raise _dafny.Break("1")
                    elif True:
                        d_13_constrainedPrompt_: _dafny.Seq
                        d_13_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_14_next_: _dafny.Seq
                        out8_: _dafny.Seq
                        out8_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_13_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                        d_14_next_ = out8_
                        if (d_14_next_) == (eosToken):
                            raise _dafny.Break("1")
                        elif True:
                            d_15_ag_: _dafny.Seq
                            d_16_ai_: bool
                            d_17_ac_: _dafny.Seq
                            out9_: _dafny.Seq
                            out10_: bool
                            out11_: _dafny.Seq
                            out9_, out10_, out11_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_14_next_)
                            d_15_ag_ = out9_
                            d_16_ai_ = out10_
                            d_17_ac_ = out11_
                            generated = d_15_ag_
                            insideConstrainedOut = d_16_ai_
                            currentConstrainedOut = d_17_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_18_closeBudget_: int
            d_18_closeBudget_ = (maxSteps) - (d_1_steps_)
            d_19_cg_: _dafny.Seq
            d_20_ci_: bool
            d_21_cc_: _dafny.Seq
            out12_: _dafny.Seq
            out13_: bool
            out14_: _dafny.Seq
            out12_, out13_, out14_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_18_closeBudget_)
            d_19_cg_ = out12_
            d_20_ci_ = out13_
            d_21_cc_ = out14_
            generated = d_19_cg_
            insideConstrainedOut = d_20_ci_
            currentConstrainedOut = d_21_cc_
            d_1_steps_ = maxSteps
        if ((d_1_steps_) == (0)) and ((maxSteps) > (0)):
            d_22_next_: _dafny.Seq
            out15_: _dafny.Seq
            out15_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
            d_22_next_ = out15_
            d_1_steps_ = (d_1_steps_) + (1)
            if (d_22_next_) != (eosToken):
                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_22_next_]))
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

