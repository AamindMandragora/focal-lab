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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Always output: SQL: <<your SQL query here>>"))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        if ((d_2_steps_) < (maxSteps)) and (not(insideConstrainedOut)):
            d_3_chunkBudget_: int
            d_3_chunkBudget_ = _dafny.euclidian_division(maxSteps, 4)
            if (d_3_chunkBudget_) < (1):
                d_3_chunkBudget_ = 1
            if (d_3_chunkBudget_) > ((maxSteps) - (d_2_steps_)):
                d_3_chunkBudget_ = (maxSteps) - (d_2_steps_)
            d_4_cg_: _dafny.Seq
            d_5_stoppedOnOpen_: bool
            d_6_stoppedOnEos_: bool
            d_7_stepsUsed_: int
            out0_: _dafny.Seq
            out1_: bool
            out2_: bool
            out3_: int
            out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_3_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
            d_4_cg_ = out0_
            d_5_stoppedOnOpen_ = out1_
            d_6_stoppedOnEos_ = out2_
            d_7_stepsUsed_ = out3_
            d_2_steps_ = (d_2_steps_) + (d_7_stepsUsed_)
            generated = d_4_cg_
            if d_5_stoppedOnOpen_:
                d_8_eg_: _dafny.Seq
                d_9_ei_: bool
                d_10_ec_: _dafny.Seq
                out4_: _dafny.Seq
                out5_: bool
                out6_: _dafny.Seq
                out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                d_8_eg_ = out4_
                d_9_ei_ = out5_
                d_10_ec_ = out6_
                generated = d_8_eg_
                insideConstrainedOut = d_9_ei_
                currentConstrainedOut = d_10_ec_
            elif d_6_stoppedOnEos_:
                cost = d_2_steps_
                return generated, insideConstrainedOut, currentConstrainedOut, cost
            elif True:
                if (d_2_steps_) < (maxSteps):
                    d_11_og_: _dafny.Seq
                    d_12_oi_: bool
                    d_13_oc_: _dafny.Seq
                    out7_: _dafny.Seq
                    out8_: bool
                    out9_: _dafny.Seq
                    out7_, out8_, out9_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                    d_11_og_ = out7_
                    d_12_oi_ = out8_
                    d_13_oc_ = out9_
                    d_2_steps_ = (d_2_steps_) + (1)
                    generated = d_11_og_
                    insideConstrainedOut = d_12_oi_
                    currentConstrainedOut = d_13_oc_
        with _dafny.label("0"):
            while ((d_2_steps_) < (maxSteps)) and (insideConstrainedOut):
                with _dafny.c_label("0"):
                    d_14_cg_: _dafny.Seq
                    d_15_ci_: bool
                    d_16_cc_: _dafny.Seq
                    d_17_closed_: bool
                    out10_: _dafny.Seq
                    out11_: bool
                    out12_: _dafny.Seq
                    out13_: bool
                    out10_, out11_, out12_, out13_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                    d_14_cg_ = out10_
                    d_15_ci_ = out11_
                    d_16_cc_ = out12_
                    d_17_closed_ = out13_
                    d_2_steps_ = (d_2_steps_) + (1)
                    if d_17_closed_:
                        generated = d_14_cg_
                        insideConstrainedOut = d_15_ci_
                        currentConstrainedOut = d_16_cc_
                    elif True:
                        d_18_constrainedPrompt_: _dafny.Seq
                        d_18_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_19_next_: _dafny.Seq
                        out14_: _dafny.Seq
                        out14_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_18_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                        d_19_next_ = out14_
                        if (d_19_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_20_ag_: _dafny.Seq
                            d_21_ai_: bool
                            d_22_ac_: _dafny.Seq
                            out15_: _dafny.Seq
                            out16_: bool
                            out17_: _dafny.Seq
                            out15_, out16_, out17_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_19_next_)
                            d_20_ag_ = out15_
                            d_21_ai_ = out16_
                            d_22_ac_ = out17_
                            generated = d_20_ag_
                            insideConstrainedOut = d_21_ai_
                            currentConstrainedOut = d_22_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_23_closeBudget_: int
            d_23_closeBudget_ = (maxSteps) - (d_2_steps_)
            d_24_cg_: _dafny.Seq
            d_25_ci_: bool
            d_26_cc_: _dafny.Seq
            out18_: _dafny.Seq
            out19_: bool
            out20_: _dafny.Seq
            out18_, out19_, out20_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_23_closeBudget_)
            d_24_cg_ = out18_
            d_25_ci_ = out19_
            d_26_cc_ = out20_
            generated = d_24_cg_
            insideConstrainedOut = d_25_ci_
            currentConstrainedOut = d_26_cc_
            d_2_steps_ = maxSteps
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

