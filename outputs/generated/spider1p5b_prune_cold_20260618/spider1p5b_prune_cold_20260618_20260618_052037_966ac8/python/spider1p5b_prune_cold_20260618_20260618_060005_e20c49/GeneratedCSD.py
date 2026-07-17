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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "You are a SQL expert. Generate a single correct SQL query for the given question using only the provided schema. Output exactly: SQL: <<YOUR QUERY HERE>> with no other text, no markdown, no newlines inside the query, no semicolons.")))
        d_1_steps_: int
        d_1_steps_ = 0
        if (not(insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
            d_2_chunkBudget_: int
            d_2_chunkBudget_ = 6
            if (d_2_chunkBudget_) > ((maxSteps) - (d_1_steps_)):
                d_2_chunkBudget_ = (maxSteps) - (d_1_steps_)
            if (d_2_chunkBudget_) > (0):
                d_3_cg_: _dafny.Seq
                d_4_stoppedOnOpen_: bool
                d_5_stoppedOnEos_: bool
                d_6_stepsUsed_: int
                out0_: _dafny.Seq
                out1_: bool
                out2_: bool
                out3_: int
                out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_2_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                d_3_cg_ = out0_
                d_4_stoppedOnOpen_ = out1_
                d_5_stoppedOnEos_ = out2_
                d_6_stepsUsed_ = out3_
                generated = d_3_cg_
                d_1_steps_ = (d_1_steps_) + (d_6_stepsUsed_)
                if d_5_stoppedOnEos_:
                    cost = d_1_steps_
                    return generated, insideConstrainedOut, currentConstrainedOut, cost
                if d_4_stoppedOnOpen_:
                    d_7_eg_: _dafny.Seq
                    d_8_ei_: bool
                    d_9_ec_: _dafny.Seq
                    out4_: _dafny.Seq
                    out5_: bool
                    out6_: _dafny.Seq
                    out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                    d_7_eg_ = out4_
                    d_8_ei_ = out5_
                    d_9_ec_ = out6_
                    generated = d_7_eg_
                    insideConstrainedOut = d_8_ei_
                    currentConstrainedOut = d_9_ec_
        if (not(insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
            d_10_og_: _dafny.Seq
            d_11_oi_: bool
            d_12_oc_: _dafny.Seq
            out7_: _dafny.Seq
            out8_: bool
            out9_: _dafny.Seq
            out7_, out8_, out9_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_10_og_ = out7_
            d_11_oi_ = out8_
            d_12_oc_ = out9_
            generated = d_10_og_
            insideConstrainedOut = d_11_oi_
            currentConstrainedOut = d_12_oc_
            d_1_steps_ = (d_1_steps_) + (1)
        d_13_closeBudgetReserve_: int
        d_13_closeBudgetReserve_ = 80
        with _dafny.label("0"):
            while ((d_1_steps_) < (maxSteps)) and (insideConstrainedOut):
                with _dafny.c_label("0"):
                    if ((d_1_steps_) + (d_13_closeBudgetReserve_)) >= (maxSteps):
                        raise _dafny.Break("0")
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
                    d_1_steps_ = (d_1_steps_) + (1)
                    if d_17_closed_:
                        generated = d_14_cg_
                        insideConstrainedOut = d_15_ci_
                        currentConstrainedOut = d_16_cc_
                        raise _dafny.Break("0")
                    elif True:
                        d_18_next_: _dafny.Seq
                        out14_: _dafny.Seq
                        out14_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, prompt, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 10, eosToken)
                        d_18_next_ = out14_
                        if (d_18_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_19_ag_: _dafny.Seq
                            d_20_ai_: bool
                            d_21_ac_: _dafny.Seq
                            out15_: _dafny.Seq
                            out16_: bool
                            out17_: _dafny.Seq
                            out15_, out16_, out17_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_18_next_)
                            d_19_ag_ = out15_
                            d_20_ai_ = out16_
                            d_21_ac_ = out17_
                            generated = d_19_ag_
                            insideConstrainedOut = d_20_ai_
                            currentConstrainedOut = d_21_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_22_closeBudget_: int
            d_22_closeBudget_ = (maxSteps) - (d_1_steps_)
            d_23_fcg_: _dafny.Seq
            d_24_fci_: bool
            d_25_fcc_: _dafny.Seq
            out18_: _dafny.Seq
            out19_: bool
            out20_: _dafny.Seq
            out18_, out19_, out20_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_22_closeBudget_)
            d_23_fcg_ = out18_
            d_24_fci_ = out19_
            d_25_fcc_ = out20_
            generated = d_23_fcg_
            insideConstrainedOut = d_24_fci_
            currentConstrainedOut = d_25_fcc_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

