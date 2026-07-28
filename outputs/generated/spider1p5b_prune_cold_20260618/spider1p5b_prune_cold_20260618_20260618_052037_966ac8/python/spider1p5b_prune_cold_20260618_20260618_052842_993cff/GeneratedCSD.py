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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Output exactly: SQL: <<YOUR SQL QUERY HERE>> where the content between << and >> is a single valid SQL SELECT statement using only the tables and columns from the schema. Nothing else. No semicolons inside the span. No markdown.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_narrowThreshold_: int
        d_2_narrowThreshold_ = 16
        if (not(insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
            d_3_preambleMax_: int
            d_3_preambleMax_ = 8
            if (d_3_preambleMax_) > ((maxSteps) - (d_1_steps_)):
                d_3_preambleMax_ = (maxSteps) - (d_1_steps_)
            d_4_chunkGenerated_: _dafny.Seq
            d_5_stoppedOnOpenSpan_: bool
            d_6_stoppedOnEos_: bool
            d_7_chunkStepsUsed_: int
            out0_: _dafny.Seq
            out1_: bool
            out2_: bool
            out3_: int
            out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_3_preambleMax_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
            d_4_chunkGenerated_ = out0_
            d_5_stoppedOnOpenSpan_ = out1_
            d_6_stoppedOnEos_ = out2_
            d_7_chunkStepsUsed_ = out3_
            generated = d_4_chunkGenerated_
            d_1_steps_ = (d_1_steps_) + (d_7_chunkStepsUsed_)
            if d_6_stoppedOnEos_:
                cost = d_1_steps_
                insideConstrainedOut = False
                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                return generated, insideConstrainedOut, currentConstrainedOut, cost
            elif d_5_stoppedOnOpenSpan_:
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
            elif True:
                if (d_1_steps_) < (maxSteps):
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
                    generated = d_11_og_
                    insideConstrainedOut = d_12_oi_
                    currentConstrainedOut = d_13_oc_
                    d_1_steps_ = (d_1_steps_) + (1)
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        raise _dafny.Break("0")
                    elif True:
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
                            d_18_remainingAfterClose_: int
                            if (d_1_steps_) < (maxSteps):
                                d_18_remainingAfterClose_ = (maxSteps) - (d_1_steps_)
                            elif True:
                                d_18_remainingAfterClose_ = 0
                            if ((d_18_remainingAfterClose_) <= (4)) and ((d_1_steps_) < (maxSteps)):
                                d_19_closeBudget_: int
                                d_19_closeBudget_ = (maxSteps) - (d_1_steps_)
                                d_20_fcg_: _dafny.Seq
                                d_21_fci_: bool
                                d_22_fcc_: _dafny.Seq
                                out14_: _dafny.Seq
                                out15_: bool
                                out16_: _dafny.Seq
                                out14_, out15_, out16_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_19_closeBudget_)
                                d_20_fcg_ = out14_
                                d_21_fci_ = out15_
                                d_22_fcc_ = out16_
                                generated = d_20_fcg_
                                insideConstrainedOut = d_21_fci_
                                currentConstrainedOut = d_22_fcc_
                                d_1_steps_ = maxSteps
                                raise _dafny.Break("0")
                            elif True:
                                d_23_constrainedPrompt_: _dafny.Seq
                                d_23_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_24_next_: _dafny.Seq
                                out17_: _dafny.Seq
                                out17_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_23_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_2_narrowThreshold_, eosToken)
                                d_24_next_ = out17_
                                if (d_24_next_) == (eosToken):
                                    if (d_1_steps_) < (maxSteps):
                                        d_25_closeBudget_: int
                                        d_25_closeBudget_ = (maxSteps) - (d_1_steps_)
                                        d_26_fcg_: _dafny.Seq
                                        d_27_fci_: bool
                                        d_28_fcc_: _dafny.Seq
                                        out18_: _dafny.Seq
                                        out19_: bool
                                        out20_: _dafny.Seq
                                        out18_, out19_, out20_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_25_closeBudget_)
                                        d_26_fcg_ = out18_
                                        d_27_fci_ = out19_
                                        d_28_fcc_ = out20_
                                        generated = d_26_fcg_
                                        insideConstrainedOut = d_27_fci_
                                        currentConstrainedOut = d_28_fcc_
                                        d_1_steps_ = maxSteps
                                    raise _dafny.Break("0")
                                elif True:
                                    d_29_appendedGenerated_: _dafny.Seq
                                    d_30_appendedInside_: bool
                                    d_31_appendedCurrent_: _dafny.Seq
                                    out21_: _dafny.Seq
                                    out22_: bool
                                    out23_: _dafny.Seq
                                    out21_, out22_, out23_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_24_next_)
                                    d_29_appendedGenerated_ = out21_
                                    d_30_appendedInside_ = out22_
                                    d_31_appendedCurrent_ = out23_
                                    generated = d_29_appendedGenerated_
                                    insideConstrainedOut = d_30_appendedInside_
                                    currentConstrainedOut = d_31_appendedCurrent_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_32_closeBudget_: int
            d_32_closeBudget_ = (maxSteps) - (d_1_steps_)
            d_33_fcg_: _dafny.Seq
            d_34_fci_: bool
            d_35_fcc_: _dafny.Seq
            out24_: _dafny.Seq
            out25_: bool
            out26_: _dafny.Seq
            out24_, out25_, out26_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_32_closeBudget_)
            d_33_fcg_ = out24_
            d_34_fci_ = out25_
            d_35_fcc_ = out26_
            generated = d_33_fcg_
            insideConstrainedOut = d_34_fci_
            currentConstrainedOut = d_35_fcc_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

