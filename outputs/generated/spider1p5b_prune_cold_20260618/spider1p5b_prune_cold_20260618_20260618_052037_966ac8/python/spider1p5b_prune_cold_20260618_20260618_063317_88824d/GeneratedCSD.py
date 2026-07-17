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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Output exactly: SQL: <<YOUR_SQL_QUERY>> where YOUR_SQL_QUERY is a single valid SQL SELECT statement using only schema tables and columns. No semicolons. No double quotes. Use single quotes for strings.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_closeBudgetReserve_: int
        d_2_closeBudgetReserve_ = 300
        if (not(insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
            d_3_preambleBudget_: int
            d_3_preambleBudget_ = 4
            if (d_3_preambleBudget_) > ((maxSteps) - (d_1_steps_)):
                d_3_preambleBudget_ = (maxSteps) - (d_1_steps_)
            d_4_gout_: _dafny.Seq
            d_5_stoppedOnOpen_: bool
            d_6_stoppedOnEos_: bool
            d_7_stepsUsed_: int
            out0_: _dafny.Seq
            out1_: bool
            out2_: bool
            out3_: int
            out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_3_preambleBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
            d_4_gout_ = out0_
            d_5_stoppedOnOpen_ = out1_
            d_6_stoppedOnEos_ = out2_
            d_7_stepsUsed_ = out3_
            generated = d_4_gout_
            d_1_steps_ = (d_1_steps_) + (d_7_stepsUsed_)
            if d_6_stoppedOnEos_:
                pass
            elif d_5_stoppedOnOpen_:
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
            while ((d_1_steps_) < (maxSteps)) and (insideConstrainedOut):
                with _dafny.c_label("0"):
                    if ((d_1_steps_) + (d_2_closeBudgetReserve_)) >= (maxSteps):
                        d_14_closeBudget_: int
                        d_14_closeBudget_ = (maxSteps) - (d_1_steps_)
                        d_15_fcg_: _dafny.Seq
                        d_16_fci_: bool
                        d_17_fcc_: _dafny.Seq
                        out10_: _dafny.Seq
                        out11_: bool
                        out12_: _dafny.Seq
                        out10_, out11_, out12_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_14_closeBudget_)
                        d_15_fcg_ = out10_
                        d_16_fci_ = out11_
                        d_17_fcc_ = out12_
                        generated = d_15_fcg_
                        insideConstrainedOut = d_16_fci_
                        currentConstrainedOut = d_17_fcc_
                        d_1_steps_ = maxSteps
                        raise _dafny.Break("0")
                    d_18_narrow_: bool
                    out13_: bool
                    out13_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                    d_18_narrow_ = out13_
                    if d_18_narrow_:
                        d_19_rg_: _dafny.Seq
                        d_20_rc_: _dafny.Seq
                        out14_: _dafny.Seq
                        out15_: _dafny.Seq
                        out14_, out15_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                        d_19_rg_ = out14_
                        d_20_rc_ = out15_
                        generated = d_19_rg_
                        currentConstrainedOut = d_20_rc_
                    d_21_cg_: _dafny.Seq
                    d_22_ci_: bool
                    d_23_cc_: _dafny.Seq
                    d_24_closed_: bool
                    out16_: _dafny.Seq
                    out17_: bool
                    out18_: _dafny.Seq
                    out19_: bool
                    out16_, out17_, out18_, out19_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                    d_21_cg_ = out16_
                    d_22_ci_ = out17_
                    d_23_cc_ = out18_
                    d_24_closed_ = out19_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if d_24_closed_:
                        generated = d_21_cg_
                        insideConstrainedOut = d_22_ci_
                        currentConstrainedOut = d_23_cc_
                        raise _dafny.Break("0")
                    if (d_1_steps_) >= (maxSteps):
                        raise _dafny.Break("0")
                    d_25_constrainedPrompt_: _dafny.Seq
                    d_25_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                    d_26_next_: _dafny.Seq
                    out20_: _dafny.Seq
                    out20_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_25_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('3e0'), eosToken)
                    d_26_next_ = out20_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if (d_26_next_) == (eosToken):
                        d_27_closeBudget_: int
                        d_27_closeBudget_ = (maxSteps) - (d_1_steps_)
                        if (d_27_closeBudget_) > (0):
                            d_28_fcg_: _dafny.Seq
                            d_29_fci_: bool
                            d_30_fcc_: _dafny.Seq
                            out21_: _dafny.Seq
                            out22_: bool
                            out23_: _dafny.Seq
                            out21_, out22_, out23_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_27_closeBudget_)
                            d_28_fcg_ = out21_
                            d_29_fci_ = out22_
                            d_30_fcc_ = out23_
                            generated = d_28_fcg_
                            insideConstrainedOut = d_29_fci_
                            currentConstrainedOut = d_30_fcc_
                            d_1_steps_ = maxSteps
                        raise _dafny.Break("0")
                    elif True:
                        d_31_appendedGenerated_: _dafny.Seq
                        d_32_appendedInside_: bool
                        d_33_appendedCurrent_: _dafny.Seq
                        out24_: _dafny.Seq
                        out25_: bool
                        out26_: _dafny.Seq
                        out24_, out25_, out26_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_26_next_)
                        d_31_appendedGenerated_ = out24_
                        d_32_appendedInside_ = out25_
                        d_33_appendedCurrent_ = out26_
                        generated = d_31_appendedGenerated_
                        insideConstrainedOut = d_32_appendedInside_
                        currentConstrainedOut = d_33_appendedCurrent_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_34_closeBudget_: int
            d_34_closeBudget_ = (maxSteps) - (d_1_steps_)
            d_35_fcg_: _dafny.Seq
            d_36_fci_: bool
            d_37_fcc_: _dafny.Seq
            out27_: _dafny.Seq
            out28_: bool
            out29_: _dafny.Seq
            out27_, out28_, out29_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_34_closeBudget_)
            d_35_fcg_ = out27_
            d_36_fci_ = out28_
            d_37_fcc_ = out29_
            generated = d_35_fcg_
            insideConstrainedOut = d_36_fci_
            currentConstrainedOut = d_37_fcc_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

