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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Output exactly: SQL: <<your SQL query here>>. Use only tables and columns from the provided schema. No explanation, no markdown. Keep the SQL concise and correct.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_narrowThreshold_: int
        d_2_narrowThreshold_ = 12
        d_3_spanHasContent_: bool
        if (insideConstrained) and ((len(currentConstrained)) > (0)):
            d_3_spanHasContent_ = True
        elif True:
            d_3_spanHasContent_ = False
        d_4_spanTokenCount_: int
        if insideConstrained:
            d_4_spanTokenCount_ = len(currentConstrained)
        elif True:
            d_4_spanTokenCount_ = 0
        d_5_closingReserve_: int
        d_5_closingReserve_ = 10
        d_6_maxSpanTokens_: int
        d_6_maxSpanTokens_ = 200
        d_7_unconstrainedCount_: int
        d_7_unconstrainedCount_ = 0
        d_8_maxUnconstrainedBeforeForce_: int
        d_8_maxUnconstrainedBeforeForce_ = 6
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if (d_7_unconstrainedCount_) >= (d_8_maxUnconstrainedBeforeForce_):
                            d_9_og_: _dafny.Seq
                            d_10_oi_: bool
                            d_11_oc_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_9_og_ = out0_
                            d_10_oi_ = out1_
                            d_11_oc_ = out2_
                            generated = d_9_og_
                            insideConstrainedOut = d_10_oi_
                            currentConstrainedOut = d_11_oc_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_3_spanHasContent_ = False
                            d_4_spanTokenCount_ = 0
                        elif True:
                            d_12_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_12_next_ = out3_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_12_next_) == (eosToken):
                                if (d_1_steps_) < (maxSteps):
                                    d_13_og_: _dafny.Seq
                                    d_14_oi_: bool
                                    d_15_oc_: _dafny.Seq
                                    out4_: _dafny.Seq
                                    out5_: bool
                                    out6_: _dafny.Seq
                                    out4_, out5_, out6_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                    d_13_og_ = out4_
                                    d_14_oi_ = out5_
                                    d_15_oc_ = out6_
                                    generated = d_13_og_
                                    insideConstrainedOut = d_14_oi_
                                    currentConstrainedOut = d_15_oc_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    d_3_spanHasContent_ = False
                                    d_4_spanTokenCount_ = 0
                                    d_7_unconstrainedCount_ = (d_7_unconstrainedCount_) + (1)
                                elif True:
                                    raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_12_next_]))
                                d_7_unconstrainedCount_ = (d_7_unconstrainedCount_) + (1)
                                if (d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    d_16_eg_: _dafny.Seq
                                    d_17_ei_: bool
                                    d_18_ec_: _dafny.Seq
                                    out7_: _dafny.Seq
                                    out8_: bool
                                    out9_: _dafny.Seq
                                    out7_, out8_, out9_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                    d_16_eg_ = out7_
                                    d_17_ei_ = out8_
                                    d_18_ec_ = out9_
                                    generated = d_16_eg_
                                    insideConstrainedOut = d_17_ei_
                                    currentConstrainedOut = d_18_ec_
                                    d_3_spanHasContent_ = False
                                    d_4_spanTokenCount_ = 0
                    elif True:
                        d_19_budgetLeft_: int
                        d_19_budgetLeft_ = (maxSteps) - (d_1_steps_)
                        d_20_shouldForceClose_: bool
                        d_20_shouldForceClose_ = ((d_4_spanTokenCount_) >= (d_6_maxSpanTokens_)) or ((d_19_budgetLeft_) <= (d_5_closingReserve_))
                        if d_20_shouldForceClose_:
                            d_21_rg_: _dafny.Seq
                            d_22_rc_: _dafny.Seq
                            out10_: _dafny.Seq
                            out11_: _dafny.Seq
                            out10_, out11_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                            d_21_rg_ = out10_
                            d_22_rc_ = out11_
                            generated = d_21_rg_
                            currentConstrainedOut = d_22_rc_
                            if (parser).IsCompletePrefix(currentConstrainedOut):
                                d_23_cg2_: _dafny.Seq
                                d_24_ci2_: bool
                                d_25_cc2_: _dafny.Seq
                                out12_: _dafny.Seq
                                out13_: bool
                                out14_: _dafny.Seq
                                out12_, out13_, out14_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_23_cg2_ = out12_
                                d_24_ci2_ = out13_
                                d_25_cc2_ = out14_
                                generated = d_23_cg2_
                                insideConstrainedOut = d_24_ci2_
                                currentConstrainedOut = d_25_cc2_
                                d_1_steps_ = (d_1_steps_) + (1)
                                d_3_spanHasContent_ = False
                                d_4_spanTokenCount_ = 0
                            elif True:
                                raise _dafny.Break("0")
                        elif d_3_spanHasContent_:
                            d_26_cg_: _dafny.Seq
                            d_27_ci_: bool
                            d_28_cc_: _dafny.Seq
                            d_29_closed_: bool
                            out15_: _dafny.Seq
                            out16_: bool
                            out17_: _dafny.Seq
                            out18_: bool
                            out15_, out16_, out17_, out18_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                            d_26_cg_ = out15_
                            d_27_ci_ = out16_
                            d_28_cc_ = out17_
                            d_29_closed_ = out18_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if d_29_closed_:
                                generated = d_26_cg_
                                insideConstrainedOut = d_27_ci_
                                currentConstrainedOut = d_28_cc_
                                d_3_spanHasContent_ = False
                                d_4_spanTokenCount_ = 0
                            elif True:
                                d_30_constrainedPrompt_: _dafny.Seq
                                d_30_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_31_next_: _dafny.Seq
                                out19_: _dafny.Seq
                                out19_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_30_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_2_narrowThreshold_, eosToken)
                                d_31_next_ = out19_
                                if (d_31_next_) == (eosToken):
                                    d_32_rg2_: _dafny.Seq
                                    d_33_rc2_: _dafny.Seq
                                    out20_: _dafny.Seq
                                    out21_: _dafny.Seq
                                    out20_, out21_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                                    d_32_rg2_ = out20_
                                    d_33_rc2_ = out21_
                                    generated = d_32_rg2_
                                    currentConstrainedOut = d_33_rc2_
                                    if (parser).IsCompletePrefix(currentConstrainedOut):
                                        d_34_cg3_: _dafny.Seq
                                        d_35_ci3_: bool
                                        d_36_cc3_: _dafny.Seq
                                        out22_: _dafny.Seq
                                        out23_: bool
                                        out24_: _dafny.Seq
                                        out22_, out23_, out24_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                        d_34_cg3_ = out22_
                                        d_35_ci3_ = out23_
                                        d_36_cc3_ = out24_
                                        generated = d_34_cg3_
                                        insideConstrainedOut = d_35_ci3_
                                        currentConstrainedOut = d_36_cc3_
                                        d_4_spanTokenCount_ = 0
                                    raise _dafny.Break("0")
                                elif True:
                                    if not((parser).IsCompletePrefix(currentConstrainedOut)):
                                        d_37_ag_: _dafny.Seq
                                        d_38_ai_: bool
                                        d_39_ac_: _dafny.Seq
                                        out25_: _dafny.Seq
                                        out26_: bool
                                        out27_: _dafny.Seq
                                        out25_, out26_, out27_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_31_next_)
                                        d_37_ag_ = out25_
                                        d_38_ai_ = out26_
                                        d_39_ac_ = out27_
                                        generated = d_37_ag_
                                        insideConstrainedOut = d_38_ai_
                                        currentConstrainedOut = d_39_ac_
                                        d_3_spanHasContent_ = True
                                        d_4_spanTokenCount_ = (d_4_spanTokenCount_) + (1)
                        elif True:
                            d_40_constrainedPrompt_: _dafny.Seq
                            d_40_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_41_next_: _dafny.Seq
                            out28_: _dafny.Seq
                            out28_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_40_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_2_narrowThreshold_, eosToken)
                            d_41_next_ = out28_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_41_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                if not((parser).IsCompletePrefix(currentConstrainedOut)):
                                    d_42_ag_: _dafny.Seq
                                    d_43_ai_: bool
                                    d_44_ac_: _dafny.Seq
                                    out29_: _dafny.Seq
                                    out30_: bool
                                    out31_: _dafny.Seq
                                    out29_, out30_, out31_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_41_next_)
                                    d_42_ag_ = out29_
                                    d_43_ai_ = out30_
                                    d_44_ac_ = out31_
                                    generated = d_42_ag_
                                    insideConstrainedOut = d_43_ai_
                                    currentConstrainedOut = d_44_ac_
                                    d_3_spanHasContent_ = True
                                    d_4_spanTokenCount_ = (d_4_spanTokenCount_) + (1)
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

