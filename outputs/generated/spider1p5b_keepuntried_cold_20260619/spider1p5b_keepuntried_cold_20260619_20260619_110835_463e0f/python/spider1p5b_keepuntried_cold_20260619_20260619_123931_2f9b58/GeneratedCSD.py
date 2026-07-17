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
        d_5_closingReserve_ = 8
        d_6_maxSpanTokens_: int
        d_6_maxSpanTokens_ = 150
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_7_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_7_next_ = out0_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_7_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_7_next_]))
                            if (d_7_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                d_8_eg_: _dafny.Seq
                                d_9_ei_: bool
                                d_10_ec_: _dafny.Seq
                                out1_: _dafny.Seq
                                out2_: bool
                                out3_: _dafny.Seq
                                out1_, out2_, out3_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_8_eg_ = out1_
                                d_9_ei_ = out2_
                                d_10_ec_ = out3_
                                generated = d_8_eg_
                                insideConstrainedOut = d_9_ei_
                                currentConstrainedOut = d_10_ec_
                                d_3_spanHasContent_ = False
                                d_4_spanTokenCount_ = 0
                    elif True:
                        d_11_budgetLeft_: int
                        d_11_budgetLeft_ = (maxSteps) - (d_1_steps_)
                        d_12_shouldForceClose_: bool
                        d_12_shouldForceClose_ = ((d_4_spanTokenCount_) >= (d_6_maxSpanTokens_)) or ((d_11_budgetLeft_) <= (d_5_closingReserve_))
                        if d_12_shouldForceClose_:
                            d_13_rg_: _dafny.Seq
                            d_14_rc_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: _dafny.Seq
                            out4_, out5_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                            d_13_rg_ = out4_
                            d_14_rc_ = out5_
                            generated = d_13_rg_
                            currentConstrainedOut = d_14_rc_
                            if (parser).IsCompletePrefix(currentConstrainedOut):
                                d_15_cg2_: _dafny.Seq
                                d_16_ci2_: bool
                                d_17_cc2_: _dafny.Seq
                                out6_: _dafny.Seq
                                out7_: bool
                                out8_: _dafny.Seq
                                out6_, out7_, out8_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_15_cg2_ = out6_
                                d_16_ci2_ = out7_
                                d_17_cc2_ = out8_
                                generated = d_15_cg2_
                                insideConstrainedOut = d_16_ci2_
                                currentConstrainedOut = d_17_cc2_
                                d_1_steps_ = (d_1_steps_) + (1)
                                d_3_spanHasContent_ = False
                                d_4_spanTokenCount_ = 0
                            elif True:
                                raise _dafny.Break("0")
                        elif d_3_spanHasContent_:
                            d_18_cg_: _dafny.Seq
                            d_19_ci_: bool
                            d_20_cc_: _dafny.Seq
                            d_21_closed_: bool
                            out9_: _dafny.Seq
                            out10_: bool
                            out11_: _dafny.Seq
                            out12_: bool
                            out9_, out10_, out11_, out12_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                            d_18_cg_ = out9_
                            d_19_ci_ = out10_
                            d_20_cc_ = out11_
                            d_21_closed_ = out12_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if d_21_closed_:
                                generated = d_18_cg_
                                insideConstrainedOut = d_19_ci_
                                currentConstrainedOut = d_20_cc_
                                d_3_spanHasContent_ = False
                                d_4_spanTokenCount_ = 0
                            elif True:
                                d_22_constrainedPrompt_: _dafny.Seq
                                d_22_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_23_next_: _dafny.Seq
                                out13_: _dafny.Seq
                                out13_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_22_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_2_narrowThreshold_, eosToken)
                                d_23_next_ = out13_
                                if (d_23_next_) == (eosToken):
                                    d_24_rg2_: _dafny.Seq
                                    d_25_rc2_: _dafny.Seq
                                    out14_: _dafny.Seq
                                    out15_: _dafny.Seq
                                    out14_, out15_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                                    d_24_rg2_ = out14_
                                    d_25_rc2_ = out15_
                                    generated = d_24_rg2_
                                    currentConstrainedOut = d_25_rc2_
                                    if (parser).IsCompletePrefix(currentConstrainedOut):
                                        d_26_cg3_: _dafny.Seq
                                        d_27_ci3_: bool
                                        d_28_cc3_: _dafny.Seq
                                        out16_: _dafny.Seq
                                        out17_: bool
                                        out18_: _dafny.Seq
                                        out16_, out17_, out18_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                        d_26_cg3_ = out16_
                                        d_27_ci3_ = out17_
                                        d_28_cc3_ = out18_
                                        generated = d_26_cg3_
                                        insideConstrainedOut = d_27_ci3_
                                        currentConstrainedOut = d_28_cc3_
                                        d_4_spanTokenCount_ = 0
                                    raise _dafny.Break("0")
                                elif True:
                                    if not((parser).IsCompletePrefix(currentConstrainedOut)):
                                        d_29_ag_: _dafny.Seq
                                        d_30_ai_: bool
                                        d_31_ac_: _dafny.Seq
                                        out19_: _dafny.Seq
                                        out20_: bool
                                        out21_: _dafny.Seq
                                        out19_, out20_, out21_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_23_next_)
                                        d_29_ag_ = out19_
                                        d_30_ai_ = out20_
                                        d_31_ac_ = out21_
                                        generated = d_29_ag_
                                        insideConstrainedOut = d_30_ai_
                                        currentConstrainedOut = d_31_ac_
                                        d_3_spanHasContent_ = True
                                        d_4_spanTokenCount_ = (d_4_spanTokenCount_) + (1)
                        elif True:
                            d_32_constrainedPrompt_: _dafny.Seq
                            d_32_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_33_next_: _dafny.Seq
                            out22_: _dafny.Seq
                            out22_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_32_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_2_narrowThreshold_, eosToken)
                            d_33_next_ = out22_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_33_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                if not((parser).IsCompletePrefix(currentConstrainedOut)):
                                    d_34_ag_: _dafny.Seq
                                    d_35_ai_: bool
                                    d_36_ac_: _dafny.Seq
                                    out23_: _dafny.Seq
                                    out24_: bool
                                    out25_: _dafny.Seq
                                    out23_, out24_, out25_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_33_next_)
                                    d_34_ag_ = out23_
                                    d_35_ai_ = out24_
                                    d_36_ac_ = out25_
                                    generated = d_34_ag_
                                    insideConstrainedOut = d_35_ai_
                                    currentConstrainedOut = d_36_ac_
                                    d_3_spanHasContent_ = True
                                    d_4_spanTokenCount_ = (d_4_spanTokenCount_) + (1)
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

