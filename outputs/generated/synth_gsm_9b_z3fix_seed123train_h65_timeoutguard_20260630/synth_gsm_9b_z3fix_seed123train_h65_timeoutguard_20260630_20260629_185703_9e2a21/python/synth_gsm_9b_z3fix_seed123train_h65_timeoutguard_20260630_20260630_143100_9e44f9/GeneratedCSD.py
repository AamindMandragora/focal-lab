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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. Show each calculation inside <<expression>> delimiters. Use plain variable names, numbers, and +, -, *, /, //, %, int(). No ** or ^. Wrap ALL intermediate results and final answer in <<>>. Final answer must be inside the LAST <<expression>>.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_forceOpenThreshold_: int
        d_2_forceOpenThreshold_ = 80
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_3_remainingBudget_: int
                        d_3_remainingBudget_ = (maxSteps) - (d_1_steps_)
                        if (d_3_remainingBudget_) <= (2):
                            raise _dafny.Break("0")
                        elif (d_3_remainingBudget_) <= (d_2_forceOpenThreshold_):
                            d_4_og_: _dafny.Seq
                            d_5_oi_: bool
                            d_6_oc_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_4_og_ = out0_
                            d_5_oi_ = out1_
                            d_6_oc_ = out2_
                            generated = d_4_og_
                            insideConstrainedOut = d_5_oi_
                            currentConstrainedOut = d_6_oc_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_7_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_7_next_ = out3_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_7_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_7_next_]))
                                if (d_7_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    d_8_og2_: _dafny.Seq
                                    d_9_oi2_: bool
                                    d_10_oc2_: _dafny.Seq
                                    out4_: _dafny.Seq
                                    out5_: bool
                                    out6_: _dafny.Seq
                                    out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                    d_8_og2_ = out4_
                                    d_9_oi2_ = out5_
                                    d_10_oc2_ = out6_
                                    generated = d_8_og2_
                                    insideConstrainedOut = d_9_oi2_
                                    currentConstrainedOut = d_10_oc2_
                    elif True:
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
                        elif True:
                            d_15_constrainedPrompt_: _dafny.Seq
                            d_15_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_16_next_: _dafny.Seq
                            out11_: _dafny.Seq
                            out11_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_15_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('2e0'), eosToken)
                            d_16_next_ = out11_
                            if (d_16_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_17_valid_: bool
                                out12_: bool
                                out12_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_16_next_)
                                d_17_valid_ = out12_
                                if d_17_valid_:
                                    d_18_ag_: _dafny.Seq
                                    d_19_ai_: bool
                                    d_20_ac_: _dafny.Seq
                                    out13_: _dafny.Seq
                                    out14_: bool
                                    out15_: _dafny.Seq
                                    out13_, out14_, out15_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_16_next_)
                                    d_18_ag_ = out13_
                                    d_19_ai_ = out14_
                                    d_20_ac_ = out15_
                                    generated = d_18_ag_
                                    insideConstrainedOut = d_19_ai_
                                    currentConstrainedOut = d_20_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_21_remainingPost_: int
            d_21_remainingPost_ = (maxSteps) - (d_1_steps_)
            d_22_postBudget_: int
            if (d_21_remainingPost_) < (40):
                d_22_postBudget_ = d_21_remainingPost_
            elif True:
                d_22_postBudget_ = 40
            if (d_22_postBudget_) > (0):
                d_23_cgP_: _dafny.Seq
                d_24_ciP_: bool
                d_25_ccP_: _dafny.Seq
                out16_: _dafny.Seq
                out17_: bool
                out18_: _dafny.Seq
                out16_, out17_, out18_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_22_postBudget_)
                d_23_cgP_ = out16_
                d_24_ciP_ = out17_
                d_25_ccP_ = out18_
                generated = d_23_cgP_
                insideConstrainedOut = d_24_ciP_
                currentConstrainedOut = d_25_ccP_
                d_1_steps_ = (d_1_steps_) + (d_22_postBudget_)
        if (not(insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
            d_26_genStr_: _dafny.Seq
            d_26_genStr_ = VerifiedDecoderAgent.CSDHelpers.PrefixToString(generated)
            d_27_openCount_: int
            d_27_openCount_ = VerifiedDecoderAgent.CSDHelpers.CountSubstring(d_26_genStr_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))
            if (d_27_openCount_) == (0):
                d_28_remainingFinal_: int
                d_28_remainingFinal_ = (maxSteps) - (d_1_steps_)
                if (d_28_remainingFinal_) >= (5):
                    d_29_ogF_: _dafny.Seq
                    d_30_oiF_: bool
                    d_31_ocF_: _dafny.Seq
                    out19_: _dafny.Seq
                    out20_: bool
                    out21_: _dafny.Seq
                    out19_, out20_, out21_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                    d_29_ogF_ = out19_
                    d_30_oiF_ = out20_
                    d_31_ocF_ = out21_
                    generated = d_29_ogF_
                    insideConstrainedOut = d_30_oiF_
                    currentConstrainedOut = d_31_ocF_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if (d_1_steps_) < (maxSteps):
                        d_32_remainingF2_: int
                        d_32_remainingF2_ = (maxSteps) - (d_1_steps_)
                        d_33_closeBudgetF_: int
                        if (d_32_remainingF2_) < (60):
                            d_33_closeBudgetF_ = d_32_remainingF2_
                        elif True:
                            d_33_closeBudgetF_ = 60
                        if (d_33_closeBudgetF_) > (0):
                            d_34_cgF_: _dafny.Seq
                            d_35_ciF_: bool
                            d_36_ccF_: _dafny.Seq
                            out22_: _dafny.Seq
                            out23_: bool
                            out24_: _dafny.Seq
                            out22_, out23_, out24_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_33_closeBudgetF_)
                            d_34_cgF_ = out22_
                            d_35_ciF_ = out23_
                            d_36_ccF_ = out24_
                            generated = d_34_cgF_
                            insideConstrainedOut = d_35_ciF_
                            currentConstrainedOut = d_36_ccF_
                            d_1_steps_ = (d_1_steps_) + (d_33_closeBudgetF_)
        if (d_1_steps_) > (maxSteps):
            cost = maxSteps
        elif True:
            cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

