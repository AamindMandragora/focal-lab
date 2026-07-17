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
        d_1_guidance_: _dafny.Seq
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math problem step by step. Compute with the actual numbers and variable names given. At the end, write your final answer inside << >> (e.g. <<42>> or <<n * count>>)."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_narrowThreshold_: int
        d_3_narrowThreshold_ = 12
        d_4_freeStepsTarget_: int
        d_4_freeStepsTarget_ = _dafny.euclidian_division((maxSteps) * (7), 10)
        d_5_forcedFinalSpan_: bool
        d_5_forcedFinalSpan_ = False
        d_6_anySpanClosed_: bool
        d_6_anySpanClosed_ = False
        d_7_lastClosedSpanLength_: int
        d_7_lastClosedSpanLength_ = 0
        d_8_spanLengthThreshold_: int
        d_8_spanLengthThreshold_ = 3
        with _dafny.label("0"):
            while (d_2_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_9_budgetLeft_: int
                        d_9_budgetLeft_ = (maxSteps) - (d_2_steps_)
                        d_10_shouldForce_: bool
                        d_10_shouldForce_ = (not(d_5_forcedFinalSpan_)) and (((d_2_steps_) >= (d_4_freeStepsTarget_)) or ((d_9_budgetLeft_) <= (5)))
                        if (d_10_shouldForce_) and ((d_9_budgetLeft_) >= (3)):
                            d_11_og_: _dafny.Seq
                            d_12_oi_: bool
                            d_13_oc_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_11_og_ = out0_
                            d_12_oi_ = out1_
                            d_13_oc_ = out2_
                            generated = d_11_og_
                            insideConstrainedOut = d_12_oi_
                            currentConstrainedOut = d_13_oc_
                            d_2_steps_ = (d_2_steps_) + (1)
                            d_5_forcedFinalSpan_ = True
                        elif d_10_shouldForce_:
                            raise _dafny.Break("0")
                        elif True:
                            d_14_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_14_next_ = out3_
                            d_2_steps_ = (d_2_steps_) + (1)
                            if (d_14_next_) == (eosToken):
                                if (not(d_5_forcedFinalSpan_)) and ((d_9_budgetLeft_) >= (3)):
                                    d_15_remainBudget_: int
                                    d_15_remainBudget_ = (d_9_budgetLeft_) - (1)
                                    if (d_15_remainBudget_) >= (2):
                                        d_16_og_: _dafny.Seq
                                        d_17_oi_: bool
                                        d_18_oc_: _dafny.Seq
                                        out4_: _dafny.Seq
                                        out5_: bool
                                        out6_: _dafny.Seq
                                        out4_, out5_, out6_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                        d_16_og_ = out4_
                                        d_17_oi_ = out5_
                                        d_18_oc_ = out6_
                                        generated = d_16_og_
                                        insideConstrainedOut = d_17_oi_
                                        currentConstrainedOut = d_18_oc_
                                        d_2_steps_ = (d_2_steps_) + (1)
                                        d_5_forcedFinalSpan_ = True
                                    elif True:
                                        raise _dafny.Break("0")
                                elif True:
                                    raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_14_next_]))
                                if (d_14_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    if (d_6_anySpanClosed_) and ((d_7_lastClosedSpanLength_) >= (d_8_spanLengthThreshold_)):
                                        out7_: _dafny.Seq
                                        out8_: bool
                                        out9_: _dafny.Seq
                                        out7_, out8_, out9_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                        generated = out7_
                                        insideConstrainedOut = out8_
                                        currentConstrainedOut = out9_
                                    elif True:
                                        out10_: _dafny.Seq
                                        out11_: bool
                                        out12_: _dafny.Seq
                                        out10_, out11_, out12_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                        generated = out10_
                                        insideConstrainedOut = out11_
                                        currentConstrainedOut = out12_
                    elif True:
                        d_19_budgetLeft2_: int
                        d_19_budgetLeft2_ = (maxSteps) - (d_2_steps_)
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            if (d_19_budgetLeft2_) >= (1):
                                d_20_closedSpanLength_: int
                                d_20_closedSpanLength_ = len(currentConstrainedOut)
                                d_21_cg_: _dafny.Seq
                                d_22_ci_: bool
                                d_23_cc_: _dafny.Seq
                                out13_: _dafny.Seq
                                out14_: bool
                                out15_: _dafny.Seq
                                out13_, out14_, out15_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_21_cg_ = out13_
                                d_22_ci_ = out14_
                                d_23_cc_ = out15_
                                generated = d_21_cg_
                                insideConstrainedOut = d_22_ci_
                                currentConstrainedOut = d_23_cc_
                                d_2_steps_ = (d_2_steps_) + (1)
                                d_6_anySpanClosed_ = True
                                d_7_lastClosedSpanLength_ = d_20_closedSpanLength_
                                if d_5_forcedFinalSpan_:
                                    raise _dafny.Break("0")
                            elif True:
                                raise _dafny.Break("0")
                        elif (d_19_budgetLeft2_) <= (4):
                            d_24_closeBudget_: int
                            d_24_closeBudget_ = d_19_budgetLeft2_
                            d_25_cg_: _dafny.Seq
                            d_26_ci_: bool
                            d_27_cc_: _dafny.Seq
                            out16_: _dafny.Seq
                            out17_: bool
                            out18_: _dafny.Seq
                            out16_, out17_, out18_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_24_closeBudget_)
                            d_25_cg_ = out16_
                            d_26_ci_ = out17_
                            d_27_cc_ = out18_
                            generated = d_25_cg_
                            insideConstrainedOut = d_26_ci_
                            currentConstrainedOut = d_27_cc_
                            d_2_steps_ = maxSteps
                            raise _dafny.Break("0")
                        elif True:
                            d_28_constrainedPrompt_: _dafny.Seq
                            d_28_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_29_next_: _dafny.Seq
                            out19_: _dafny.Seq
                            out19_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_28_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_3_narrowThreshold_, eosToken)
                            d_29_next_ = out19_
                            d_2_steps_ = (d_2_steps_) + (1)
                            if (d_29_next_) == (eosToken):
                                if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_2_steps_) < (maxSteps)):
                                    d_30_closedSpanLength_: int
                                    d_30_closedSpanLength_ = len(currentConstrainedOut)
                                    d_31_cg_: _dafny.Seq
                                    d_32_ci_: bool
                                    d_33_cc_: _dafny.Seq
                                    out20_: _dafny.Seq
                                    out21_: bool
                                    out22_: _dafny.Seq
                                    out20_, out21_, out22_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_31_cg_ = out20_
                                    d_32_ci_ = out21_
                                    d_33_cc_ = out22_
                                    generated = d_31_cg_
                                    insideConstrainedOut = d_32_ci_
                                    currentConstrainedOut = d_33_cc_
                                    d_2_steps_ = (d_2_steps_) + (1)
                                    d_6_anySpanClosed_ = True
                                    d_7_lastClosedSpanLength_ = d_30_closedSpanLength_
                                elif (d_2_steps_) < (maxSteps):
                                    d_34_closeBudget2_: int
                                    d_34_closeBudget2_ = (maxSteps) - (d_2_steps_)
                                    d_35_cg_: _dafny.Seq
                                    d_36_ci_: bool
                                    d_37_cc_: _dafny.Seq
                                    out23_: _dafny.Seq
                                    out24_: bool
                                    out25_: _dafny.Seq
                                    out23_, out24_, out25_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_34_closeBudget2_)
                                    d_35_cg_ = out23_
                                    d_36_ci_ = out24_
                                    d_37_cc_ = out25_
                                    generated = d_35_cg_
                                    insideConstrainedOut = d_36_ci_
                                    currentConstrainedOut = d_37_cc_
                                    d_2_steps_ = maxSteps
                                raise _dafny.Break("0")
                            elif True:
                                d_38_ag_: _dafny.Seq
                                d_39_ai_: bool
                                d_40_ac_: _dafny.Seq
                                out26_: _dafny.Seq
                                out27_: bool
                                out28_: _dafny.Seq
                                out26_, out27_, out28_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_29_next_)
                                d_38_ag_ = out26_
                                d_39_ai_ = out27_
                                d_40_ac_ = out28_
                                generated = d_38_ag_
                                insideConstrainedOut = d_39_ai_
                                currentConstrainedOut = d_40_ac_
                    pass
            pass
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

