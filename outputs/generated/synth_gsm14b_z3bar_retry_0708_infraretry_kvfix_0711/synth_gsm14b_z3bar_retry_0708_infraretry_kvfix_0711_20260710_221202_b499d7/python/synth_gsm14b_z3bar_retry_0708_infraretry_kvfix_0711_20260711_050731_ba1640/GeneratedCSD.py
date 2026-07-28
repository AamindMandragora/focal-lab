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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve this math problem with the exact given numbers. Show step by step arithmetic. Write the final computed numeric answer (just a number) inside << >> at the very end, like <<42>> or <<3.5>>."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_narrowThreshold_: int
        d_3_narrowThreshold_ = 12
        d_4_freeStepsTarget_: int
        d_4_freeStepsTarget_ = _dafny.euclidian_division((maxSteps) * (7), 10)
        d_5_forcedFinalSpan_: bool
        d_5_forcedFinalSpan_ = False
        d_6_eosSeenBeforeSpan_: bool
        d_6_eosSeenBeforeSpan_ = False
        d_7_anySpanOpened_: bool
        d_7_anySpanOpened_ = insideConstrained
        with _dafny.label("0"):
            while (d_2_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_8_remainingSteps_: int
                        d_8_remainingSteps_ = (maxSteps) - (d_2_steps_)
                        d_9_shouldForce_: bool
                        d_9_shouldForce_ = (not(d_5_forcedFinalSpan_)) and ((((d_2_steps_) >= (d_4_freeStepsTarget_)) or (((d_8_remainingSteps_) <= (5)) and (not(d_7_anySpanOpened_)))) or (d_6_eosSeenBeforeSpan_))
                        if (d_9_shouldForce_) and ((d_8_remainingSteps_) >= (2)):
                            d_10_og_: _dafny.Seq
                            d_11_oi_: bool
                            d_12_oc_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_10_og_ = out0_
                            d_11_oi_ = out1_
                            d_12_oc_ = out2_
                            generated = d_10_og_
                            insideConstrainedOut = d_11_oi_
                            currentConstrainedOut = d_12_oc_
                            d_2_steps_ = (d_2_steps_) + (1)
                            d_5_forcedFinalSpan_ = True
                            d_7_anySpanOpened_ = True
                            d_6_eosSeenBeforeSpan_ = False
                        elif (d_9_shouldForce_) and ((d_8_remainingSteps_) < (2)):
                            raise _dafny.Break("0")
                        elif True:
                            d_13_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_13_next_ = out3_
                            d_2_steps_ = (d_2_steps_) + (1)
                            if (d_13_next_) == (eosToken):
                                if not(d_7_anySpanOpened_):
                                    d_6_eosSeenBeforeSpan_ = True
                                elif True:
                                    raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_13_next_]))
                                if (d_13_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    out4_: _dafny.Seq
                                    out5_: bool
                                    out6_: _dafny.Seq
                                    out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                    generated = out4_
                                    insideConstrainedOut = out5_
                                    currentConstrainedOut = out6_
                                    d_7_anySpanOpened_ = True
                                    d_6_eosSeenBeforeSpan_ = False
                    elif True:
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            d_14_cg_: _dafny.Seq
                            d_15_ci_: bool
                            d_16_cc_: _dafny.Seq
                            out7_: _dafny.Seq
                            out8_: bool
                            out9_: _dafny.Seq
                            out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_14_cg_ = out7_
                            d_15_ci_ = out8_
                            d_16_cc_ = out9_
                            generated = d_14_cg_
                            insideConstrainedOut = d_15_ci_
                            currentConstrainedOut = d_16_cc_
                            d_2_steps_ = (d_2_steps_) + (1)
                        elif ((maxSteps) - (d_2_steps_)) <= (4):
                            d_17_closeBudget_: int
                            d_17_closeBudget_ = (maxSteps) - (d_2_steps_)
                            d_18_cg_: _dafny.Seq
                            d_19_ci_: bool
                            d_20_cc_: _dafny.Seq
                            out10_: _dafny.Seq
                            out11_: bool
                            out12_: _dafny.Seq
                            out10_, out11_, out12_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_17_closeBudget_)
                            d_18_cg_ = out10_
                            d_19_ci_ = out11_
                            d_20_cc_ = out12_
                            generated = d_18_cg_
                            insideConstrainedOut = d_19_ci_
                            currentConstrainedOut = d_20_cc_
                            d_2_steps_ = maxSteps
                        elif True:
                            d_21_constrainedPrompt_: _dafny.Seq
                            d_21_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_22_next_: _dafny.Seq
                            out13_: _dafny.Seq
                            out13_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_21_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_3_narrowThreshold_, eosToken)
                            d_22_next_ = out13_
                            d_2_steps_ = (d_2_steps_) + (1)
                            if (d_22_next_) == (eosToken):
                                d_23_remainingAfterEos_: int
                                d_23_remainingAfterEos_ = (maxSteps) - (d_2_steps_)
                                if (d_23_remainingAfterEos_) >= (1):
                                    d_24_closeBudget_: int
                                    d_24_closeBudget_ = d_23_remainingAfterEos_
                                    d_25_cg_: _dafny.Seq
                                    d_26_ci_: bool
                                    d_27_cc_: _dafny.Seq
                                    out14_: _dafny.Seq
                                    out15_: bool
                                    out16_: _dafny.Seq
                                    out14_, out15_, out16_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_24_closeBudget_)
                                    d_25_cg_ = out14_
                                    d_26_ci_ = out15_
                                    d_27_cc_ = out16_
                                    generated = d_25_cg_
                                    insideConstrainedOut = d_26_ci_
                                    currentConstrainedOut = d_27_cc_
                                    d_2_steps_ = maxSteps
                                raise _dafny.Break("0")
                            elif True:
                                d_28_ag_: _dafny.Seq
                                d_29_ai_: bool
                                d_30_ac_: _dafny.Seq
                                out17_: _dafny.Seq
                                out18_: bool
                                out19_: _dafny.Seq
                                out17_, out18_, out19_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_22_next_)
                                d_28_ag_ = out17_
                                d_29_ai_ = out18_
                                d_30_ac_ = out19_
                                generated = d_28_ag_
                                insideConstrainedOut = d_29_ai_
                                currentConstrainedOut = d_30_ac_
                    pass
            pass
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

