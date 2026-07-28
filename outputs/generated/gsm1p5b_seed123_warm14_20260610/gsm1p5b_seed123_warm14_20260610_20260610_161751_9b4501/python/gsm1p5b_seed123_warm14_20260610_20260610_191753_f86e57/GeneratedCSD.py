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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve this math word problem step by step. At the very end, write ONLY the final arithmetic expression inside << >> delimiters. Use ONLY: numbers, variable names (letters/underscores), +, -, *, /, (, ). Do NOT use ^, **, int(), //, {}, [] inside the delimiters.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_constrainedReserve_: int
        if (maxSteps) >= (120):
            d_2_constrainedReserve_ = 120
        elif (maxSteps) >= (10):
            d_2_constrainedReserve_ = _dafny.euclidian_division(maxSteps, 2)
        elif True:
            d_2_constrainedReserve_ = maxSteps
        d_3_freePhaseLimit_: int
        if (maxSteps) > (d_2_constrainedReserve_):
            d_3_freePhaseLimit_ = (maxSteps) - (d_2_constrainedReserve_)
        elif True:
            d_3_freePhaseLimit_ = 0
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if (d_1_steps_) >= (d_3_freePhaseLimit_):
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
                                if ((d_1_steps_) + (3)) <= (maxSteps):
                                    d_8_og_: _dafny.Seq
                                    d_9_oi_: bool
                                    d_10_oc_: _dafny.Seq
                                    out4_: _dafny.Seq
                                    out5_: bool
                                    out6_: _dafny.Seq
                                    out4_, out5_, out6_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                    d_8_og_ = out4_
                                    d_9_oi_ = out5_
                                    d_10_oc_ = out6_
                                    generated = d_8_og_
                                    insideConstrainedOut = d_9_oi_
                                    currentConstrainedOut = d_10_oc_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                elif True:
                                    raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_7_next_]))
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_11_cg_: _dafny.Seq
                        d_12_ci_: bool
                        d_13_cc_: _dafny.Seq
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: _dafny.Seq
                        out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_11_cg_ = out7_
                        d_12_ci_ = out8_
                        d_13_cc_ = out9_
                        generated = d_11_cg_
                        insideConstrainedOut = d_12_ci_
                        currentConstrainedOut = d_13_cc_
                        d_1_steps_ = (d_1_steps_) + (1)
                        raise _dafny.Break("0")
                    elif True:
                        d_14_deadEnd_: bool
                        out10_: bool
                        out10_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                        d_14_deadEnd_ = out10_
                        if d_14_deadEnd_:
                            d_15_remainingSteps_: int
                            d_15_remainingSteps_ = (maxSteps) - (d_1_steps_)
                            if (d_15_remainingSteps_) >= (3):
                                d_16_constrainedPrompt_: _dafny.Seq
                                d_16_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_17_rg_: _dafny.Seq
                                d_18_rc_: _dafny.Seq
                                out11_: _dafny.Seq
                                out12_: _dafny.Seq
                                out11_, out12_ = (d_0_helpers_).RollbackAndContinue(lm, parser, d_16_constrainedPrompt_, generated, currentConstrainedOut, eosToken, d_15_remainingSteps_, 1, 3)
                                d_17_rg_ = out11_
                                d_18_rc_ = out12_
                                d_19_stepsBefore_: int
                                d_19_stepsBefore_ = d_1_steps_
                                d_1_steps_ = (maxSteps) - (1)
                                generated = d_17_rg_
                                currentConstrainedOut = d_18_rc_
                                insideConstrainedOut = True
                                if ((parser).IsCompletePrefix(currentConstrainedOut)) and (((d_1_steps_) + (1)) <= (maxSteps)):
                                    d_20_cg2_: _dafny.Seq
                                    d_21_ci2_: bool
                                    d_22_cc2_: _dafny.Seq
                                    out13_: _dafny.Seq
                                    out14_: bool
                                    out15_: _dafny.Seq
                                    out13_, out14_, out15_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_20_cg2_ = out13_
                                    d_21_ci2_ = out14_
                                    d_22_cc2_ = out15_
                                    generated = d_20_cg2_
                                    insideConstrainedOut = d_21_ci2_
                                    currentConstrainedOut = d_22_cc2_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                elif not((parser).IsCompletePrefix(currentConstrainedOut)):
                                    insideConstrainedOut = False
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                            elif True:
                                d_23_rg_: _dafny.Seq
                                d_24_rc_: _dafny.Seq
                                out16_: _dafny.Seq
                                out17_: _dafny.Seq
                                out16_, out17_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                                d_23_rg_ = out16_
                                d_24_rc_ = out17_
                                generated = d_23_rg_
                                currentConstrainedOut = d_24_rc_
                                if ((parser).IsCompletePrefix(currentConstrainedOut)) and (((d_1_steps_) + (1)) <= (maxSteps)):
                                    d_25_cg2_: _dafny.Seq
                                    d_26_ci2_: bool
                                    d_27_cc2_: _dafny.Seq
                                    out18_: _dafny.Seq
                                    out19_: bool
                                    out20_: _dafny.Seq
                                    out18_, out19_, out20_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_25_cg2_ = out18_
                                    d_26_ci2_ = out19_
                                    d_27_cc2_ = out20_
                                    generated = d_25_cg2_
                                    insideConstrainedOut = d_26_ci2_
                                    currentConstrainedOut = d_27_cc2_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                elif True:
                                    insideConstrainedOut = False
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                            raise _dafny.Break("0")
                        elif True:
                            d_28_constrainedPrompt_: _dafny.Seq
                            d_28_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_29_next_: _dafny.Seq
                            out21_: _dafny.Seq
                            out21_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_28_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                            d_29_next_ = out21_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_29_next_) == (eosToken):
                                d_30_rg_: _dafny.Seq
                                d_31_rc_: _dafny.Seq
                                out22_: _dafny.Seq
                                out23_: _dafny.Seq
                                out22_, out23_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                                d_30_rg_ = out22_
                                d_31_rc_ = out23_
                                generated = d_30_rg_
                                currentConstrainedOut = d_31_rc_
                                if ((parser).IsCompletePrefix(currentConstrainedOut)) and (((d_1_steps_) + (1)) <= (maxSteps)):
                                    d_32_cg2_: _dafny.Seq
                                    d_33_ci2_: bool
                                    d_34_cc2_: _dafny.Seq
                                    out24_: _dafny.Seq
                                    out25_: bool
                                    out26_: _dafny.Seq
                                    out24_, out25_, out26_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_32_cg2_ = out24_
                                    d_33_ci2_ = out25_
                                    d_34_cc2_ = out26_
                                    generated = d_32_cg2_
                                    insideConstrainedOut = d_33_ci2_
                                    currentConstrainedOut = d_34_cc2_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                elif True:
                                    insideConstrainedOut = False
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                raise _dafny.Break("0")
                            elif True:
                                d_35_ag_: _dafny.Seq
                                d_36_ai_: bool
                                d_37_ac_: _dafny.Seq
                                out27_: _dafny.Seq
                                out28_: bool
                                out29_: _dafny.Seq
                                out27_, out28_, out29_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_29_next_)
                                d_35_ag_ = out27_
                                d_36_ai_ = out28_
                                d_37_ac_ = out29_
                                generated = d_35_ag_
                                insideConstrainedOut = d_36_ai_
                                currentConstrainedOut = d_37_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            if (parser).IsCompletePrefix(currentConstrainedOut):
                d_38_cg_: _dafny.Seq
                d_39_ci_: bool
                d_40_cc_: _dafny.Seq
                out30_: _dafny.Seq
                out31_: bool
                out32_: _dafny.Seq
                out30_, out31_, out32_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_38_cg_ = out30_
                d_39_ci_ = out31_
                d_40_cc_ = out32_
                generated = d_38_cg_
                insideConstrainedOut = d_39_ci_
                currentConstrainedOut = d_40_cc_
                d_1_steps_ = (d_1_steps_) + (1)
            elif True:
                d_41_rg_: _dafny.Seq
                d_42_rc_: _dafny.Seq
                out33_: _dafny.Seq
                out34_: _dafny.Seq
                out33_, out34_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                d_41_rg_ = out33_
                d_42_rc_ = out34_
                generated = d_41_rg_
                currentConstrainedOut = d_42_rc_
                if ((parser).IsCompletePrefix(currentConstrainedOut)) and (((d_1_steps_) + (1)) <= (maxSteps)):
                    d_43_cg2_: _dafny.Seq
                    d_44_ci2_: bool
                    d_45_cc2_: _dafny.Seq
                    out35_: _dafny.Seq
                    out36_: bool
                    out37_: _dafny.Seq
                    out35_, out36_, out37_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                    d_43_cg2_ = out35_
                    d_44_ci2_ = out36_
                    d_45_cc2_ = out37_
                    generated = d_43_cg2_
                    insideConstrainedOut = d_44_ci2_
                    currentConstrainedOut = d_45_cc2_
                    d_1_steps_ = (d_1_steps_) + (1)
                elif True:
                    insideConstrainedOut = False
                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

