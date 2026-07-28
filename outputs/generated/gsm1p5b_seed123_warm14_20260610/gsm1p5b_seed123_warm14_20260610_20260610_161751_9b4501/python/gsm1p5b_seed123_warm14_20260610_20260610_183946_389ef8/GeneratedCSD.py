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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve this math word problem. Show brief reasoning, then write the final arithmetic expression inside << >> delimiters. Use only numbers, variable names, +, -, *, /, (, ) inside the delimiters.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_constrainedReserve_: int
        if (maxSteps) >= (60):
            d_2_constrainedReserve_ = 50
        elif (maxSteps) >= (20):
            d_2_constrainedReserve_ = _dafny.euclidian_division(maxSteps, 2)
        elif True:
            d_2_constrainedReserve_ = maxSteps
        d_3_freePhaseLimit_: int
        if (maxSteps) > (d_2_constrainedReserve_):
            d_3_freePhaseLimit_ = (maxSteps) - (d_2_constrainedReserve_)
        elif True:
            d_3_freePhaseLimit_ = 0
        d_4_effectiveFreeLimit_: int
        if (d_3_freePhaseLimit_) > (30):
            d_4_effectiveFreeLimit_ = 30
        elif True:
            d_4_effectiveFreeLimit_ = d_3_freePhaseLimit_
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if (d_1_steps_) >= (d_4_effectiveFreeLimit_):
                            d_5_og_: _dafny.Seq
                            d_6_oi_: bool
                            d_7_oc_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_5_og_ = out0_
                            d_6_oi_ = out1_
                            d_7_oc_ = out2_
                            generated = d_5_og_
                            insideConstrainedOut = d_6_oi_
                            currentConstrainedOut = d_7_oc_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_8_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_8_next_ = out3_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_8_next_) == (eosToken):
                                if ((d_1_steps_) + (2)) <= (maxSteps):
                                    d_9_og_: _dafny.Seq
                                    d_10_oi_: bool
                                    d_11_oc_: _dafny.Seq
                                    out4_: _dafny.Seq
                                    out5_: bool
                                    out6_: _dafny.Seq
                                    out4_, out5_, out6_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                    d_9_og_ = out4_
                                    d_10_oi_ = out5_
                                    d_11_oc_ = out6_
                                    generated = d_9_og_
                                    insideConstrainedOut = d_10_oi_
                                    currentConstrainedOut = d_11_oc_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                elif True:
                                    raise _dafny.Break("0")
                            elif (d_8_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_8_next_]))
                                d_12_eg_: _dafny.Seq
                                d_13_ei_: bool
                                d_14_ec_: _dafny.Seq
                                out7_: _dafny.Seq
                                out8_: bool
                                out9_: _dafny.Seq
                                out7_, out8_, out9_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_12_eg_ = out7_
                                d_13_ei_ = out8_
                                d_14_ec_ = out9_
                                generated = d_12_eg_
                                insideConstrainedOut = d_13_ei_
                                currentConstrainedOut = d_14_ec_
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_8_next_]))
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_15_cg_: _dafny.Seq
                        d_16_ci_: bool
                        d_17_cc_: _dafny.Seq
                        out10_: _dafny.Seq
                        out11_: bool
                        out12_: _dafny.Seq
                        out10_, out11_, out12_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_15_cg_ = out10_
                        d_16_ci_ = out11_
                        d_17_cc_ = out12_
                        generated = d_15_cg_
                        insideConstrainedOut = d_16_ci_
                        currentConstrainedOut = d_17_cc_
                        d_1_steps_ = (d_1_steps_) + (1)
                        raise _dafny.Break("0")
                    elif True:
                        d_18_constrainedPrompt_: _dafny.Seq
                        d_18_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_19_next_: _dafny.Seq
                        out13_: _dafny.Seq
                        out13_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_18_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                        d_19_next_ = out13_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_19_next_) == (eosToken):
                            d_20_rg_: _dafny.Seq
                            d_21_rc_: _dafny.Seq
                            out14_: _dafny.Seq
                            out15_: _dafny.Seq
                            out14_, out15_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                            d_20_rg_ = out14_
                            d_21_rc_ = out15_
                            generated = d_20_rg_
                            currentConstrainedOut = d_21_rc_
                            if ((parser).IsCompletePrefix(currentConstrainedOut)) and (((d_1_steps_) + (1)) <= (maxSteps)):
                                d_22_cg2_: _dafny.Seq
                                d_23_ci2_: bool
                                d_24_cc2_: _dafny.Seq
                                out16_: _dafny.Seq
                                out17_: bool
                                out18_: _dafny.Seq
                                out16_, out17_, out18_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_22_cg2_ = out16_
                                d_23_ci2_ = out17_
                                d_24_cc2_ = out18_
                                generated = d_22_cg2_
                                insideConstrainedOut = d_23_ci2_
                                currentConstrainedOut = d_24_cc2_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                insideConstrainedOut = False
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                            raise _dafny.Break("0")
                        elif True:
                            d_25_ag_: _dafny.Seq
                            d_26_ai_: bool
                            d_27_ac_: _dafny.Seq
                            out19_: _dafny.Seq
                            out20_: bool
                            out21_: _dafny.Seq
                            out19_, out20_, out21_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_19_next_)
                            d_25_ag_ = out19_
                            d_26_ai_ = out20_
                            d_27_ac_ = out21_
                            generated = d_25_ag_
                            insideConstrainedOut = d_26_ai_
                            currentConstrainedOut = d_27_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            if (parser).IsCompletePrefix(currentConstrainedOut):
                d_28_cg_: _dafny.Seq
                d_29_ci_: bool
                d_30_cc_: _dafny.Seq
                out22_: _dafny.Seq
                out23_: bool
                out24_: _dafny.Seq
                out22_, out23_, out24_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_28_cg_ = out22_
                d_29_ci_ = out23_
                d_30_cc_ = out24_
                generated = d_28_cg_
                insideConstrainedOut = d_29_ci_
                currentConstrainedOut = d_30_cc_
                d_1_steps_ = (d_1_steps_) + (1)
            elif True:
                d_31_rg_: _dafny.Seq
                d_32_rc_: _dafny.Seq
                out25_: _dafny.Seq
                out26_: _dafny.Seq
                out25_, out26_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                d_31_rg_ = out25_
                d_32_rc_ = out26_
                generated = d_31_rg_
                currentConstrainedOut = d_32_rc_
                if ((parser).IsCompletePrefix(currentConstrainedOut)) and (((d_1_steps_) + (1)) <= (maxSteps)):
                    d_33_cg2_: _dafny.Seq
                    d_34_ci2_: bool
                    d_35_cc2_: _dafny.Seq
                    out27_: _dafny.Seq
                    out28_: bool
                    out29_: _dafny.Seq
                    out27_, out28_, out29_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                    d_33_cg2_ = out27_
                    d_34_ci2_ = out28_
                    d_35_cc2_ = out29_
                    generated = d_33_cg2_
                    insideConstrainedOut = d_34_ci2_
                    currentConstrainedOut = d_35_cc2_
                    d_1_steps_ = (d_1_steps_) + (1)
                elif True:
                    insideConstrainedOut = False
                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

