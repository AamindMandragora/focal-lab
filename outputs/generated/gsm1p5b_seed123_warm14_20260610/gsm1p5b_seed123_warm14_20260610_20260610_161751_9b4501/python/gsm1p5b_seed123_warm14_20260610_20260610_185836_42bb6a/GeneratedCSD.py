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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve this math word problem step by step. At the end, write the final arithmetic expression inside << >> delimiters. Use only numbers, variable names, +, -, *, /, (, ) inside the delimiters. Keep the expression concise.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_constrainedReserve_: int
        if (maxSteps) >= (100):
            d_2_constrainedReserve_ = 100
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
        if (d_3_freePhaseLimit_) > (200):
            d_4_effectiveFreeLimit_ = 200
        elif True:
            d_4_effectiveFreeLimit_ = d_3_freePhaseLimit_
        d_5_constrainedSpanStart_: int
        d_5_constrainedSpanStart_ = 0
        d_6_maxConstrainedSpanSteps_: int
        d_6_maxConstrainedSpanSteps_ = 60
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if (d_1_steps_) >= (d_4_effectiveFreeLimit_):
                            d_7_og_: _dafny.Seq
                            d_8_oi_: bool
                            d_9_oc_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_7_og_ = out0_
                            d_8_oi_ = out1_
                            d_9_oc_ = out2_
                            generated = d_7_og_
                            insideConstrainedOut = d_8_oi_
                            currentConstrainedOut = d_9_oc_
                            d_5_constrainedSpanStart_ = d_1_steps_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_10_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_10_next_ = out3_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_10_next_) == (eosToken):
                                if ((d_1_steps_) + (2)) <= (maxSteps):
                                    d_11_og_: _dafny.Seq
                                    d_12_oi_: bool
                                    d_13_oc_: _dafny.Seq
                                    out4_: _dafny.Seq
                                    out5_: bool
                                    out6_: _dafny.Seq
                                    out4_, out5_, out6_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                    d_11_og_ = out4_
                                    d_12_oi_ = out5_
                                    d_13_oc_ = out6_
                                    generated = d_11_og_
                                    insideConstrainedOut = d_12_oi_
                                    currentConstrainedOut = d_13_oc_
                                    d_5_constrainedSpanStart_ = d_1_steps_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                elif True:
                                    raise _dafny.Break("0")
                            elif (d_10_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_10_next_]))
                                d_14_eg_: _dafny.Seq
                                d_15_ei_: bool
                                d_16_ec_: _dafny.Seq
                                out7_: _dafny.Seq
                                out8_: bool
                                out9_: _dafny.Seq
                                out7_, out8_, out9_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_14_eg_ = out7_
                                d_15_ei_ = out8_
                                d_16_ec_ = out9_
                                generated = d_14_eg_
                                insideConstrainedOut = d_15_ei_
                                currentConstrainedOut = d_16_ec_
                                d_5_constrainedSpanStart_ = d_1_steps_
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_10_next_]))
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_17_cg_: _dafny.Seq
                        d_18_ci_: bool
                        d_19_cc_: _dafny.Seq
                        out10_: _dafny.Seq
                        out11_: bool
                        out12_: _dafny.Seq
                        out10_, out11_, out12_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_17_cg_ = out10_
                        d_18_ci_ = out11_
                        d_19_cc_ = out12_
                        generated = d_17_cg_
                        insideConstrainedOut = d_18_ci_
                        currentConstrainedOut = d_19_cc_
                        d_1_steps_ = (d_1_steps_) + (1)
                        raise _dafny.Break("0")
                    elif ((d_1_steps_) - (d_5_constrainedSpanStart_)) >= (d_6_maxConstrainedSpanSteps_):
                        d_20_rg_: _dafny.Seq
                        d_21_rc_: _dafny.Seq
                        out13_: _dafny.Seq
                        out14_: _dafny.Seq
                        out13_, out14_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                        d_20_rg_ = out13_
                        d_21_rc_ = out14_
                        generated = d_20_rg_
                        currentConstrainedOut = d_21_rc_
                        if ((parser).IsCompletePrefix(currentConstrainedOut)) and (((d_1_steps_) + (1)) <= (maxSteps)):
                            d_22_cg2_: _dafny.Seq
                            d_23_ci2_: bool
                            d_24_cc2_: _dafny.Seq
                            out15_: _dafny.Seq
                            out16_: bool
                            out17_: _dafny.Seq
                            out15_, out16_, out17_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_22_cg2_ = out15_
                            d_23_ci2_ = out16_
                            d_24_cc2_ = out17_
                            generated = d_22_cg2_
                            insideConstrainedOut = d_23_ci2_
                            currentConstrainedOut = d_24_cc2_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            insideConstrainedOut = False
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                        raise _dafny.Break("0")
                    elif True:
                        d_25_constrainedPrompt_: _dafny.Seq
                        d_25_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_26_next_: _dafny.Seq
                        out18_: _dafny.Seq
                        out18_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_25_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('2e0'), eosToken)
                        d_26_next_ = out18_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_26_next_) == (eosToken):
                            d_27_rg_: _dafny.Seq
                            d_28_rc_: _dafny.Seq
                            out19_: _dafny.Seq
                            out20_: _dafny.Seq
                            out19_, out20_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                            d_27_rg_ = out19_
                            d_28_rc_ = out20_
                            generated = d_27_rg_
                            currentConstrainedOut = d_28_rc_
                            if ((parser).IsCompletePrefix(currentConstrainedOut)) and (((d_1_steps_) + (1)) <= (maxSteps)):
                                d_29_cg2_: _dafny.Seq
                                d_30_ci2_: bool
                                d_31_cc2_: _dafny.Seq
                                out21_: _dafny.Seq
                                out22_: bool
                                out23_: _dafny.Seq
                                out21_, out22_, out23_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_29_cg2_ = out21_
                                d_30_ci2_ = out22_
                                d_31_cc2_ = out23_
                                generated = d_29_cg2_
                                insideConstrainedOut = d_30_ci2_
                                currentConstrainedOut = d_31_cc2_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                insideConstrainedOut = False
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                            raise _dafny.Break("0")
                        elif True:
                            d_32_ag_: _dafny.Seq
                            d_33_ai_: bool
                            d_34_ac_: _dafny.Seq
                            out24_: _dafny.Seq
                            out25_: bool
                            out26_: _dafny.Seq
                            out24_, out25_, out26_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_26_next_)
                            d_32_ag_ = out24_
                            d_33_ai_ = out25_
                            d_34_ac_ = out26_
                            generated = d_32_ag_
                            insideConstrainedOut = d_33_ai_
                            currentConstrainedOut = d_34_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            if (parser).IsCompletePrefix(currentConstrainedOut):
                d_35_cg_: _dafny.Seq
                d_36_ci_: bool
                d_37_cc_: _dafny.Seq
                out27_: _dafny.Seq
                out28_: bool
                out29_: _dafny.Seq
                out27_, out28_, out29_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_35_cg_ = out27_
                d_36_ci_ = out28_
                d_37_cc_ = out29_
                generated = d_35_cg_
                insideConstrainedOut = d_36_ci_
                currentConstrainedOut = d_37_cc_
                d_1_steps_ = (d_1_steps_) + (1)
            elif True:
                d_38_rg_: _dafny.Seq
                d_39_rc_: _dafny.Seq
                out30_: _dafny.Seq
                out31_: _dafny.Seq
                out30_, out31_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                d_38_rg_ = out30_
                d_39_rc_ = out31_
                generated = d_38_rg_
                currentConstrainedOut = d_39_rc_
                if ((parser).IsCompletePrefix(currentConstrainedOut)) and (((d_1_steps_) + (1)) <= (maxSteps)):
                    d_40_cg2_: _dafny.Seq
                    d_41_ci2_: bool
                    d_42_cc2_: _dafny.Seq
                    out32_: _dafny.Seq
                    out33_: bool
                    out34_: _dafny.Seq
                    out32_, out33_, out34_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                    d_40_cg2_ = out32_
                    d_41_ci2_ = out33_
                    d_42_cc2_ = out34_
                    generated = d_40_cg2_
                    insideConstrainedOut = d_41_ci2_
                    currentConstrainedOut = d_42_cc2_
                    d_1_steps_ = (d_1_steps_) + (1)
                elif True:
                    insideConstrainedOut = False
                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

