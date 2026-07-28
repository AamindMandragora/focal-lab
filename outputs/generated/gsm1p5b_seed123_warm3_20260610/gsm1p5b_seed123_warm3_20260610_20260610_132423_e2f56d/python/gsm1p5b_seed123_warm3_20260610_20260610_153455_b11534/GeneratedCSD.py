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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve this math word problem step by step. Think carefully. At the very end, write the final arithmetic expression inside << >>. Use only numbers, variable names, +, -, *, /, (, ) inside the delimiters. Write exactly one complete expression.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_freePhaseLimit_: int
        if (maxSteps) >= (100):
            d_2_freePhaseLimit_ = 60
        elif (maxSteps) >= (20):
            d_2_freePhaseLimit_ = _dafny.euclidian_division((maxSteps) * (3), 5)
        elif True:
            d_2_freePhaseLimit_ = _dafny.euclidian_division(maxSteps, 2)
        d_3_constrainedBudget_: int
        if (maxSteps) >= (100):
            d_3_constrainedBudget_ = 40
        elif (maxSteps) >= (10):
            d_3_constrainedBudget_ = _dafny.euclidian_division(maxSteps, 4)
        elif True:
            d_3_constrainedBudget_ = 3
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if (d_1_steps_) >= (d_2_freePhaseLimit_):
                            if ((d_1_steps_) + (1)) <= (maxSteps):
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
                                raise _dafny.Break("0")
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
                            raise _dafny.Break("0")
                        elif True:
                            if (len(currentConstrainedOut)) >= (d_3_constrainedBudget_):
                                d_15_rg_: _dafny.Seq
                                d_16_rc_: _dafny.Seq
                                out11_: _dafny.Seq
                                out12_: _dafny.Seq
                                out11_, out12_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                                d_15_rg_ = out11_
                                d_16_rc_ = out12_
                                generated = d_15_rg_
                                currentConstrainedOut = d_16_rc_
                                if ((parser).IsCompletePrefix(currentConstrainedOut)) and (((d_1_steps_) + (1)) <= (maxSteps)):
                                    d_17_cg2_: _dafny.Seq
                                    d_18_ci2_: bool
                                    d_19_cc2_: _dafny.Seq
                                    out13_: _dafny.Seq
                                    out14_: bool
                                    out15_: _dafny.Seq
                                    out13_, out14_, out15_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_17_cg2_ = out13_
                                    d_18_ci2_ = out14_
                                    d_19_cc2_ = out15_
                                    generated = d_17_cg2_
                                    insideConstrainedOut = d_18_ci2_
                                    currentConstrainedOut = d_19_cc2_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                elif True:
                                    insideConstrainedOut = False
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                raise _dafny.Break("0")
                            elif True:
                                d_20_constrainedPrompt_: _dafny.Seq
                                d_20_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_21_next_: _dafny.Seq
                                out16_: _dafny.Seq
                                out16_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_20_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                                d_21_next_ = out16_
                                if (d_21_next_) == (eosToken):
                                    d_22_rg_: _dafny.Seq
                                    d_23_rc_: _dafny.Seq
                                    out17_: _dafny.Seq
                                    out18_: _dafny.Seq
                                    out17_, out18_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                                    d_22_rg_ = out17_
                                    d_23_rc_ = out18_
                                    generated = d_22_rg_
                                    currentConstrainedOut = d_23_rc_
                                    if ((parser).IsCompletePrefix(currentConstrainedOut)) and (((d_1_steps_) + (1)) <= (maxSteps)):
                                        d_24_cg2_: _dafny.Seq
                                        d_25_ci2_: bool
                                        d_26_cc2_: _dafny.Seq
                                        out19_: _dafny.Seq
                                        out20_: bool
                                        out21_: _dafny.Seq
                                        out19_, out20_, out21_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                        d_24_cg2_ = out19_
                                        d_25_ci2_ = out20_
                                        d_26_cc2_ = out21_
                                        generated = d_24_cg2_
                                        insideConstrainedOut = d_25_ci2_
                                        currentConstrainedOut = d_26_cc2_
                                        d_1_steps_ = (d_1_steps_) + (1)
                                    elif True:
                                        insideConstrainedOut = False
                                        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                    raise _dafny.Break("0")
                                elif True:
                                    d_27_ag_: _dafny.Seq
                                    d_28_ai_: bool
                                    d_29_ac_: _dafny.Seq
                                    out22_: _dafny.Seq
                                    out23_: bool
                                    out24_: _dafny.Seq
                                    out22_, out23_, out24_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_21_next_)
                                    d_27_ag_ = out22_
                                    d_28_ai_ = out23_
                                    d_29_ac_ = out24_
                                    generated = d_27_ag_
                                    insideConstrainedOut = d_28_ai_
                                    currentConstrainedOut = d_29_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            if (parser).IsCompletePrefix(currentConstrainedOut):
                d_30_cg_: _dafny.Seq
                d_31_ci_: bool
                d_32_cc_: _dafny.Seq
                out25_: _dafny.Seq
                out26_: bool
                out27_: _dafny.Seq
                out25_, out26_, out27_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_30_cg_ = out25_
                d_31_ci_ = out26_
                d_32_cc_ = out27_
                generated = d_30_cg_
                insideConstrainedOut = d_31_ci_
                currentConstrainedOut = d_32_cc_
                d_1_steps_ = (d_1_steps_) + (1)
            elif True:
                d_33_rg_: _dafny.Seq
                d_34_rc_: _dafny.Seq
                out28_: _dafny.Seq
                out29_: _dafny.Seq
                out28_, out29_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                d_33_rg_ = out28_
                d_34_rc_ = out29_
                generated = d_33_rg_
                currentConstrainedOut = d_34_rc_
                if ((parser).IsCompletePrefix(currentConstrainedOut)) and (((d_1_steps_) + (1)) <= (maxSteps)):
                    d_35_cg2_: _dafny.Seq
                    d_36_ci2_: bool
                    d_37_cc2_: _dafny.Seq
                    out30_: _dafny.Seq
                    out31_: bool
                    out32_: _dafny.Seq
                    out30_, out31_, out32_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                    d_35_cg2_ = out30_
                    d_36_ci2_ = out31_
                    d_37_cc2_ = out32_
                    generated = d_35_cg2_
                    insideConstrainedOut = d_36_ci2_
                    currentConstrainedOut = d_37_cc2_
                    d_1_steps_ = (d_1_steps_) + (1)
                elif True:
                    insideConstrainedOut = False
                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

