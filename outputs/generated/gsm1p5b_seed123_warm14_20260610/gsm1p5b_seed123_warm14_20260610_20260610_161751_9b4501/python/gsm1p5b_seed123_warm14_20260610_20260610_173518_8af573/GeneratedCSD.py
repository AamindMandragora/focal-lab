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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Output ONLY a Python arithmetic expression that computes the answer. Use integers and +,-,*,/,(). No text, no variables, just the expression.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_freePhaseLimit_: int
        if (maxSteps) >= (10):
            d_2_freePhaseLimit_ = 3
        elif True:
            d_2_freePhaseLimit_ = 0
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if (d_1_steps_) >= (d_2_freePhaseLimit_):
                            d_3_og_: _dafny.Seq
                            d_4_oi_: bool
                            d_5_oc_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_3_og_ = out0_
                            d_4_oi_ = out1_
                            d_5_oc_ = out2_
                            generated = d_3_og_
                            insideConstrainedOut = d_4_oi_
                            currentConstrainedOut = d_5_oc_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_6_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_6_next_ = out3_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_6_next_) == (eosToken):
                                if ((d_1_steps_) + (2)) <= (maxSteps):
                                    d_7_og_: _dafny.Seq
                                    d_8_oi_: bool
                                    d_9_oc_: _dafny.Seq
                                    out4_: _dafny.Seq
                                    out5_: bool
                                    out6_: _dafny.Seq
                                    out4_, out5_, out6_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                    d_7_og_ = out4_
                                    d_8_oi_ = out5_
                                    d_9_oc_ = out6_
                                    generated = d_7_og_
                                    insideConstrainedOut = d_8_oi_
                                    currentConstrainedOut = d_9_oc_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                elif True:
                                    raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_6_next_]))
                    elif True:
                        d_10_cg_: _dafny.Seq
                        d_11_ci_: bool
                        d_12_cc_: _dafny.Seq
                        d_13_closed_: bool
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: _dafny.Seq
                        out10_: bool
                        out7_, out8_, out9_, out10_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                        d_10_cg_ = out7_
                        d_11_ci_ = out8_
                        d_12_cc_ = out9_
                        d_13_closed_ = out10_
                        if d_13_closed_:
                            generated = d_10_cg_
                            insideConstrainedOut = d_11_ci_
                            currentConstrainedOut = d_12_cc_
                            d_1_steps_ = (d_1_steps_) + (1)
                            raise _dafny.Break("0")
                        elif True:
                            if ((d_1_steps_) + (2)) > (maxSteps):
                                d_14_rg_: _dafny.Seq
                                d_15_rc_: _dafny.Seq
                                out11_: _dafny.Seq
                                out12_: _dafny.Seq
                                out11_, out12_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                                d_14_rg_ = out11_
                                d_15_rc_ = out12_
                                generated = d_14_rg_
                                currentConstrainedOut = d_15_rc_
                                if ((parser).IsCompletePrefix(currentConstrainedOut)) and (((d_1_steps_) + (1)) <= (maxSteps)):
                                    d_16_cg2_: _dafny.Seq
                                    d_17_ci2_: bool
                                    d_18_cc2_: _dafny.Seq
                                    out13_: _dafny.Seq
                                    out14_: bool
                                    out15_: _dafny.Seq
                                    out13_, out14_, out15_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_16_cg2_ = out13_
                                    d_17_ci2_ = out14_
                                    d_18_cc2_ = out15_
                                    generated = d_16_cg2_
                                    insideConstrainedOut = d_17_ci2_
                                    currentConstrainedOut = d_18_cc2_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                elif True:
                                    insideConstrainedOut = False
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                    d_1_steps_ = (d_1_steps_) + (1)
                                raise _dafny.Break("0")
                            elif True:
                                d_19_constrainedPrompt_: _dafny.Seq
                                d_19_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_20_next_: _dafny.Seq
                                out16_: _dafny.Seq
                                out16_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_19_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                                d_20_next_ = out16_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_20_next_) == (eosToken):
                                    d_21_rg_: _dafny.Seq
                                    d_22_rc_: _dafny.Seq
                                    out17_: _dafny.Seq
                                    out18_: _dafny.Seq
                                    out17_, out18_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                                    d_21_rg_ = out17_
                                    d_22_rc_ = out18_
                                    generated = d_21_rg_
                                    currentConstrainedOut = d_22_rc_
                                    if ((parser).IsCompletePrefix(currentConstrainedOut)) and (((d_1_steps_) + (1)) <= (maxSteps)):
                                        d_23_cg2_: _dafny.Seq
                                        d_24_ci2_: bool
                                        d_25_cc2_: _dafny.Seq
                                        out19_: _dafny.Seq
                                        out20_: bool
                                        out21_: _dafny.Seq
                                        out19_, out20_, out21_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                        d_23_cg2_ = out19_
                                        d_24_ci2_ = out20_
                                        d_25_cc2_ = out21_
                                        generated = d_23_cg2_
                                        insideConstrainedOut = d_24_ci2_
                                        currentConstrainedOut = d_25_cc2_
                                        d_1_steps_ = (d_1_steps_) + (1)
                                    elif True:
                                        insideConstrainedOut = False
                                        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                    raise _dafny.Break("0")
                                elif True:
                                    d_26_ag_: _dafny.Seq
                                    d_27_ai_: bool
                                    d_28_ac_: _dafny.Seq
                                    out22_: _dafny.Seq
                                    out23_: bool
                                    out24_: _dafny.Seq
                                    out22_, out23_, out24_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_20_next_)
                                    d_26_ag_ = out22_
                                    d_27_ai_ = out23_
                                    d_28_ac_ = out24_
                                    generated = d_26_ag_
                                    insideConstrainedOut = d_27_ai_
                                    currentConstrainedOut = d_28_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            if (parser).IsCompletePrefix(currentConstrainedOut):
                d_29_cg_: _dafny.Seq
                d_30_ci_: bool
                d_31_cc_: _dafny.Seq
                out25_: _dafny.Seq
                out26_: bool
                out27_: _dafny.Seq
                out25_, out26_, out27_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_29_cg_ = out25_
                d_30_ci_ = out26_
                d_31_cc_ = out27_
                generated = d_29_cg_
                insideConstrainedOut = d_30_ci_
                currentConstrainedOut = d_31_cc_
                d_1_steps_ = (d_1_steps_) + (1)
            elif True:
                d_32_rg_: _dafny.Seq
                d_33_rc_: _dafny.Seq
                out28_: _dafny.Seq
                out29_: _dafny.Seq
                out28_, out29_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                d_32_rg_ = out28_
                d_33_rc_ = out29_
                generated = d_32_rg_
                currentConstrainedOut = d_33_rc_
                if ((parser).IsCompletePrefix(currentConstrainedOut)) and (((d_1_steps_) + (1)) <= (maxSteps)):
                    d_34_cg2_: _dafny.Seq
                    d_35_ci2_: bool
                    d_36_cc2_: _dafny.Seq
                    out30_: _dafny.Seq
                    out31_: bool
                    out32_: _dafny.Seq
                    out30_, out31_, out32_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                    d_34_cg2_ = out30_
                    d_35_ci2_ = out31_
                    d_36_cc2_ = out32_
                    generated = d_34_cg2_
                    insideConstrainedOut = d_35_ci2_
                    currentConstrainedOut = d_36_cc2_
                    d_1_steps_ = (d_1_steps_) + (1)
                elif True:
                    insideConstrainedOut = False
                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

