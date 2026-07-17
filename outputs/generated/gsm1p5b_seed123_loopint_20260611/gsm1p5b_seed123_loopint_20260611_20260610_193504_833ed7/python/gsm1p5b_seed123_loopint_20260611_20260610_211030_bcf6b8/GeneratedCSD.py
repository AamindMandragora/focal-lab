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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve this math problem step by step, showing your work. At the end, provide the final numeric answer as an arithmetic expression using only numbers and operators.")))
        d_1_reserve_: int
        d_1_reserve_ = 50
        if (d_1_reserve_) > (maxSteps):
            d_1_reserve_ = maxSteps
        d_2_freeLimit_: int
        d_2_freeLimit_ = (maxSteps) - (d_1_reserve_)
        d_3_steps_: int
        d_3_steps_ = 0
        d_4_eosHit_: bool
        d_4_eosHit_ = False
        with _dafny.label("0"):
            while (d_3_steps_) < (d_2_freeLimit_):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_5_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_5_next_ = out0_
                        d_3_steps_ = (d_3_steps_) + (1)
                        if (d_5_next_) == (eosToken):
                            d_4_eosHit_ = True
                            raise _dafny.Break("0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_5_next_]))
                            if (d_5_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                d_6_og_: _dafny.Seq
                                d_7_oi_: bool
                                d_8_oc_: _dafny.Seq
                                out1_: _dafny.Seq
                                out2_: bool
                                out3_: _dafny.Seq
                                out1_, out2_, out3_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_6_og_ = out1_
                                d_7_oi_ = out2_
                                d_8_oc_ = out3_
                                generated = d_6_og_
                                insideConstrainedOut = d_7_oi_
                                currentConstrainedOut = d_8_oc_
                    elif True:
                        d_9_cg_: _dafny.Seq
                        d_10_ci_: bool
                        d_11_cc_: _dafny.Seq
                        d_12_closed_: bool
                        out4_: _dafny.Seq
                        out5_: bool
                        out6_: _dafny.Seq
                        out7_: bool
                        out4_, out5_, out6_, out7_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                        d_9_cg_ = out4_
                        d_10_ci_ = out5_
                        d_11_cc_ = out6_
                        d_12_closed_ = out7_
                        d_3_steps_ = (d_3_steps_) + (1)
                        if d_12_closed_:
                            generated = d_9_cg_
                            insideConstrainedOut = d_10_ci_
                            currentConstrainedOut = d_11_cc_
                        elif True:
                            if (d_3_steps_) >= (d_2_freeLimit_):
                                d_13_rg_: _dafny.Seq
                                d_14_rc_: _dafny.Seq
                                out8_: _dafny.Seq
                                out9_: _dafny.Seq
                                out8_, out9_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                                d_13_rg_ = out8_
                                d_14_rc_ = out9_
                                generated = d_13_rg_
                                currentConstrainedOut = d_14_rc_
                                insideConstrainedOut = False
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                raise _dafny.Break("0")
                            d_15_constrainedPrompt_: _dafny.Seq
                            d_15_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_16_next_: _dafny.Seq
                            out10_: _dafny.Seq
                            out10_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_15_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                            d_16_next_ = out10_
                            d_3_steps_ = (d_3_steps_) + (1)
                            if (d_16_next_) == (eosToken):
                                d_17_rg_: _dafny.Seq
                                d_18_rc_: _dafny.Seq
                                out11_: _dafny.Seq
                                out12_: _dafny.Seq
                                out11_, out12_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                                d_17_rg_ = out11_
                                d_18_rc_ = out12_
                                generated = d_17_rg_
                                currentConstrainedOut = d_18_rc_
                                insideConstrainedOut = False
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                raise _dafny.Break("0")
                            elif True:
                                d_19_ag_: _dafny.Seq
                                d_20_ai_: bool
                                d_21_ac_: _dafny.Seq
                                out13_: _dafny.Seq
                                out14_: bool
                                out15_: _dafny.Seq
                                out13_, out14_, out15_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_16_next_)
                                d_19_ag_ = out13_
                                d_20_ai_ = out14_
                                d_21_ac_ = out15_
                                generated = d_19_ag_
                                insideConstrainedOut = d_20_ai_
                                currentConstrainedOut = d_21_ac_
                    pass
            pass
        if insideConstrainedOut:
            d_22_rg_: _dafny.Seq
            d_23_rc_: _dafny.Seq
            out16_: _dafny.Seq
            out17_: _dafny.Seq
            out16_, out17_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
            d_22_rg_ = out16_
            d_23_rc_ = out17_
            generated = d_22_rg_
            currentConstrainedOut = d_23_rc_
            insideConstrainedOut = False
            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
        if (d_3_steps_) < (maxSteps):
            d_24_fg_: _dafny.Seq
            d_25_fi_: bool
            d_26_fc_: _dafny.Seq
            out18_: _dafny.Seq
            out19_: bool
            out20_: _dafny.Seq
            out18_, out19_, out20_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_24_fg_ = out18_
            d_25_fi_ = out19_
            d_26_fc_ = out20_
            d_3_steps_ = (d_3_steps_) + (1)
            generated = d_24_fg_
            insideConstrainedOut = d_25_fi_
            currentConstrainedOut = d_26_fc_
            with _dafny.label("3_0"):
                while (d_3_steps_) < (maxSteps):
                    with _dafny.c_label("3_0"):
                        if not(insideConstrainedOut):
                            raise _dafny.Break("3_0")
                        d_27_cg2_: _dafny.Seq
                        d_28_ci2_: bool
                        d_29_cc2_: _dafny.Seq
                        d_30_closed2_: bool
                        out21_: _dafny.Seq
                        out22_: bool
                        out23_: _dafny.Seq
                        out24_: bool
                        out21_, out22_, out23_, out24_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                        d_27_cg2_ = out21_
                        d_28_ci2_ = out22_
                        d_29_cc2_ = out23_
                        d_30_closed2_ = out24_
                        d_3_steps_ = (d_3_steps_) + (1)
                        if d_30_closed2_:
                            generated = d_27_cg2_
                            insideConstrainedOut = d_28_ci2_
                            currentConstrainedOut = d_29_cc2_
                            raise _dafny.Break("3_0")
                        elif True:
                            if (d_3_steps_) >= (maxSteps):
                                d_31_rg2_: _dafny.Seq
                                d_32_rc2_: _dafny.Seq
                                out25_: _dafny.Seq
                                out26_: _dafny.Seq
                                out25_, out26_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                                d_31_rg2_ = out25_
                                d_32_rc2_ = out26_
                                generated = d_31_rg2_
                                currentConstrainedOut = d_32_rc2_
                                insideConstrainedOut = False
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                raise _dafny.Break("3_0")
                            d_33_constrainedPrompt2_: _dafny.Seq
                            d_33_constrainedPrompt2_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_34_next2_: _dafny.Seq
                            out27_: _dafny.Seq
                            out27_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_33_constrainedPrompt2_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                            d_34_next2_ = out27_
                            d_3_steps_ = (d_3_steps_) + (1)
                            if (d_34_next2_) == (eosToken):
                                d_35_rg2_: _dafny.Seq
                                d_36_rc2_: _dafny.Seq
                                out28_: _dafny.Seq
                                out29_: _dafny.Seq
                                out28_, out29_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                                d_35_rg2_ = out28_
                                d_36_rc2_ = out29_
                                generated = d_35_rg2_
                                currentConstrainedOut = d_36_rc2_
                                if (d_3_steps_) < (maxSteps):
                                    d_37_cg3_: _dafny.Seq
                                    d_38_ci3_: bool
                                    d_39_cc3_: _dafny.Seq
                                    d_40_closed3_: bool
                                    out30_: _dafny.Seq
                                    out31_: bool
                                    out32_: _dafny.Seq
                                    out33_: bool
                                    out30_, out31_, out32_, out33_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                                    d_37_cg3_ = out30_
                                    d_38_ci3_ = out31_
                                    d_39_cc3_ = out32_
                                    d_40_closed3_ = out33_
                                    d_3_steps_ = (d_3_steps_) + (1)
                                    if d_40_closed3_:
                                        generated = d_37_cg3_
                                        insideConstrainedOut = d_38_ci3_
                                        currentConstrainedOut = d_39_cc3_
                                    elif True:
                                        insideConstrainedOut = False
                                        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                elif True:
                                    insideConstrainedOut = False
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                raise _dafny.Break("3_0")
                            elif True:
                                d_41_ag2_: _dafny.Seq
                                d_42_ai2_: bool
                                d_43_ac2_: _dafny.Seq
                                out34_: _dafny.Seq
                                out35_: bool
                                out36_: _dafny.Seq
                                out34_, out35_, out36_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_34_next2_)
                                d_41_ag2_ = out34_
                                d_42_ai2_ = out35_
                                d_43_ac2_ = out36_
                                generated = d_41_ag2_
                                insideConstrainedOut = d_42_ai2_
                                currentConstrainedOut = d_43_ac2_
                        pass
                pass
        if insideConstrainedOut:
            d_44_rg3_: _dafny.Seq
            d_45_rc3_: _dafny.Seq
            out37_: _dafny.Seq
            out38_: _dafny.Seq
            out37_, out38_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
            d_44_rg3_ = out37_
            d_45_rc3_ = out38_
            generated = d_44_rg3_
            currentConstrainedOut = d_45_rc3_
            insideConstrainedOut = False
            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
        cost = d_3_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

