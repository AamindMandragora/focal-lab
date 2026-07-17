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
        d_1_guidance_: _dafny.Seq
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math problem step by step. Show calculations inside << >> delimiters, like <<3*4=12>>. Put the final numeric answer inside the last << >> span."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_freeTokensBeforeForce_: int
        d_3_freeTokensBeforeForce_ = 40
        d_4_freeTokensSinceLastSpan_: int
        d_4_freeTokensSinceLastSpan_ = 0
        d_5_closeReserve_: int
        d_5_closeReserve_ = 20
        with _dafny.label("0"):
            while (d_2_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    d_6_remaining_: int
                    d_6_remaining_ = (maxSteps) - (d_2_steps_)
                    if not(insideConstrainedOut):
                        if ((d_4_freeTokensSinceLastSpan_) >= (d_3_freeTokensBeforeForce_)) and ((d_6_remaining_) > ((d_5_closeReserve_) + (5))):
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
                            d_2_steps_ = (d_2_steps_) + (1)
                            generated = d_7_og_
                            insideConstrainedOut = d_8_oi_
                            currentConstrainedOut = d_9_oc_
                            d_4_freeTokensSinceLastSpan_ = 0
                        elif True:
                            d_10_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_10_next_ = out3_
                            d_2_steps_ = (d_2_steps_) + (1)
                            if (d_10_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_10_next_]))
                                d_4_freeTokensSinceLastSpan_ = (d_4_freeTokensSinceLastSpan_) + (1)
                                if (d_10_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    d_11_eg_: _dafny.Seq
                                    d_12_ei_: bool
                                    d_13_ec_: _dafny.Seq
                                    out4_: _dafny.Seq
                                    out5_: bool
                                    out6_: _dafny.Seq
                                    out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                    d_11_eg_ = out4_
                                    d_12_ei_ = out5_
                                    d_13_ec_ = out6_
                                    insideConstrainedOut = d_12_ei_
                                    currentConstrainedOut = d_13_ec_
                                    d_4_freeTokensSinceLastSpan_ = 0
                    elif True:
                        if (d_6_remaining_) <= (d_5_closeReserve_):
                            d_14_rg_: _dafny.Seq
                            d_15_rc_: _dafny.Seq
                            out7_: _dafny.Seq
                            out8_: _dafny.Seq
                            out7_, out8_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                            d_14_rg_ = out7_
                            d_15_rc_ = out8_
                            generated = d_14_rg_
                            currentConstrainedOut = d_15_rc_
                            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_2_steps_) < (maxSteps)):
                                d_16_cg_: _dafny.Seq
                                d_17_ci_: bool
                                d_18_cc_: _dafny.Seq
                                out9_: _dafny.Seq
                                out10_: bool
                                out11_: _dafny.Seq
                                out9_, out10_, out11_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_16_cg_ = out9_
                                d_17_ci_ = out10_
                                d_18_cc_ = out11_
                                d_2_steps_ = (d_2_steps_) + (1)
                                generated = d_16_cg_
                                insideConstrainedOut = d_17_ci_
                                currentConstrainedOut = d_18_cc_
                            elif True:
                                d_2_steps_ = (d_2_steps_) + (1)
                                raise _dafny.Break("0")
                        elif (parser).IsCompletePrefix(currentConstrainedOut):
                            d_19_cg_: _dafny.Seq
                            d_20_ci_: bool
                            d_21_cc_: _dafny.Seq
                            out12_: _dafny.Seq
                            out13_: bool
                            out14_: _dafny.Seq
                            out12_, out13_, out14_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_19_cg_ = out12_
                            d_20_ci_ = out13_
                            d_21_cc_ = out14_
                            d_2_steps_ = (d_2_steps_) + (1)
                            generated = d_19_cg_
                            insideConstrainedOut = d_20_ci_
                            currentConstrainedOut = d_21_cc_
                            d_4_freeTokensSinceLastSpan_ = 0
                        elif True:
                            d_22_constrainedPrompt_: _dafny.Seq
                            d_22_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_23_next_: _dafny.Seq
                            out15_: _dafny.Seq
                            out15_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_22_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 20, eosToken)
                            d_23_next_ = out15_
                            d_2_steps_ = (d_2_steps_) + (1)
                            if (d_23_next_) == (eosToken):
                                d_24_rg_: _dafny.Seq
                                d_25_rc_: _dafny.Seq
                                out16_: _dafny.Seq
                                out17_: _dafny.Seq
                                out16_, out17_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                                d_24_rg_ = out16_
                                d_25_rc_ = out17_
                                generated = d_24_rg_
                                currentConstrainedOut = d_25_rc_
                                if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_2_steps_) < (maxSteps)):
                                    d_26_cg_: _dafny.Seq
                                    d_27_ci_: bool
                                    d_28_cc_: _dafny.Seq
                                    out18_: _dafny.Seq
                                    out19_: bool
                                    out20_: _dafny.Seq
                                    out18_, out19_, out20_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_26_cg_ = out18_
                                    d_27_ci_ = out19_
                                    d_28_cc_ = out20_
                                    d_2_steps_ = (d_2_steps_) + (1)
                                    generated = d_26_cg_
                                    insideConstrainedOut = d_27_ci_
                                    currentConstrainedOut = d_28_cc_
                                elif True:
                                    raise _dafny.Break("0")
                            elif True:
                                d_29_ag_: _dafny.Seq
                                d_30_ai_: bool
                                d_31_ac_: _dafny.Seq
                                out21_: _dafny.Seq
                                out22_: bool
                                out23_: _dafny.Seq
                                out21_, out22_, out23_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_23_next_)
                                d_29_ag_ = out21_
                                d_30_ai_ = out22_
                                d_31_ac_ = out23_
                                generated = d_29_ag_
                                insideConstrainedOut = d_30_ai_
                                currentConstrainedOut = d_31_ac_
                                if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_2_steps_) < (maxSteps)):
                                    d_32_cg_: _dafny.Seq
                                    d_33_ci_: bool
                                    d_34_cc_: _dafny.Seq
                                    out24_: _dafny.Seq
                                    out25_: bool
                                    out26_: _dafny.Seq
                                    out24_, out25_, out26_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_32_cg_ = out24_
                                    d_33_ci_ = out25_
                                    d_34_cc_ = out26_
                                    d_2_steps_ = (d_2_steps_) + (1)
                                    generated = d_32_cg_
                                    insideConstrainedOut = d_33_ci_
                                    currentConstrainedOut = d_34_cc_
                                    d_4_freeTokensSinceLastSpan_ = 0
                    pass
            pass
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

