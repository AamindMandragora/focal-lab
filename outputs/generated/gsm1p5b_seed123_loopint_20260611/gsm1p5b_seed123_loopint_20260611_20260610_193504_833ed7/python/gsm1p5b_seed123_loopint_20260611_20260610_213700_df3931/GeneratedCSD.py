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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve this math word problem step by step. Compute with actual numbers at each step. At the very end, write your final numeric answer as a single arithmetic expression using ONLY digits and operators (+, -, *, /) between << and >>. Example: <<(15 * 3) / 5>>. The content between << and >> must contain only numbers and arithmetic operators, no words or variable names.")))
        d_1_steps_: int
        d_1_steps_ = 0
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_2_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_2_next_ = out0_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_2_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_2_next_]))
                            if (d_2_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                d_3_og_: _dafny.Seq
                                d_4_oi_: bool
                                d_5_oc_: _dafny.Seq
                                out1_: _dafny.Seq
                                out2_: bool
                                out3_: _dafny.Seq
                                out1_, out2_, out3_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_3_og_ = out1_
                                d_4_oi_ = out2_
                                d_5_oc_ = out3_
                                generated = d_3_og_
                                insideConstrainedOut = d_4_oi_
                                currentConstrainedOut = d_5_oc_
                    elif True:
                        d_6_cg_: _dafny.Seq
                        d_7_ci_: bool
                        d_8_cc_: _dafny.Seq
                        d_9_closed_: bool
                        out4_: _dafny.Seq
                        out5_: bool
                        out6_: _dafny.Seq
                        out7_: bool
                        out4_, out5_, out6_, out7_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                        d_6_cg_ = out4_
                        d_7_ci_ = out5_
                        d_8_cc_ = out6_
                        d_9_closed_ = out7_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if d_9_closed_:
                            generated = d_6_cg_
                            insideConstrainedOut = d_7_ci_
                            currentConstrainedOut = d_8_cc_
                        elif True:
                            if ((d_1_steps_) + (1)) >= (maxSteps):
                                d_10_rg_: _dafny.Seq
                                d_11_rc_: _dafny.Seq
                                out8_: _dafny.Seq
                                out9_: _dafny.Seq
                                out8_, out9_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                                d_10_rg_ = out8_
                                d_11_rc_ = out9_
                                generated = d_10_rg_
                                currentConstrainedOut = d_11_rc_
                                if (d_1_steps_) < (maxSteps):
                                    d_12_cg2_: _dafny.Seq
                                    d_13_ci2_: bool
                                    d_14_cc2_: _dafny.Seq
                                    d_15_closed2_: bool
                                    out10_: _dafny.Seq
                                    out11_: bool
                                    out12_: _dafny.Seq
                                    out13_: bool
                                    out10_, out11_, out12_, out13_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                                    d_12_cg2_ = out10_
                                    d_13_ci2_ = out11_
                                    d_14_cc2_ = out12_
                                    d_15_closed2_ = out13_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    generated = d_12_cg2_
                                    insideConstrainedOut = d_13_ci2_
                                    currentConstrainedOut = d_14_cc2_
                                if insideConstrainedOut:
                                    insideConstrainedOut = False
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                raise _dafny.Break("0")
                            d_16_constrainedPrompt_: _dafny.Seq
                            d_16_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_17_next_: _dafny.Seq
                            out14_: _dafny.Seq
                            out14_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_16_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                            d_17_next_ = out14_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_17_next_) == (eosToken):
                                d_18_rg_: _dafny.Seq
                                d_19_rc_: _dafny.Seq
                                out15_: _dafny.Seq
                                out16_: _dafny.Seq
                                out15_, out16_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                                d_18_rg_ = out15_
                                d_19_rc_ = out16_
                                generated = d_18_rg_
                                currentConstrainedOut = d_19_rc_
                                if (d_1_steps_) < (maxSteps):
                                    d_20_cg3_: _dafny.Seq
                                    d_21_ci3_: bool
                                    d_22_cc3_: _dafny.Seq
                                    d_23_closed3_: bool
                                    out17_: _dafny.Seq
                                    out18_: bool
                                    out19_: _dafny.Seq
                                    out20_: bool
                                    out17_, out18_, out19_, out20_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                                    d_20_cg3_ = out17_
                                    d_21_ci3_ = out18_
                                    d_22_cc3_ = out19_
                                    d_23_closed3_ = out20_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    generated = d_20_cg3_
                                    insideConstrainedOut = d_21_ci3_
                                    currentConstrainedOut = d_22_cc3_
                                if insideConstrainedOut:
                                    insideConstrainedOut = False
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                            elif True:
                                d_24_ag_: _dafny.Seq
                                d_25_ai_: bool
                                d_26_ac_: _dafny.Seq
                                out21_: _dafny.Seq
                                out22_: bool
                                out23_: _dafny.Seq
                                out21_, out22_, out23_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_17_next_)
                                d_24_ag_ = out21_
                                d_25_ai_ = out22_
                                d_26_ac_ = out23_
                                generated = d_24_ag_
                                insideConstrainedOut = d_25_ai_
                                currentConstrainedOut = d_26_ac_
                    pass
            pass
        if insideConstrainedOut:
            d_27_rg4_: _dafny.Seq
            d_28_rc4_: _dafny.Seq
            out24_: _dafny.Seq
            out25_: _dafny.Seq
            out24_, out25_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
            d_27_rg4_ = out24_
            d_28_rc4_ = out25_
            generated = d_27_rg4_
            currentConstrainedOut = d_28_rc4_
            if (d_1_steps_) < (maxSteps):
                d_29_cg5_: _dafny.Seq
                d_30_ci5_: bool
                d_31_cc5_: _dafny.Seq
                d_32_closed5_: bool
                out26_: _dafny.Seq
                out27_: bool
                out28_: _dafny.Seq
                out29_: bool
                out26_, out27_, out28_, out29_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                d_29_cg5_ = out26_
                d_30_ci5_ = out27_
                d_31_cc5_ = out28_
                d_32_closed5_ = out29_
                d_1_steps_ = (d_1_steps_) + (1)
                generated = d_29_cg5_
                insideConstrainedOut = d_30_ci5_
                currentConstrainedOut = d_31_cc5_
            if insideConstrainedOut:
                insideConstrainedOut = False
                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

