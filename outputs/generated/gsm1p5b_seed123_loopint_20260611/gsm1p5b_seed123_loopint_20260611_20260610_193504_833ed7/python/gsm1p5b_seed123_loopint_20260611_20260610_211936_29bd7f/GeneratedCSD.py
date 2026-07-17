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
        d_1_reserve_: int
        d_1_reserve_ = 100
        if (d_1_reserve_) > (maxSteps):
            d_1_reserve_ = maxSteps
        d_2_freeLimit_: int
        d_2_freeLimit_ = (maxSteps) - (d_1_reserve_)
        d_3_steps_: int
        d_3_steps_ = 0
        if insideConstrainedOut:
            d_4_rg0_: _dafny.Seq
            d_5_rc0_: _dafny.Seq
            out0_: _dafny.Seq
            out1_: _dafny.Seq
            out0_, out1_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
            d_4_rg0_ = out0_
            d_5_rc0_ = out1_
            generated = d_4_rg0_
            currentConstrainedOut = d_5_rc0_
            insideConstrainedOut = False
            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
        with _dafny.label("0"):
            while (d_3_steps_) < (d_2_freeLimit_):
                with _dafny.c_label("0"):
                    d_6_next_: _dafny.Seq
                    out2_: _dafny.Seq
                    out2_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                    d_6_next_ = out2_
                    d_3_steps_ = (d_3_steps_) + (1)
                    if (d_6_next_) == (eosToken):
                        raise _dafny.Break("0")
                    elif True:
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_6_next_]))
                    pass
            pass
        if (d_3_steps_) < (maxSteps):
            d_7_fg_: _dafny.Seq
            d_8_fi_: bool
            d_9_fc_: _dafny.Seq
            out3_: _dafny.Seq
            out4_: bool
            out5_: _dafny.Seq
            out3_, out4_, out5_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_7_fg_ = out3_
            d_8_fi_ = out4_
            d_9_fc_ = out5_
            d_3_steps_ = (d_3_steps_) + (1)
            generated = d_7_fg_
            insideConstrainedOut = d_8_fi_
            currentConstrainedOut = d_9_fc_
            with _dafny.label("3_0"):
                while (d_3_steps_) < (maxSteps):
                    with _dafny.c_label("3_0"):
                        if not(insideConstrainedOut):
                            raise _dafny.Break("3_0")
                        d_10_cg_: _dafny.Seq
                        d_11_ci_: bool
                        d_12_cc_: _dafny.Seq
                        d_13_closed_: bool
                        out6_: _dafny.Seq
                        out7_: bool
                        out8_: _dafny.Seq
                        out9_: bool
                        out6_, out7_, out8_, out9_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                        d_10_cg_ = out6_
                        d_11_ci_ = out7_
                        d_12_cc_ = out8_
                        d_13_closed_ = out9_
                        d_3_steps_ = (d_3_steps_) + (1)
                        if d_13_closed_:
                            generated = d_10_cg_
                            insideConstrainedOut = d_11_ci_
                            currentConstrainedOut = d_12_cc_
                            raise _dafny.Break("3_0")
                        elif True:
                            if (d_3_steps_) >= (maxSteps):
                                d_14_rg_: _dafny.Seq
                                d_15_rc_: _dafny.Seq
                                out10_: _dafny.Seq
                                out11_: _dafny.Seq
                                out10_, out11_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                                d_14_rg_ = out10_
                                d_15_rc_ = out11_
                                generated = d_14_rg_
                                currentConstrainedOut = d_15_rc_
                                insideConstrainedOut = False
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                raise _dafny.Break("3_0")
                            d_16_constrainedPrompt_: _dafny.Seq
                            d_16_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_17_next2_: _dafny.Seq
                            out12_: _dafny.Seq
                            out12_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_16_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                            d_17_next2_ = out12_
                            d_3_steps_ = (d_3_steps_) + (1)
                            if (d_17_next2_) == (eosToken):
                                d_18_rg_: _dafny.Seq
                                d_19_rc_: _dafny.Seq
                                out13_: _dafny.Seq
                                out14_: _dafny.Seq
                                out13_, out14_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                                d_18_rg_ = out13_
                                d_19_rc_ = out14_
                                generated = d_18_rg_
                                currentConstrainedOut = d_19_rc_
                                if (d_3_steps_) < (maxSteps):
                                    d_20_cg2_: _dafny.Seq
                                    d_21_ci2_: bool
                                    d_22_cc2_: _dafny.Seq
                                    d_23_closed2_: bool
                                    out15_: _dafny.Seq
                                    out16_: bool
                                    out17_: _dafny.Seq
                                    out18_: bool
                                    out15_, out16_, out17_, out18_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                                    d_20_cg2_ = out15_
                                    d_21_ci2_ = out16_
                                    d_22_cc2_ = out17_
                                    d_23_closed2_ = out18_
                                    d_3_steps_ = (d_3_steps_) + (1)
                                    if d_23_closed2_:
                                        generated = d_20_cg2_
                                        insideConstrainedOut = d_21_ci2_
                                        currentConstrainedOut = d_22_cc2_
                                    elif True:
                                        insideConstrainedOut = False
                                        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                elif True:
                                    insideConstrainedOut = False
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                raise _dafny.Break("3_0")
                            elif True:
                                d_24_ag_: _dafny.Seq
                                d_25_ai_: bool
                                d_26_ac_: _dafny.Seq
                                out19_: _dafny.Seq
                                out20_: bool
                                out21_: _dafny.Seq
                                out19_, out20_, out21_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_17_next2_)
                                d_24_ag_ = out19_
                                d_25_ai_ = out20_
                                d_26_ac_ = out21_
                                generated = d_24_ag_
                                insideConstrainedOut = d_25_ai_
                                currentConstrainedOut = d_26_ac_
                        pass
                pass
        if insideConstrainedOut:
            d_27_rg3_: _dafny.Seq
            d_28_rc3_: _dafny.Seq
            out22_: _dafny.Seq
            out23_: _dafny.Seq
            out22_, out23_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
            d_27_rg3_ = out22_
            d_28_rc3_ = out23_
            generated = d_27_rg3_
            currentConstrainedOut = d_28_rc3_
            insideConstrainedOut = False
            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
        cost = d_3_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

