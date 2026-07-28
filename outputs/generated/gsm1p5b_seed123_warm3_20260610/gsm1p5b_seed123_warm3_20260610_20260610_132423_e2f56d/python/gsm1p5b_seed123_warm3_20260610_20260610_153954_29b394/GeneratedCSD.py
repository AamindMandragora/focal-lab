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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve this math word problem step by step. At the end, write ONLY the final arithmetic expression inside << >> delimiters. Use only variable names, numbers, +, -, *, /, (, ) inside the delimiters. Write exactly one complete, closed expression.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_freePhaseLimit_: int
        if (maxSteps) >= (60):
            d_2_freePhaseLimit_ = 30
        elif (maxSteps) >= (10):
            d_2_freePhaseLimit_ = _dafny.euclidian_division(maxSteps, 3)
        elif True:
            d_2_freePhaseLimit_ = 0
        d_3_maxConstrainedSteps_: int
        if (maxSteps) >= (60):
            d_3_maxConstrainedSteps_ = 25
        elif (maxSteps) >= (10):
            d_3_maxConstrainedSteps_ = _dafny.euclidian_division(maxSteps, 4)
        elif True:
            d_3_maxConstrainedSteps_ = 3
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
                        if (parser).IsCompletePrefix(currentConstrainedOut):
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
                        elif (len(currentConstrainedOut)) >= (d_3_maxConstrainedSteps_):
                            d_14_rg_: _dafny.Seq
                            d_15_rc_: _dafny.Seq
                            out10_: _dafny.Seq
                            out11_: _dafny.Seq
                            out10_, out11_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                            d_14_rg_ = out10_
                            d_15_rc_ = out11_
                            generated = d_14_rg_
                            currentConstrainedOut = d_15_rc_
                            if ((parser).IsCompletePrefix(currentConstrainedOut)) and (((d_1_steps_) + (1)) <= (maxSteps)):
                                d_16_cg2_: _dafny.Seq
                                d_17_ci2_: bool
                                d_18_cc2_: _dafny.Seq
                                out12_: _dafny.Seq
                                out13_: bool
                                out14_: _dafny.Seq
                                out12_, out13_, out14_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_16_cg2_ = out12_
                                d_17_ci2_ = out13_
                                d_18_cc2_ = out14_
                                generated = d_16_cg2_
                                insideConstrainedOut = d_17_ci2_
                                currentConstrainedOut = d_18_cc2_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                insideConstrainedOut = False
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                            raise _dafny.Break("0")
                        elif True:
                            d_19_constrainedPrompt_: _dafny.Seq
                            d_19_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_20_next_: _dafny.Seq
                            out15_: _dafny.Seq
                            out15_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_19_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                            d_20_next_ = out15_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_20_next_) == (eosToken):
                                d_21_rg_: _dafny.Seq
                                d_22_rc_: _dafny.Seq
                                out16_: _dafny.Seq
                                out17_: _dafny.Seq
                                out16_, out17_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                                d_21_rg_ = out16_
                                d_22_rc_ = out17_
                                generated = d_21_rg_
                                currentConstrainedOut = d_22_rc_
                                if ((parser).IsCompletePrefix(currentConstrainedOut)) and (((d_1_steps_) + (1)) <= (maxSteps)):
                                    d_23_cg2_: _dafny.Seq
                                    d_24_ci2_: bool
                                    d_25_cc2_: _dafny.Seq
                                    out18_: _dafny.Seq
                                    out19_: bool
                                    out20_: _dafny.Seq
                                    out18_, out19_, out20_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_23_cg2_ = out18_
                                    d_24_ci2_ = out19_
                                    d_25_cc2_ = out20_
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
                                out21_: _dafny.Seq
                                out22_: bool
                                out23_: _dafny.Seq
                                out21_, out22_, out23_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_20_next_)
                                d_26_ag_ = out21_
                                d_27_ai_ = out22_
                                d_28_ac_ = out23_
                                generated = d_26_ag_
                                insideConstrainedOut = d_27_ai_
                                currentConstrainedOut = d_28_ac_
                    pass
            pass
        if insideConstrainedOut:
            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                d_29_cg_: _dafny.Seq
                d_30_ci_: bool
                d_31_cc_: _dafny.Seq
                out24_: _dafny.Seq
                out25_: bool
                out26_: _dafny.Seq
                out24_, out25_, out26_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_29_cg_ = out24_
                d_30_ci_ = out25_
                d_31_cc_ = out26_
                generated = d_29_cg_
                insideConstrainedOut = d_30_ci_
                currentConstrainedOut = d_31_cc_
                d_1_steps_ = (d_1_steps_) + (1)
            elif True:
                d_32_rg_: _dafny.Seq
                d_33_rc_: _dafny.Seq
                out27_: _dafny.Seq
                out28_: _dafny.Seq
                out27_, out28_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                d_32_rg_ = out27_
                d_33_rc_ = out28_
                generated = d_32_rg_
                currentConstrainedOut = d_33_rc_
                if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                    d_34_cg2_: _dafny.Seq
                    d_35_ci2_: bool
                    d_36_cc2_: _dafny.Seq
                    out29_: _dafny.Seq
                    out30_: bool
                    out31_: _dafny.Seq
                    out29_, out30_, out31_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                    d_34_cg2_ = out29_
                    d_35_ci2_ = out30_
                    d_36_cc2_ = out31_
                    generated = d_34_cg2_
                    insideConstrainedOut = d_35_ci2_
                    currentConstrainedOut = d_36_cc2_
                    d_1_steps_ = (d_1_steps_) + (1)
                elif True:
                    insideConstrainedOut = False
                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

