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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve this math problem step by step. Show your reasoning with the actual numbers given. At the very end, write the final numeric answer as a single arithmetic expression inside << >> delimiters. Use only numbers and operators +,-,*,/,(,) inside the << >>. Do not use variable names inside << >>. Example format: <<(15+3)*2>>")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_reasoningBudget_: int
        if (maxSteps) > (50):
            d_2_reasoningBudget_ = (maxSteps) - (40)
        elif True:
            if (maxSteps) > (10):
                d_2_reasoningBudget_ = (maxSteps) - (8)
            elif True:
                d_2_reasoningBudget_ = _dafny.euclidian_division(maxSteps, 2)
        d_3_spanSteps_: int
        d_3_spanSteps_ = 0
        d_4_maxSpanSteps_: int
        d_4_maxSpanSteps_ = 35
        d_5_forcedOpen_: bool
        d_5_forcedOpen_ = False
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if ((d_1_steps_) >= (d_2_reasoningBudget_)) and (((d_1_steps_) + (5)) <= (maxSteps)):
                            d_6_og_: _dafny.Seq
                            d_7_oi_: bool
                            d_8_oc_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_6_og_ = out0_
                            d_7_oi_ = out1_
                            d_8_oc_ = out2_
                            generated = d_6_og_
                            insideConstrainedOut = d_7_oi_
                            currentConstrainedOut = d_8_oc_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_3_spanSteps_ = 0
                            d_5_forcedOpen_ = True
                        elif True:
                            d_9_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_9_next_ = out3_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_9_next_) == (eosToken):
                                if ((d_1_steps_) + (5)) <= (maxSteps):
                                    d_10_og_: _dafny.Seq
                                    d_11_oi_: bool
                                    d_12_oc_: _dafny.Seq
                                    out4_: _dafny.Seq
                                    out5_: bool
                                    out6_: _dafny.Seq
                                    out4_, out5_, out6_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                    d_10_og_ = out4_
                                    d_11_oi_ = out5_
                                    d_12_oc_ = out6_
                                    generated = d_10_og_
                                    insideConstrainedOut = d_11_oi_
                                    currentConstrainedOut = d_12_oc_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    d_3_spanSteps_ = 0
                                    d_5_forcedOpen_ = True
                                elif True:
                                    raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_9_next_]))
                                if (d_9_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    d_13_og_: _dafny.Seq
                                    d_14_oi_: bool
                                    d_15_oc_: _dafny.Seq
                                    out7_: _dafny.Seq
                                    out8_: bool
                                    out9_: _dafny.Seq
                                    out7_, out8_, out9_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                    d_13_og_ = out7_
                                    d_14_oi_ = out8_
                                    d_15_oc_ = out9_
                                    generated = d_13_og_
                                    insideConstrainedOut = d_14_oi_
                                    currentConstrainedOut = d_15_oc_
                                    d_3_spanSteps_ = 0
                                    d_5_forcedOpen_ = False
                    elif True:
                        d_16_isComplete_: bool
                        d_16_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_16_isComplete_:
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
                            d_3_spanSteps_ = 0
                            if d_5_forcedOpen_:
                                raise _dafny.Break("0")
                        elif ((d_3_spanSteps_) >= (d_4_maxSpanSteps_)) or (((d_1_steps_) + (2)) > (maxSteps)):
                            d_20_rg_: _dafny.Seq
                            d_21_rc_: _dafny.Seq
                            out13_: _dafny.Seq
                            out14_: _dafny.Seq
                            out13_, out14_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                            d_20_rg_ = out13_
                            d_21_rc_ = out14_
                            generated = d_20_rg_
                            currentConstrainedOut = d_21_rc_
                            d_22_isNowComplete_: bool
                            d_22_isNowComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                            if (d_22_isNowComplete_) and (((d_1_steps_) + (1)) <= (maxSteps)):
                                d_23_cg_: _dafny.Seq
                                d_24_ci_: bool
                                d_25_cc_: _dafny.Seq
                                out15_: _dafny.Seq
                                out16_: bool
                                out17_: _dafny.Seq
                                out15_, out16_, out17_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_23_cg_ = out15_
                                d_24_ci_ = out16_
                                d_25_cc_ = out17_
                                generated = d_23_cg_
                                insideConstrainedOut = d_24_ci_
                                currentConstrainedOut = d_25_cc_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                insideConstrainedOut = False
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                d_1_steps_ = (d_1_steps_) + (1)
                            d_3_spanSteps_ = 0
                            raise _dafny.Break("0")
                        elif True:
                            d_26_constrainedPrompt_: _dafny.Seq
                            d_26_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_27_next_: _dafny.Seq
                            out18_: _dafny.Seq
                            out18_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_26_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 15, eosToken)
                            d_27_next_ = out18_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_3_spanSteps_ = (d_3_spanSteps_) + (1)
                            if (d_27_next_) == (eosToken):
                                d_28_isNowComplete_: bool
                                d_28_isNowComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                                if (d_28_isNowComplete_) and ((d_1_steps_) <= (maxSteps)):
                                    if (d_1_steps_) < (maxSteps):
                                        d_29_cg_: _dafny.Seq
                                        d_30_ci_: bool
                                        d_31_cc_: _dafny.Seq
                                        out19_: _dafny.Seq
                                        out20_: bool
                                        out21_: _dafny.Seq
                                        out19_, out20_, out21_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                        d_29_cg_ = out19_
                                        d_30_ci_ = out20_
                                        d_31_cc_ = out21_
                                        generated = d_29_cg_
                                        insideConstrainedOut = d_30_ci_
                                        currentConstrainedOut = d_31_cc_
                                        d_1_steps_ = (d_1_steps_) + (1)
                                elif True:
                                    d_32_rg_: _dafny.Seq
                                    d_33_rc_: _dafny.Seq
                                    out22_: _dafny.Seq
                                    out23_: _dafny.Seq
                                    out22_, out23_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                                    d_32_rg_ = out22_
                                    d_33_rc_ = out23_
                                    generated = d_32_rg_
                                    currentConstrainedOut = d_33_rc_
                                    d_34_isRbComplete_: bool
                                    d_34_isRbComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                                    if (d_34_isRbComplete_) and ((d_1_steps_) < (maxSteps)):
                                        d_35_cg_: _dafny.Seq
                                        d_36_ci_: bool
                                        d_37_cc_: _dafny.Seq
                                        out24_: _dafny.Seq
                                        out25_: bool
                                        out26_: _dafny.Seq
                                        out24_, out25_, out26_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                        d_35_cg_ = out24_
                                        d_36_ci_ = out25_
                                        d_37_cc_ = out26_
                                        generated = d_35_cg_
                                        insideConstrainedOut = d_36_ci_
                                        currentConstrainedOut = d_37_cc_
                                        d_1_steps_ = (d_1_steps_) + (1)
                                    elif True:
                                        insideConstrainedOut = False
                                        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                raise _dafny.Break("0")
                            elif True:
                                d_38_ag_: _dafny.Seq
                                d_39_ai_: bool
                                d_40_ac_: _dafny.Seq
                                out27_: _dafny.Seq
                                out28_: bool
                                out29_: _dafny.Seq
                                out27_, out28_, out29_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_27_next_)
                                d_38_ag_ = out27_
                                d_39_ai_ = out28_
                                d_40_ac_ = out29_
                                generated = d_38_ag_
                                insideConstrainedOut = d_39_ai_
                                currentConstrainedOut = d_40_ac_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

