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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve this math word problem step by step using actual numbers from the problem. At the very end, write the final answer as an arithmetic expression with ONLY digits and operators inside << >>. Example: <<(15+3)*2>>. Only numbers and +,-,*,/,(,) inside << >>, absolutely no variable names or letters.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_spanSteps_: int
        d_2_spanSteps_ = 0
        d_3_maxSpanSteps_: int
        d_3_maxSpanSteps_ = 25
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_4_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_4_next_ = out0_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_4_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_4_next_]))
                            if (d_4_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                d_5_og_: _dafny.Seq
                                d_6_oi_: bool
                                d_7_oc_: _dafny.Seq
                                out1_: _dafny.Seq
                                out2_: bool
                                out3_: _dafny.Seq
                                out1_, out2_, out3_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_5_og_ = out1_
                                d_6_oi_ = out2_
                                d_7_oc_ = out3_
                                generated = d_5_og_
                                insideConstrainedOut = d_6_oi_
                                currentConstrainedOut = d_7_oc_
                                d_2_spanSteps_ = 0
                    elif True:
                        d_8_isComplete_: bool
                        d_8_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_8_isComplete_:
                            d_9_cg_: _dafny.Seq
                            d_10_ci_: bool
                            d_11_cc_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_9_cg_ = out4_
                            d_10_ci_ = out5_
                            d_11_cc_ = out6_
                            generated = d_9_cg_
                            insideConstrainedOut = d_10_ci_
                            currentConstrainedOut = d_11_cc_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_2_spanSteps_ = 0
                        elif (d_2_spanSteps_) >= (d_3_maxSpanSteps_):
                            d_12_rg_: _dafny.Seq
                            d_13_rc_: _dafny.Seq
                            out7_: _dafny.Seq
                            out8_: _dafny.Seq
                            out7_, out8_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                            d_12_rg_ = out7_
                            d_13_rc_ = out8_
                            generated = d_12_rg_
                            currentConstrainedOut = d_13_rc_
                            d_14_isNowComplete_: bool
                            d_14_isNowComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                            if (d_14_isNowComplete_) and ((d_1_steps_) < (maxSteps)):
                                d_15_cg_: _dafny.Seq
                                d_16_ci_: bool
                                d_17_cc_: _dafny.Seq
                                out9_: _dafny.Seq
                                out10_: bool
                                out11_: _dafny.Seq
                                out9_, out10_, out11_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_15_cg_ = out9_
                                d_16_ci_ = out10_
                                d_17_cc_ = out11_
                                generated = d_15_cg_
                                insideConstrainedOut = d_16_ci_
                                currentConstrainedOut = d_17_cc_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                insideConstrainedOut = False
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                d_1_steps_ = (d_1_steps_) + (1)
                            d_2_spanSteps_ = 0
                        elif True:
                            d_18_validCount_: int
                            out12_: int
                            out12_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                            d_18_validCount_ = out12_
                            if (d_18_validCount_) == (0):
                                d_19_rg_: _dafny.Seq
                                d_20_rc_: _dafny.Seq
                                out13_: _dafny.Seq
                                out14_: _dafny.Seq
                                out13_, out14_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                                d_19_rg_ = out13_
                                d_20_rc_ = out14_
                                generated = d_19_rg_
                                currentConstrainedOut = d_20_rc_
                                d_21_isNowComplete_: bool
                                d_21_isNowComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                                if (d_21_isNowComplete_) and ((d_1_steps_) < (maxSteps)):
                                    d_22_cg_: _dafny.Seq
                                    d_23_ci_: bool
                                    d_24_cc_: _dafny.Seq
                                    out15_: _dafny.Seq
                                    out16_: bool
                                    out17_: _dafny.Seq
                                    out15_, out16_, out17_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_22_cg_ = out15_
                                    d_23_ci_ = out16_
                                    d_24_cc_ = out17_
                                    generated = d_22_cg_
                                    insideConstrainedOut = d_23_ci_
                                    currentConstrainedOut = d_24_cc_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                elif True:
                                    insideConstrainedOut = False
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                    d_1_steps_ = (d_1_steps_) + (1)
                                d_2_spanSteps_ = 0
                            elif True:
                                d_25_constrainedPrompt_: _dafny.Seq
                                d_25_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_26_next_: _dafny.Seq
                                out18_: _dafny.Seq
                                out18_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_25_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('3e0'), eosToken)
                                d_26_next_ = out18_
                                d_1_steps_ = (d_1_steps_) + (1)
                                d_2_spanSteps_ = (d_2_spanSteps_) + (1)
                                if (d_26_next_) == (eosToken):
                                    d_27_isNowComplete_: bool
                                    d_27_isNowComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                                    if (d_27_isNowComplete_) and ((d_1_steps_) < (maxSteps)):
                                        d_28_cg_: _dafny.Seq
                                        d_29_ci_: bool
                                        d_30_cc_: _dafny.Seq
                                        out19_: _dafny.Seq
                                        out20_: bool
                                        out21_: _dafny.Seq
                                        out19_, out20_, out21_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                        d_28_cg_ = out19_
                                        d_29_ci_ = out20_
                                        d_30_cc_ = out21_
                                        generated = d_28_cg_
                                        insideConstrainedOut = d_29_ci_
                                        currentConstrainedOut = d_30_cc_
                                        d_1_steps_ = (d_1_steps_) + (1)
                                    elif True:
                                        d_31_rg_: _dafny.Seq
                                        d_32_rc_: _dafny.Seq
                                        out22_: _dafny.Seq
                                        out23_: _dafny.Seq
                                        out22_, out23_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                                        d_31_rg_ = out22_
                                        d_32_rc_ = out23_
                                        generated = d_31_rg_
                                        currentConstrainedOut = d_32_rc_
                                        d_33_isRbComplete_: bool
                                        d_33_isRbComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                                        if (d_33_isRbComplete_) and ((d_1_steps_) < (maxSteps)):
                                            d_34_cg_: _dafny.Seq
                                            d_35_ci_: bool
                                            d_36_cc_: _dafny.Seq
                                            out24_: _dafny.Seq
                                            out25_: bool
                                            out26_: _dafny.Seq
                                            out24_, out25_, out26_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                            d_34_cg_ = out24_
                                            d_35_ci_ = out25_
                                            d_36_cc_ = out26_
                                            generated = d_34_cg_
                                            insideConstrainedOut = d_35_ci_
                                            currentConstrainedOut = d_36_cc_
                                            d_1_steps_ = (d_1_steps_) + (1)
                                        elif True:
                                            insideConstrainedOut = False
                                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                    raise _dafny.Break("0")
                                elif True:
                                    d_37_ag_: _dafny.Seq
                                    d_38_ai_: bool
                                    d_39_ac_: _dafny.Seq
                                    out27_: _dafny.Seq
                                    out28_: bool
                                    out29_: _dafny.Seq
                                    out27_, out28_, out29_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_26_next_)
                                    d_37_ag_ = out27_
                                    d_38_ai_ = out28_
                                    d_39_ac_ = out29_
                                    generated = d_37_ag_
                                    insideConstrainedOut = d_38_ai_
                                    currentConstrainedOut = d_39_ac_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

