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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve this math word problem step by step. Compute with actual numbers. Write the final numeric answer as an arithmetic expression using ONLY digits and operators +,-,*,/,(,) inside << >>. No variables, no letters, no spaces inside the expression. Example: <<(15+3)*2>>. The last << >> must contain only digits and arithmetic operators.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_spanSteps_: int
        d_2_spanSteps_ = 0
        d_3_maxSpanSteps_: int
        d_3_maxSpanSteps_ = 20
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
                                d_5_validCount_: int
                                out1_: int
                                out1_ = (d_0_helpers_).ValidTokenCount(parser, _dafny.SeqWithoutIsStrInference([]))
                                d_5_validCount_ = out1_
                                if (d_5_validCount_) >= (2):
                                    d_6_og_: _dafny.Seq
                                    d_7_oi_: bool
                                    d_8_oc_: _dafny.Seq
                                    out2_: _dafny.Seq
                                    out3_: bool
                                    out4_: _dafny.Seq
                                    out2_, out3_, out4_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                    d_6_og_ = out2_
                                    d_7_oi_ = out3_
                                    d_8_oc_ = out4_
                                    generated = d_6_og_
                                    insideConstrainedOut = d_7_oi_
                                    currentConstrainedOut = d_8_oc_
                                    d_2_spanSteps_ = 0
                    elif True:
                        d_9_isComplete_: bool
                        d_9_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_9_isComplete_:
                            d_10_cg_: _dafny.Seq
                            d_11_ci_: bool
                            d_12_cc_: _dafny.Seq
                            out5_: _dafny.Seq
                            out6_: bool
                            out7_: _dafny.Seq
                            out5_, out6_, out7_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_10_cg_ = out5_
                            d_11_ci_ = out6_
                            d_12_cc_ = out7_
                            generated = d_10_cg_
                            insideConstrainedOut = d_11_ci_
                            currentConstrainedOut = d_12_cc_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_2_spanSteps_ = 0
                        elif (d_2_spanSteps_) >= (d_3_maxSpanSteps_):
                            d_13_rg_: _dafny.Seq
                            d_14_rc_: _dafny.Seq
                            out8_: _dafny.Seq
                            out9_: _dafny.Seq
                            out8_, out9_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                            d_13_rg_ = out8_
                            d_14_rc_ = out9_
                            generated = d_13_rg_
                            currentConstrainedOut = d_14_rc_
                            d_15_isNowComplete_: bool
                            d_15_isNowComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                            if (d_15_isNowComplete_) and ((d_1_steps_) < (maxSteps)):
                                d_16_cg_: _dafny.Seq
                                d_17_ci_: bool
                                d_18_cc_: _dafny.Seq
                                out10_: _dafny.Seq
                                out11_: bool
                                out12_: _dafny.Seq
                                out10_, out11_, out12_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_16_cg_ = out10_
                                d_17_ci_ = out11_
                                d_18_cc_ = out12_
                                generated = d_16_cg_
                                insideConstrainedOut = d_17_ci_
                                currentConstrainedOut = d_18_cc_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                insideConstrainedOut = False
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                d_1_steps_ = (d_1_steps_) + (1)
                            d_2_spanSteps_ = 0
                        elif True:
                            d_19_validCount_: int
                            out13_: int
                            out13_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                            d_19_validCount_ = out13_
                            if (d_19_validCount_) == (0):
                                d_20_rg_: _dafny.Seq
                                d_21_rc_: _dafny.Seq
                                out14_: _dafny.Seq
                                out15_: _dafny.Seq
                                out14_, out15_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                                d_20_rg_ = out14_
                                d_21_rc_ = out15_
                                generated = d_20_rg_
                                currentConstrainedOut = d_21_rc_
                                d_22_isNowComplete_: bool
                                d_22_isNowComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                                if (d_22_isNowComplete_) and ((d_1_steps_) < (maxSteps)):
                                    d_23_cg_: _dafny.Seq
                                    d_24_ci_: bool
                                    d_25_cc_: _dafny.Seq
                                    out16_: _dafny.Seq
                                    out17_: bool
                                    out18_: _dafny.Seq
                                    out16_, out17_, out18_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_23_cg_ = out16_
                                    d_24_ci_ = out17_
                                    d_25_cc_ = out18_
                                    generated = d_23_cg_
                                    insideConstrainedOut = d_24_ci_
                                    currentConstrainedOut = d_25_cc_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                elif True:
                                    insideConstrainedOut = False
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                    d_1_steps_ = (d_1_steps_) + (1)
                                d_2_spanSteps_ = 0
                            elif True:
                                d_26_constrainedPrompt_: _dafny.Seq
                                d_26_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_27_next_: _dafny.Seq
                                out19_: _dafny.Seq
                                out19_ = (d_0_helpers_).SafeTemperatureConstrainedStep(lm, parser, d_26_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('6e-1'), eosToken)
                                d_27_next_ = out19_
                                d_1_steps_ = (d_1_steps_) + (1)
                                d_2_spanSteps_ = (d_2_spanSteps_) + (1)
                                if (d_27_next_) == (eosToken):
                                    d_28_isNowComplete_: bool
                                    d_28_isNowComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                                    if (d_28_isNowComplete_) and ((d_1_steps_) < (maxSteps)):
                                        d_29_cg_: _dafny.Seq
                                        d_30_ci_: bool
                                        d_31_cc_: _dafny.Seq
                                        out20_: _dafny.Seq
                                        out21_: bool
                                        out22_: _dafny.Seq
                                        out20_, out21_, out22_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                        d_29_cg_ = out20_
                                        d_30_ci_ = out21_
                                        d_31_cc_ = out22_
                                        generated = d_29_cg_
                                        insideConstrainedOut = d_30_ci_
                                        currentConstrainedOut = d_31_cc_
                                        d_1_steps_ = (d_1_steps_) + (1)
                                    elif True:
                                        d_32_rg_: _dafny.Seq
                                        d_33_rc_: _dafny.Seq
                                        out23_: _dafny.Seq
                                        out24_: _dafny.Seq
                                        out23_, out24_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                                        d_32_rg_ = out23_
                                        d_33_rc_ = out24_
                                        generated = d_32_rg_
                                        currentConstrainedOut = d_33_rc_
                                        d_34_isRbComplete_: bool
                                        d_34_isRbComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                                        if (d_34_isRbComplete_) and ((d_1_steps_) < (maxSteps)):
                                            d_35_cg_: _dafny.Seq
                                            d_36_ci_: bool
                                            d_37_cc_: _dafny.Seq
                                            out25_: _dafny.Seq
                                            out26_: bool
                                            out27_: _dafny.Seq
                                            out25_, out26_, out27_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                            d_35_cg_ = out25_
                                            d_36_ci_ = out26_
                                            d_37_cc_ = out27_
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
                                    out28_: _dafny.Seq
                                    out29_: bool
                                    out30_: _dafny.Seq
                                    out28_, out29_, out30_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_27_next_)
                                    d_38_ag_ = out28_
                                    d_39_ai_ = out29_
                                    d_40_ac_ = out30_
                                    generated = d_38_ag_
                                    insideConstrainedOut = d_39_ai_
                                    currentConstrainedOut = d_40_ac_
                                    d_41_nowComplete_: bool
                                    d_41_nowComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                                    if (d_41_nowComplete_) and ((d_1_steps_) < (maxSteps)):
                                        d_42_cg_: _dafny.Seq
                                        d_43_ci_: bool
                                        d_44_cc_: _dafny.Seq
                                        out31_: _dafny.Seq
                                        out32_: bool
                                        out33_: _dafny.Seq
                                        out31_, out32_, out33_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                        d_42_cg_ = out31_
                                        d_43_ci_ = out32_
                                        d_44_cc_ = out33_
                                        generated = d_42_cg_
                                        insideConstrainedOut = d_43_ci_
                                        currentConstrainedOut = d_44_cc_
                                        d_1_steps_ = (d_1_steps_) + (1)
                                        d_2_spanSteps_ = 0
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

