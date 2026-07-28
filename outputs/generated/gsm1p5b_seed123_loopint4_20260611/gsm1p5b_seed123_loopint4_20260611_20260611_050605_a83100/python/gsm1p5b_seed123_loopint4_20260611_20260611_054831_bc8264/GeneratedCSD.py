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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve this math word problem. Show your work step by step. At the very end, write the final numeric answer as a single arithmetic expression inside << >> with ONLY digits and operators +,-,*,/,(,). No variable names or letters inside << >>. Final line must be: <<number_expression>>")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_spanSteps_: int
        d_2_spanSteps_ = 0
        d_3_maxSpanSteps_: int
        d_3_maxSpanSteps_ = 25
        d_4_forceOpenThreshold_: int
        if (maxSteps) > (35):
            d_4_forceOpenThreshold_ = (maxSteps) - (30)
        elif True:
            d_4_forceOpenThreshold_ = _dafny.euclidian_division(maxSteps, 2)
        d_5_forcedSpanOpen_: bool
        d_5_forcedSpanOpen_ = False
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if (((d_1_steps_) >= (d_4_forceOpenThreshold_)) and (((d_1_steps_) + (5)) <= (maxSteps))) and (not(d_5_forcedSpanOpen_)):
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
                            d_2_spanSteps_ = 0
                            d_5_forcedSpanOpen_ = True
                        elif True:
                            d_9_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_9_next_ = out3_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_9_next_) == (eosToken):
                                if (((d_1_steps_) + (5)) <= (maxSteps)) and (not(d_5_forcedSpanOpen_)):
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
                                    d_2_spanSteps_ = 0
                                    d_5_forcedSpanOpen_ = True
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
                                    d_2_spanSteps_ = 0
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
                            d_2_spanSteps_ = 0
                            if d_5_forcedSpanOpen_:
                                raise _dafny.Break("0")
                        elif True:
                            d_20_validCount_: int
                            out13_: int
                            out13_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                            d_20_validCount_ = out13_
                            if (d_20_validCount_) == (0):
                                insideConstrainedOut = False
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                d_2_spanSteps_ = 0
                            elif (d_2_spanSteps_) >= (d_3_maxSpanSteps_):
                                d_21_rg_: _dafny.Seq
                                d_22_rc_: _dafny.Seq
                                out14_: _dafny.Seq
                                out15_: _dafny.Seq
                                out14_, out15_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                                d_21_rg_ = out14_
                                d_22_rc_ = out15_
                                generated = d_21_rg_
                                currentConstrainedOut = d_22_rc_
                                d_23_isNowComplete_: bool
                                d_23_isNowComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                                if (d_23_isNowComplete_) and ((d_1_steps_) < (maxSteps)):
                                    d_24_cg_: _dafny.Seq
                                    d_25_ci_: bool
                                    d_26_cc_: _dafny.Seq
                                    out16_: _dafny.Seq
                                    out17_: bool
                                    out18_: _dafny.Seq
                                    out16_, out17_, out18_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_24_cg_ = out16_
                                    d_25_ci_ = out17_
                                    d_26_cc_ = out18_
                                    generated = d_24_cg_
                                    insideConstrainedOut = d_25_ci_
                                    currentConstrainedOut = d_26_cc_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                elif True:
                                    insideConstrainedOut = False
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                    d_1_steps_ = (d_1_steps_) + (1)
                                d_2_spanSteps_ = 0
                                if d_5_forcedSpanOpen_:
                                    raise _dafny.Break("0")
                            elif True:
                                d_27_constrainedPrompt_: _dafny.Seq
                                d_27_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_28_next_: _dafny.Seq
                                out19_: _dafny.Seq
                                out19_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_27_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_28_next_ = out19_
                                d_1_steps_ = (d_1_steps_) + (1)
                                d_2_spanSteps_ = (d_2_spanSteps_) + (1)
                                if (d_28_next_) == (eosToken):
                                    d_29_isNowComplete_: bool
                                    d_29_isNowComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                                    if (d_29_isNowComplete_) and ((d_1_steps_) < (maxSteps)):
                                        d_30_cg_: _dafny.Seq
                                        d_31_ci_: bool
                                        d_32_cc_: _dafny.Seq
                                        out20_: _dafny.Seq
                                        out21_: bool
                                        out22_: _dafny.Seq
                                        out20_, out21_, out22_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                        d_30_cg_ = out20_
                                        d_31_ci_ = out21_
                                        d_32_cc_ = out22_
                                        generated = d_30_cg_
                                        insideConstrainedOut = d_31_ci_
                                        currentConstrainedOut = d_32_cc_
                                        d_1_steps_ = (d_1_steps_) + (1)
                                    elif True:
                                        d_33_rg_: _dafny.Seq
                                        d_34_rc_: _dafny.Seq
                                        out23_: _dafny.Seq
                                        out24_: _dafny.Seq
                                        out23_, out24_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                                        d_33_rg_ = out23_
                                        d_34_rc_ = out24_
                                        generated = d_33_rg_
                                        currentConstrainedOut = d_34_rc_
                                        d_35_isRbComplete_: bool
                                        d_35_isRbComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                                        if (d_35_isRbComplete_) and ((d_1_steps_) < (maxSteps)):
                                            d_36_cg_: _dafny.Seq
                                            d_37_ci_: bool
                                            d_38_cc_: _dafny.Seq
                                            out25_: _dafny.Seq
                                            out26_: bool
                                            out27_: _dafny.Seq
                                            out25_, out26_, out27_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                            d_36_cg_ = out25_
                                            d_37_ci_ = out26_
                                            d_38_cc_ = out27_
                                            generated = d_36_cg_
                                            insideConstrainedOut = d_37_ci_
                                            currentConstrainedOut = d_38_cc_
                                            d_1_steps_ = (d_1_steps_) + (1)
                                        elif True:
                                            insideConstrainedOut = False
                                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                    raise _dafny.Break("0")
                                elif True:
                                    d_39_ag_: _dafny.Seq
                                    d_40_ai_: bool
                                    d_41_ac_: _dafny.Seq
                                    out28_: _dafny.Seq
                                    out29_: bool
                                    out30_: _dafny.Seq
                                    out28_, out29_, out30_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_28_next_)
                                    d_39_ag_ = out28_
                                    d_40_ai_ = out29_
                                    d_41_ac_ = out30_
                                    generated = d_39_ag_
                                    insideConstrainedOut = d_40_ai_
                                    currentConstrainedOut = d_41_ac_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

