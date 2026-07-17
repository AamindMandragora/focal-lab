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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve this math word problem. Show your reasoning. At the end, output the final numeric answer as a single arithmetic expression using ONLY numbers and operators (+, -, *, /, (, )). Put it between << and >>. Use ONLY numbers, not variable names.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_answerPhase_: bool
        d_2_answerPhase_ = False
        if insideConstrained:
            d_2_answerPhase_ = True
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    d_3_budgetLeft_: int
                    d_3_budgetLeft_ = (maxSteps) - (d_1_steps_)
                    if not(insideConstrainedOut):
                        if d_2_answerPhase_:
                            if (d_3_budgetLeft_) >= (2):
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
                                d_7_next_: _dafny.Seq
                                out3_: _dafny.Seq
                                out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                                d_7_next_ = out3_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_7_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_7_next_]))
                                    if (d_7_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                        d_8_og_: _dafny.Seq
                                        d_9_oi_: bool
                                        d_10_oc_: _dafny.Seq
                                        out4_: _dafny.Seq
                                        out5_: bool
                                        out6_: _dafny.Seq
                                        out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                        d_8_og_ = out4_
                                        d_9_oi_ = out5_
                                        d_10_oc_ = out6_
                                        generated = d_8_og_
                                        insideConstrainedOut = d_9_oi_
                                        currentConstrainedOut = d_10_oc_
                        elif True:
                            if (d_3_budgetLeft_) <= (50):
                                d_2_answerPhase_ = True
                                d_11_next_: _dafny.Seq
                                out7_: _dafny.Seq
                                out7_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                                d_11_next_ = out7_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_11_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_11_next_]))
                                    if (d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                        d_12_og_: _dafny.Seq
                                        d_13_oi_: bool
                                        d_14_oc_: _dafny.Seq
                                        out8_: _dafny.Seq
                                        out9_: bool
                                        out10_: _dafny.Seq
                                        out8_, out9_, out10_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                        d_12_og_ = out8_
                                        d_13_oi_ = out9_
                                        d_14_oc_ = out10_
                                        generated = d_12_og_
                                        insideConstrainedOut = d_13_oi_
                                        currentConstrainedOut = d_14_oc_
                            elif True:
                                d_15_chunkSize_: int
                                d_15_chunkSize_ = 20
                                if ((d_3_budgetLeft_) - (50)) < (d_15_chunkSize_):
                                    d_15_chunkSize_ = (d_3_budgetLeft_) - (50)
                                if (d_15_chunkSize_) == (0):
                                    d_15_chunkSize_ = 1
                                d_16_generatedOut_: _dafny.Seq
                                d_17_stoppedOnOpenSpan_: bool
                                d_18_stoppedOnEos_: bool
                                d_19_stepsUsed_: int
                                out11_: _dafny.Seq
                                out12_: bool
                                out13_: bool
                                out14_: int
                                out11_, out12_, out13_, out14_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_15_chunkSize_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                                d_16_generatedOut_ = out11_
                                d_17_stoppedOnOpenSpan_ = out12_
                                d_18_stoppedOnEos_ = out13_
                                d_19_stepsUsed_ = out14_
                                generated = d_16_generatedOut_
                                d_1_steps_ = (d_1_steps_) + (d_19_stepsUsed_)
                                if d_18_stoppedOnEos_:
                                    raise _dafny.Break("0")
                                elif d_17_stoppedOnOpenSpan_:
                                    d_20_og_: _dafny.Seq
                                    d_21_oi_: bool
                                    d_22_oc_: _dafny.Seq
                                    out15_: _dafny.Seq
                                    out16_: bool
                                    out17_: _dafny.Seq
                                    out15_, out16_, out17_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                    d_20_og_ = out15_
                                    d_21_oi_ = out16_
                                    d_22_oc_ = out17_
                                    generated = d_20_og_
                                    insideConstrainedOut = d_21_oi_
                                    currentConstrainedOut = d_22_oc_
                                    d_2_answerPhase_ = True
                    elif True:
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            d_23_cg_: _dafny.Seq
                            d_24_ci_: bool
                            d_25_cc_: _dafny.Seq
                            out18_: _dafny.Seq
                            out19_: bool
                            out20_: _dafny.Seq
                            out18_, out19_, out20_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_23_cg_ = out18_
                            d_24_ci_ = out19_
                            d_25_cc_ = out20_
                            generated = d_23_cg_
                            insideConstrainedOut = d_24_ci_
                            currentConstrainedOut = d_25_cc_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_2_answerPhase_ = False
                            raise _dafny.Break("0")
                        elif (d_3_budgetLeft_) <= (1):
                            d_26_cg_: _dafny.Seq
                            d_27_ci_: bool
                            d_28_cc_: _dafny.Seq
                            d_29_closed_: bool
                            out21_: _dafny.Seq
                            out22_: bool
                            out23_: _dafny.Seq
                            out24_: bool
                            out21_, out22_, out23_, out24_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                            d_26_cg_ = out21_
                            d_27_ci_ = out22_
                            d_28_cc_ = out23_
                            d_29_closed_ = out24_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if d_29_closed_:
                                generated = d_26_cg_
                                insideConstrainedOut = d_27_ci_
                                currentConstrainedOut = d_28_cc_
                            elif True:
                                d_30_rg_: _dafny.Seq
                                d_31_rc_: _dafny.Seq
                                out25_: _dafny.Seq
                                out26_: _dafny.Seq
                                out25_, out26_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                                d_30_rg_ = out25_
                                d_31_rc_ = out26_
                                generated = d_30_rg_
                                currentConstrainedOut = d_31_rc_
                                if (parser).IsCompletePrefix(currentConstrainedOut):
                                    insideConstrainedOut = False
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                elif True:
                                    insideConstrainedOut = False
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                            raise _dafny.Break("0")
                        elif True:
                            d_32_constrainedPrompt_: _dafny.Seq
                            d_32_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_33_next_: _dafny.Seq
                            out27_: _dafny.Seq
                            out27_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_32_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('2e0'), eosToken)
                            d_33_next_ = out27_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_33_next_) == (eosToken):
                                d_34_rg_: _dafny.Seq
                                d_35_rc_: _dafny.Seq
                                out28_: _dafny.Seq
                                out29_: _dafny.Seq
                                out28_, out29_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                                d_34_rg_ = out28_
                                d_35_rc_ = out29_
                                generated = d_34_rg_
                                currentConstrainedOut = d_35_rc_
                                if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                    d_36_cg_: _dafny.Seq
                                    d_37_ci_: bool
                                    d_38_cc_: _dafny.Seq
                                    out30_: _dafny.Seq
                                    out31_: bool
                                    out32_: _dafny.Seq
                                    out30_, out31_, out32_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_36_cg_ = out30_
                                    d_37_ci_ = out31_
                                    d_38_cc_ = out32_
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
                                out33_: _dafny.Seq
                                out34_: bool
                                out35_: _dafny.Seq
                                out33_, out34_, out35_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_33_next_)
                                d_39_ag_ = out33_
                                d_40_ai_ = out34_
                                d_41_ac_ = out35_
                                generated = d_39_ag_
                                insideConstrainedOut = d_40_ai_
                                currentConstrainedOut = d_41_ac_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

