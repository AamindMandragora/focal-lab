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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step. Show each calculation inside << >> delimiters. The final answer must be in a << >> span after ####."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_spanSteps_: int
        d_3_spanSteps_ = 0
        with _dafny.label("0"):
            while (d_2_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_4_remaining_: int
                        d_4_remaining_ = (maxSteps) - (d_2_steps_)
                        d_5_chunkBudget_: int
                        if (d_4_remaining_) > (60):
                            d_5_chunkBudget_ = 60
                        elif True:
                            d_5_chunkBudget_ = d_4_remaining_
                        d_6_chunkGenerated_: _dafny.Seq
                        d_7_stoppedOnOpenSpan_: bool
                        d_8_stoppedOnEos_: bool
                        d_9_stepsUsed_: int
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: bool
                        out3_: int
                        out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_5_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_6_chunkGenerated_ = out0_
                        d_7_stoppedOnOpenSpan_ = out1_
                        d_8_stoppedOnEos_ = out2_
                        d_9_stepsUsed_ = out3_
                        generated = d_6_chunkGenerated_
                        d_2_steps_ = (d_2_steps_) + (d_9_stepsUsed_)
                        d_3_spanSteps_ = 0
                        if d_7_stoppedOnOpenSpan_:
                            d_10_eg_: _dafny.Seq
                            d_11_ei_: bool
                            d_12_ec_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                            d_10_eg_ = out4_
                            d_11_ei_ = out5_
                            d_12_ec_ = out6_
                            generated = d_10_eg_
                            insideConstrainedOut = d_11_ei_
                            currentConstrainedOut = d_12_ec_
                        elif d_8_stoppedOnEos_:
                            raise _dafny.Break("0")
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_13_closedGenerated_: _dafny.Seq
                        d_14_closedInside_: bool
                        d_15_closedCurrent_: _dafny.Seq
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: _dafny.Seq
                        out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_13_closedGenerated_ = out7_
                        d_14_closedInside_ = out8_
                        d_15_closedCurrent_ = out9_
                        generated = d_13_closedGenerated_
                        insideConstrainedOut = d_14_closedInside_
                        currentConstrainedOut = d_15_closedCurrent_
                        d_2_steps_ = (d_2_steps_) + (1)
                        d_3_spanSteps_ = 0
                    elif ((d_3_spanSteps_) > (40)) or ((((maxSteps) - (d_2_steps_)) < (15)) and (insideConstrainedOut)):
                        d_16_rolledGenerated_: _dafny.Seq
                        d_17_rolledCurrent_: _dafny.Seq
                        out10_: _dafny.Seq
                        out11_: _dafny.Seq
                        out10_, out11_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                        d_16_rolledGenerated_ = out10_
                        d_17_rolledCurrent_ = out11_
                        generated = d_16_rolledGenerated_
                        currentConstrainedOut = d_17_rolledCurrent_
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            d_18_closedGenerated_: _dafny.Seq
                            d_19_closedInside_: bool
                            d_20_closedCurrent_: _dafny.Seq
                            out12_: _dafny.Seq
                            out13_: bool
                            out14_: _dafny.Seq
                            out12_, out13_, out14_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_18_closedGenerated_ = out12_
                            d_19_closedInside_ = out13_
                            d_20_closedCurrent_ = out14_
                            generated = d_18_closedGenerated_
                            insideConstrainedOut = d_19_closedInside_
                            currentConstrainedOut = d_20_closedCurrent_
                            d_2_steps_ = (d_2_steps_) + (1)
                            d_3_spanSteps_ = 0
                        elif True:
                            d_21_rg2_: _dafny.Seq
                            d_22_rc2_: _dafny.Seq
                            out15_: _dafny.Seq
                            out16_: _dafny.Seq
                            out15_, out16_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                            d_21_rg2_ = out15_
                            d_22_rc2_ = out16_
                            generated = d_21_rg2_
                            currentConstrainedOut = d_22_rc2_
                            if (parser).IsCompletePrefix(currentConstrainedOut):
                                d_23_cg2_: _dafny.Seq
                                d_24_ci2_: bool
                                d_25_cc2_: _dafny.Seq
                                out17_: _dafny.Seq
                                out18_: bool
                                out19_: _dafny.Seq
                                out17_, out18_, out19_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_23_cg2_ = out17_
                                d_24_ci2_ = out18_
                                d_25_cc2_ = out19_
                                generated = d_23_cg2_
                                insideConstrainedOut = d_24_ci2_
                                currentConstrainedOut = d_25_cc2_
                                d_2_steps_ = (d_2_steps_) + (1)
                                d_3_spanSteps_ = 0
                            elif True:
                                if (d_2_steps_) < (maxSteps):
                                    d_26_constrainedPrompt_: _dafny.Seq
                                    d_26_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                    d_27_next_: _dafny.Seq
                                    out20_: _dafny.Seq
                                    out20_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_26_constrainedPrompt_, currentConstrainedOut, eosToken)
                                    d_27_next_ = out20_
                                    d_2_steps_ = (d_2_steps_) + (1)
                                    d_3_spanSteps_ = (d_3_spanSteps_) + (1)
                                    if (d_27_next_) == (eosToken):
                                        raise _dafny.Break("0")
                                    elif True:
                                        d_28_ag_: _dafny.Seq
                                        d_29_ai_: bool
                                        d_30_ac_: _dafny.Seq
                                        out21_: _dafny.Seq
                                        out22_: bool
                                        out23_: _dafny.Seq
                                        out21_, out22_, out23_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_27_next_)
                                        d_28_ag_ = out21_
                                        d_29_ai_ = out22_
                                        d_30_ac_ = out23_
                                        generated = d_28_ag_
                                        insideConstrainedOut = d_29_ai_
                                        currentConstrainedOut = d_30_ac_
                    elif True:
                        d_31_isDead_: bool
                        out24_: bool
                        out24_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                        d_31_isDead_ = out24_
                        if d_31_isDead_:
                            d_32_rolledGenerated_: _dafny.Seq
                            d_33_rolledCurrent_: _dafny.Seq
                            out25_: _dafny.Seq
                            out26_: _dafny.Seq
                            out25_, out26_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                            d_32_rolledGenerated_ = out25_
                            d_33_rolledCurrent_ = out26_
                            generated = d_32_rolledGenerated_
                            currentConstrainedOut = d_33_rolledCurrent_
                            if (d_2_steps_) < (maxSteps):
                                d_34_constrainedPrompt_: _dafny.Seq
                                d_34_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_35_next_: _dafny.Seq
                                out27_: _dafny.Seq
                                out27_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_34_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_35_next_ = out27_
                                d_2_steps_ = (d_2_steps_) + (1)
                                d_3_spanSteps_ = (d_3_spanSteps_) + (1)
                                if (d_35_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_36_ag_: _dafny.Seq
                                    d_37_ai_: bool
                                    d_38_ac_: _dafny.Seq
                                    out28_: _dafny.Seq
                                    out29_: bool
                                    out30_: _dafny.Seq
                                    out28_, out29_, out30_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_35_next_)
                                    d_36_ag_ = out28_
                                    d_37_ai_ = out29_
                                    d_38_ac_ = out30_
                                    generated = d_36_ag_
                                    insideConstrainedOut = d_37_ai_
                                    currentConstrainedOut = d_38_ac_
                        elif True:
                            d_39_constrainedPrompt_: _dafny.Seq
                            d_39_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_40_next_: _dafny.Seq
                            out31_: _dafny.Seq
                            out31_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_39_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_40_next_ = out31_
                            d_2_steps_ = (d_2_steps_) + (1)
                            d_3_spanSteps_ = (d_3_spanSteps_) + (1)
                            if (d_40_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_41_ag_: _dafny.Seq
                                d_42_ai_: bool
                                d_43_ac_: _dafny.Seq
                                out32_: _dafny.Seq
                                out33_: bool
                                out34_: _dafny.Seq
                                out32_, out33_, out34_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_40_next_)
                                d_41_ag_ = out32_
                                d_42_ai_ = out33_
                                d_43_ac_ = out34_
                                generated = d_41_ag_
                                insideConstrainedOut = d_42_ai_
                                currentConstrainedOut = d_43_ac_
                    pass
            pass
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

