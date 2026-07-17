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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math problem step by step. For EVERY arithmetic expression and the final answer, you MUST use << >> delimiters. Example: <<3+4=7>>. End with #### <<final_answer>>. Every << must be closed with >>.")))
        d_1_penaltyTokens_: _dafny.Seq
        d_1_penaltyTokens_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "!")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "?")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "@")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "$")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "%")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "^")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "&")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "~")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "`"))])
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_spanSteps_: int
        d_3_spanSteps_ = 0
        d_4_spanLimit_: int
        d_4_spanLimit_ = 30
        with _dafny.label("0"):
            while (d_2_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_3_spanSteps_ = 0
                        d_5_chunkBudget_: int
                        if ((maxSteps) - (d_2_steps_)) < (15):
                            d_5_chunkBudget_ = (maxSteps) - (d_2_steps_)
                        elif True:
                            d_5_chunkBudget_ = 15
                        if (d_5_chunkBudget_) == (0):
                            raise _dafny.Break("0")
                        d_6_generatedOut_: _dafny.Seq
                        d_7_stoppedOnOpenSpan_: bool
                        d_8_stoppedOnEos_: bool
                        d_9_stepsUsed_: int
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: bool
                        out3_: int
                        out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_5_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_6_generatedOut_ = out0_
                        d_7_stoppedOnOpenSpan_ = out1_
                        d_8_stoppedOnEos_ = out2_
                        d_9_stepsUsed_ = out3_
                        generated = d_6_generatedOut_
                        d_2_steps_ = (d_2_steps_) + (d_9_stepsUsed_)
                        if d_8_stoppedOnEos_:
                            raise _dafny.Break("0")
                        elif d_7_stoppedOnOpenSpan_:
                            d_10_g2_: _dafny.Seq
                            d_11_ins2_: bool
                            d_12_cur2_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                            d_10_g2_ = out4_
                            d_11_ins2_ = out5_
                            d_12_cur2_ = out6_
                            generated = d_10_g2_
                            insideConstrainedOut = d_11_ins2_
                            currentConstrainedOut = d_12_cur2_
                        elif True:
                            if ((len(generated)) > (0)) and (((generated)[(len(generated)) - (1)]) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))):
                                d_13_g2_: _dafny.Seq
                                d_14_ins2_: bool
                                d_15_cur2_: _dafny.Seq
                                out7_: _dafny.Seq
                                out8_: bool
                                out9_: _dafny.Seq
                                out7_, out8_, out9_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_13_g2_ = out7_
                                d_14_ins2_ = out8_
                                d_15_cur2_ = out9_
                                generated = d_13_g2_
                                insideConstrainedOut = d_14_ins2_
                                currentConstrainedOut = d_15_cur2_
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_16_closedGenerated_: _dafny.Seq
                        d_17_closedInside_: bool
                        d_18_closedCurrent_: _dafny.Seq
                        out10_: _dafny.Seq
                        out11_: bool
                        out12_: _dafny.Seq
                        out10_, out11_, out12_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_16_closedGenerated_ = out10_
                        d_17_closedInside_ = out11_
                        d_18_closedCurrent_ = out12_
                        generated = d_16_closedGenerated_
                        insideConstrainedOut = d_17_closedInside_
                        currentConstrainedOut = d_18_closedCurrent_
                        d_2_steps_ = (d_2_steps_) + (1)
                        d_3_spanSteps_ = 0
                    elif (d_3_spanSteps_) >= (d_4_spanLimit_):
                        d_19_rolledGenerated_: _dafny.Seq
                        d_20_rolledCurrent_: _dafny.Seq
                        out13_: _dafny.Seq
                        out14_: _dafny.Seq
                        out13_, out14_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                        d_19_rolledGenerated_ = out13_
                        d_20_rolledCurrent_ = out14_
                        generated = d_19_rolledGenerated_
                        currentConstrainedOut = d_20_rolledCurrent_
                        d_3_spanSteps_ = 0
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            d_21_closedGenerated_: _dafny.Seq
                            d_22_closedInside_: bool
                            d_23_closedCurrent_: _dafny.Seq
                            out15_: _dafny.Seq
                            out16_: bool
                            out17_: _dafny.Seq
                            out15_, out16_, out17_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_21_closedGenerated_ = out15_
                            d_22_closedInside_ = out16_
                            d_23_closedCurrent_ = out17_
                            generated = d_21_closedGenerated_
                            insideConstrainedOut = d_22_closedInside_
                            currentConstrainedOut = d_23_closedCurrent_
                            d_2_steps_ = (d_2_steps_) + (1)
                        elif True:
                            d_24_constrainedPrompt_: _dafny.Seq
                            d_24_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_25_next_: _dafny.Seq
                            out18_: _dafny.Seq
                            out18_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_24_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_25_next_ = out18_
                            d_2_steps_ = (d_2_steps_) + (1)
                            d_3_spanSteps_ = (d_3_spanSteps_) + (1)
                            if (d_25_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_26_valid_: bool
                                out19_: bool
                                out19_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_25_next_)
                                d_26_valid_ = out19_
                                if d_26_valid_:
                                    d_27_appendedGenerated_: _dafny.Seq
                                    d_28_appendedInside_: bool
                                    d_29_appendedCurrent_: _dafny.Seq
                                    out20_: _dafny.Seq
                                    out21_: bool
                                    out22_: _dafny.Seq
                                    out20_, out21_, out22_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_25_next_)
                                    d_27_appendedGenerated_ = out20_
                                    d_28_appendedInside_ = out21_
                                    d_29_appendedCurrent_ = out22_
                                    generated = d_27_appendedGenerated_
                                    insideConstrainedOut = d_28_appendedInside_
                                    currentConstrainedOut = d_29_appendedCurrent_
                    elif True:
                        d_30_narrow_: bool
                        out23_: bool
                        out23_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 2)
                        d_30_narrow_ = out23_
                        if (d_30_narrow_) and ((d_3_spanSteps_) > (10)):
                            d_31_rolledGenerated_: _dafny.Seq
                            d_32_rolledCurrent_: _dafny.Seq
                            out24_: _dafny.Seq
                            out25_: _dafny.Seq
                            out24_, out25_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                            d_31_rolledGenerated_ = out24_
                            d_32_rolledCurrent_ = out25_
                            generated = d_31_rolledGenerated_
                            currentConstrainedOut = d_32_rolledCurrent_
                            d_3_spanSteps_ = 0
                            if (parser).IsCompletePrefix(currentConstrainedOut):
                                d_33_closedGenerated_: _dafny.Seq
                                d_34_closedInside_: bool
                                d_35_closedCurrent_: _dafny.Seq
                                out26_: _dafny.Seq
                                out27_: bool
                                out28_: _dafny.Seq
                                out26_, out27_, out28_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_33_closedGenerated_ = out26_
                                d_34_closedInside_ = out27_
                                d_35_closedCurrent_ = out28_
                                generated = d_33_closedGenerated_
                                insideConstrainedOut = d_34_closedInside_
                                currentConstrainedOut = d_35_closedCurrent_
                                d_2_steps_ = (d_2_steps_) + (1)
                            elif True:
                                d_36_constrainedPrompt_: _dafny.Seq
                                d_36_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_37_next_: _dafny.Seq
                                out29_: _dafny.Seq
                                out29_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_36_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_37_next_ = out29_
                                d_2_steps_ = (d_2_steps_) + (1)
                                d_3_spanSteps_ = (d_3_spanSteps_) + (1)
                                if (d_37_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_38_valid_: bool
                                    out30_: bool
                                    out30_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_37_next_)
                                    d_38_valid_ = out30_
                                    if d_38_valid_:
                                        d_39_appendedGenerated_: _dafny.Seq
                                        d_40_appendedInside_: bool
                                        d_41_appendedCurrent_: _dafny.Seq
                                        out31_: _dafny.Seq
                                        out32_: bool
                                        out33_: _dafny.Seq
                                        out31_, out32_, out33_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_37_next_)
                                        d_39_appendedGenerated_ = out31_
                                        d_40_appendedInside_ = out32_
                                        d_41_appendedCurrent_ = out33_
                                        generated = d_39_appendedGenerated_
                                        insideConstrainedOut = d_40_appendedInside_
                                        currentConstrainedOut = d_41_appendedCurrent_
                        elif True:
                            d_42_constrainedPrompt_: _dafny.Seq
                            d_42_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_43_nextPen_: _dafny.Seq
                            out34_: _dafny.Seq
                            out34_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_42_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('8e0'), d_1_penaltyTokens_, _dafny.BigRational('15e0'), 12, eosToken)
                            d_43_nextPen_ = out34_
                            d_2_steps_ = (d_2_steps_) + (1)
                            d_3_spanSteps_ = (d_3_spanSteps_) + (1)
                            if (d_43_nextPen_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_44_appendedGenerated_: _dafny.Seq
                                d_45_appendedInside_: bool
                                d_46_appendedCurrent_: _dafny.Seq
                                out35_: _dafny.Seq
                                out36_: bool
                                out37_: _dafny.Seq
                                out35_, out36_, out37_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_43_nextPen_)
                                d_44_appendedGenerated_ = out35_
                                d_45_appendedInside_ = out36_
                                d_46_appendedCurrent_ = out37_
                                generated = d_44_appendedGenerated_
                                insideConstrainedOut = d_45_appendedInside_
                                currentConstrainedOut = d_46_appendedCurrent_
                    pass
            pass
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

