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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math problem step by step. Wrap every arithmetic expression and the final answer in << >> delimiters. Example: <<3+4=7>>. End with #### <<answer>>.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_spanSteps_: int
        d_2_spanSteps_ = 0
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_2_spanSteps_ = 0
                        d_3_chunkBudget_: int
                        if ((maxSteps) - (d_1_steps_)) < (25):
                            d_3_chunkBudget_ = (maxSteps) - (d_1_steps_)
                        elif True:
                            d_3_chunkBudget_ = 25
                        if (d_3_chunkBudget_) == (0):
                            raise _dafny.Break("0")
                        d_4_generatedOut_: _dafny.Seq
                        d_5_stoppedOnOpenSpan_: bool
                        d_6_stoppedOnEos_: bool
                        d_7_stepsUsed_: int
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: bool
                        out3_: int
                        out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_3_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_4_generatedOut_ = out0_
                        d_5_stoppedOnOpenSpan_ = out1_
                        d_6_stoppedOnEos_ = out2_
                        d_7_stepsUsed_ = out3_
                        generated = d_4_generatedOut_
                        d_1_steps_ = (d_1_steps_) + (d_7_stepsUsed_)
                        if d_6_stoppedOnEos_:
                            raise _dafny.Break("0")
                        elif d_5_stoppedOnOpenSpan_:
                            d_8_g2_: _dafny.Seq
                            d_9_ins2_: bool
                            d_10_cur2_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                            d_8_g2_ = out4_
                            d_9_ins2_ = out5_
                            d_10_cur2_ = out6_
                            generated = d_8_g2_
                            insideConstrainedOut = d_9_ins2_
                            currentConstrainedOut = d_10_cur2_
                        elif True:
                            if ((len(generated)) > (0)) and (((generated)[(len(generated)) - (1)]) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))):
                                d_11_g2_: _dafny.Seq
                                d_12_ins2_: bool
                                d_13_cur2_: _dafny.Seq
                                out7_: _dafny.Seq
                                out8_: bool
                                out9_: _dafny.Seq
                                out7_, out8_, out9_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_11_g2_ = out7_
                                d_12_ins2_ = out8_
                                d_13_cur2_ = out9_
                                generated = d_11_g2_
                                insideConstrainedOut = d_12_ins2_
                                currentConstrainedOut = d_13_cur2_
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_14_closedGenerated_: _dafny.Seq
                        d_15_closedInside_: bool
                        d_16_closedCurrent_: _dafny.Seq
                        out10_: _dafny.Seq
                        out11_: bool
                        out12_: _dafny.Seq
                        out10_, out11_, out12_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_14_closedGenerated_ = out10_
                        d_15_closedInside_ = out11_
                        d_16_closedCurrent_ = out12_
                        generated = d_14_closedGenerated_
                        insideConstrainedOut = d_15_closedInside_
                        currentConstrainedOut = d_16_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_2_spanSteps_ = 0
                    elif (d_2_spanSteps_) >= (40):
                        d_17_rolledGenerated_: _dafny.Seq
                        d_18_rolledCurrent_: _dafny.Seq
                        out13_: _dafny.Seq
                        out14_: _dafny.Seq
                        out13_, out14_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                        d_17_rolledGenerated_ = out13_
                        d_18_rolledCurrent_ = out14_
                        generated = d_17_rolledGenerated_
                        currentConstrainedOut = d_18_rolledCurrent_
                        d_2_spanSteps_ = 0
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            d_19_closedGenerated_: _dafny.Seq
                            d_20_closedInside_: bool
                            d_21_closedCurrent_: _dafny.Seq
                            out15_: _dafny.Seq
                            out16_: bool
                            out17_: _dafny.Seq
                            out15_, out16_, out17_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_19_closedGenerated_ = out15_
                            d_20_closedInside_ = out16_
                            d_21_closedCurrent_ = out17_
                            generated = d_19_closedGenerated_
                            insideConstrainedOut = d_20_closedInside_
                            currentConstrainedOut = d_21_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_22_constrainedPrompt_: _dafny.Seq
                            d_22_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_23_next_: _dafny.Seq
                            out18_: _dafny.Seq
                            out18_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_22_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_23_next_ = out18_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_2_spanSteps_ = (d_2_spanSteps_) + (1)
                            if (d_23_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_24_valid_: bool
                                out19_: bool
                                out19_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_23_next_)
                                d_24_valid_ = out19_
                                if d_24_valid_:
                                    d_25_appendedGenerated_: _dafny.Seq
                                    d_26_appendedInside_: bool
                                    d_27_appendedCurrent_: _dafny.Seq
                                    out20_: _dafny.Seq
                                    out21_: bool
                                    out22_: _dafny.Seq
                                    out20_, out21_, out22_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_23_next_)
                                    d_25_appendedGenerated_ = out20_
                                    d_26_appendedInside_ = out21_
                                    d_27_appendedCurrent_ = out22_
                                    generated = d_25_appendedGenerated_
                                    insideConstrainedOut = d_26_appendedInside_
                                    currentConstrainedOut = d_27_appendedCurrent_
                    elif True:
                        d_28_constrainedPrompt_: _dafny.Seq
                        d_28_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_29_next_: _dafny.Seq
                        d_30_usedFallback_: bool
                        out23_: _dafny.Seq
                        out24_: bool
                        out23_, out24_ = (d_0_helpers_).SafeSoftConstrainedStep(lm, parser, d_28_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('8e0'), eosToken)
                        d_29_next_ = out23_
                        d_30_usedFallback_ = out24_
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_2_spanSteps_ = (d_2_spanSteps_) + (1)
                        if (d_29_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_31_appendedGenerated_: _dafny.Seq
                            d_32_appendedInside_: bool
                            d_33_appendedCurrent_: _dafny.Seq
                            out25_: _dafny.Seq
                            out26_: bool
                            out27_: _dafny.Seq
                            out25_, out26_, out27_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_29_next_)
                            d_31_appendedGenerated_ = out25_
                            d_32_appendedInside_ = out26_
                            d_33_appendedCurrent_ = out27_
                            generated = d_31_appendedGenerated_
                            insideConstrainedOut = d_32_appendedInside_
                            currentConstrainedOut = d_33_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

