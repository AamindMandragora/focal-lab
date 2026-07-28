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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step. Show your work clearly. Compute the final numeric answer.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_freeChunkSize_: int
        d_2_freeChunkSize_ = 100
        d_3_spanCount_: int
        d_3_spanCount_ = 0
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_4_remainingSteps_: int
                        d_4_remainingSteps_ = (maxSteps) - (d_1_steps_)
                        d_5_chunkSize_: int
                        d_5_chunkSize_ = d_2_freeChunkSize_
                        if (d_5_chunkSize_) > (d_4_remainingSteps_):
                            d_5_chunkSize_ = d_4_remainingSteps_
                        if (d_5_chunkSize_) == (0):
                            raise _dafny.Break("0")
                        d_6_chunkGenerated_: _dafny.Seq
                        d_7_stoppedOnOpenSpan_: bool
                        d_8_stoppedOnEos_: bool
                        d_9_stepsUsed_: int
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: bool
                        out3_: int
                        out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_5_chunkSize_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_6_chunkGenerated_ = out0_
                        d_7_stoppedOnOpenSpan_ = out1_
                        d_8_stoppedOnEos_ = out2_
                        d_9_stepsUsed_ = out3_
                        generated = d_6_chunkGenerated_
                        d_1_steps_ = (d_1_steps_) + (d_9_stepsUsed_)
                        if d_8_stoppedOnEos_:
                            raise _dafny.Break("0")
                        if d_7_stoppedOnOpenSpan_:
                            d_10_enterGenerated_: _dafny.Seq
                            d_11_enterInside_: bool
                            d_12_enterCurrent_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                            d_10_enterGenerated_ = out4_
                            d_11_enterInside_ = out5_
                            d_12_enterCurrent_ = out6_
                            generated = d_10_enterGenerated_
                            insideConstrainedOut = d_11_enterInside_
                            currentConstrainedOut = d_12_enterCurrent_
                            d_3_spanCount_ = (d_3_spanCount_) + (1)
                        elif True:
                            if ((d_1_steps_) + (2)) <= (maxSteps):
                                d_13_openGenerated_: _dafny.Seq
                                d_14_openInside_: bool
                                d_15_openCurrent_: _dafny.Seq
                                out7_: _dafny.Seq
                                out8_: bool
                                out9_: _dafny.Seq
                                out7_, out8_, out9_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                d_13_openGenerated_ = out7_
                                d_14_openInside_ = out8_
                                d_15_openCurrent_ = out9_
                                generated = d_13_openGenerated_
                                insideConstrainedOut = d_14_openInside_
                                currentConstrainedOut = d_15_openCurrent_
                                d_1_steps_ = (d_1_steps_) + (1)
                                d_3_spanCount_ = (d_3_spanCount_) + (1)
                            elif True:
                                raise _dafny.Break("0")
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
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_19_isDeadEnd_: bool
                        out13_: bool
                        out13_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                        d_19_isDeadEnd_ = out13_
                        if d_19_isDeadEnd_:
                            d_20_rolledGenerated_: _dafny.Seq
                            d_21_rolledCurrent_: _dafny.Seq
                            out14_: _dafny.Seq
                            out15_: _dafny.Seq
                            out14_, out15_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                            d_20_rolledGenerated_ = out14_
                            d_21_rolledCurrent_ = out15_
                            generated = d_20_rolledGenerated_
                            currentConstrainedOut = d_21_rolledCurrent_
                            if (parser).IsDeadPrefix(currentConstrainedOut):
                                d_22_constrainedPrompt_: _dafny.Seq
                                d_22_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_23_next_: _dafny.Seq
                                out16_: _dafny.Seq
                                out16_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_22_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_23_next_ = out16_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_23_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_24_valid_: bool
                                    out17_: bool
                                    out17_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_23_next_)
                                    d_24_valid_ = out17_
                                    if d_24_valid_:
                                        d_25_appendedGenerated_: _dafny.Seq
                                        d_26_appendedInside_: bool
                                        d_27_appendedCurrent_: _dafny.Seq
                                        out18_: _dafny.Seq
                                        out19_: bool
                                        out20_: _dafny.Seq
                                        out18_, out19_, out20_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_23_next_)
                                        d_25_appendedGenerated_ = out18_
                                        d_26_appendedInside_ = out19_
                                        d_27_appendedCurrent_ = out20_
                                        generated = d_25_appendedGenerated_
                                        insideConstrainedOut = d_26_appendedInside_
                                        currentConstrainedOut = d_27_appendedCurrent_
                            elif True:
                                d_28_constrainedPrompt_: _dafny.Seq
                                d_28_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_29_next_: _dafny.Seq
                                out21_: _dafny.Seq
                                out21_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_28_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_29_next_ = out21_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_29_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_30_valid_: bool
                                    out22_: bool
                                    out22_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_29_next_)
                                    d_30_valid_ = out22_
                                    if d_30_valid_:
                                        d_31_appendedGenerated_: _dafny.Seq
                                        d_32_appendedInside_: bool
                                        d_33_appendedCurrent_: _dafny.Seq
                                        out23_: _dafny.Seq
                                        out24_: bool
                                        out25_: _dafny.Seq
                                        out23_, out24_, out25_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_29_next_)
                                        d_31_appendedGenerated_ = out23_
                                        d_32_appendedInside_ = out24_
                                        d_33_appendedCurrent_ = out25_
                                        generated = d_31_appendedGenerated_
                                        insideConstrainedOut = d_32_appendedInside_
                                        currentConstrainedOut = d_33_appendedCurrent_
                        elif True:
                            d_34_constrainedPrompt_: _dafny.Seq
                            d_34_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_35_next_: _dafny.Seq
                            out26_: _dafny.Seq
                            out26_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_34_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_35_next_ = out26_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_35_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_36_valid_: bool
                                out27_: bool
                                out27_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_35_next_)
                                d_36_valid_ = out27_
                                if d_36_valid_:
                                    d_37_appendedGenerated_: _dafny.Seq
                                    d_38_appendedInside_: bool
                                    d_39_appendedCurrent_: _dafny.Seq
                                    out28_: _dafny.Seq
                                    out29_: bool
                                    out30_: _dafny.Seq
                                    out28_, out29_, out30_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_35_next_)
                                    d_37_appendedGenerated_ = out28_
                                    d_38_appendedInside_ = out29_
                                    d_39_appendedCurrent_ = out30_
                                    generated = d_37_appendedGenerated_
                                    insideConstrainedOut = d_38_appendedInside_
                                    currentConstrainedOut = d_39_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

