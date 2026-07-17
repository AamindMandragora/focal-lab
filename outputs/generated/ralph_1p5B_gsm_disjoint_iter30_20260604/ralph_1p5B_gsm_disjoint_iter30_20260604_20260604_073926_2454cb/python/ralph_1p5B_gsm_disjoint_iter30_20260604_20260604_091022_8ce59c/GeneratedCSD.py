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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. For each calculation, write the symbolic expression inside << >>. Use simple variable names like n1, n2, t, d inside << >>. End with the final answer inside << >>.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_spanSteps_: int
        d_2_spanSteps_ = 0
        d_3_maxSpanSteps_: int
        d_3_maxSpanSteps_ = 20
        d_4_chunkBudget_: int
        d_4_chunkBudget_ = 5
        d_5_rollbackCount_: int
        d_5_rollbackCount_ = 0
        d_6_maxRollbacks_: int
        d_6_maxRollbacks_ = 3
        d_7_chunksWithoutSpan_: int
        d_7_chunksWithoutSpan_ = 0
        d_8_maxChunksWithoutSpan_: int
        d_8_maxChunksWithoutSpan_ = 8
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_9_remaining_: int
                        d_9_remaining_ = (maxSteps) - (d_1_steps_)
                        if ((d_7_chunksWithoutSpan_) >= (d_8_maxChunksWithoutSpan_)) and ((d_9_remaining_) >= (2)):
                            d_10_g2_: _dafny.Seq
                            d_11_inside2_: bool
                            d_12_current2_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_10_g2_ = out0_
                            d_11_inside2_ = out1_
                            d_12_current2_ = out2_
                            generated = d_10_g2_
                            insideConstrainedOut = d_11_inside2_
                            currentConstrainedOut = d_12_current2_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_2_spanSteps_ = 0
                            d_5_rollbackCount_ = 0
                            d_7_chunksWithoutSpan_ = 0
                        elif True:
                            d_13_actualChunk_: int
                            if (d_9_remaining_) < (d_4_chunkBudget_):
                                d_13_actualChunk_ = d_9_remaining_
                            elif True:
                                d_13_actualChunk_ = d_4_chunkBudget_
                            if (d_13_actualChunk_) == (0):
                                raise _dafny.Break("0")
                            d_14_chunkGenerated_: _dafny.Seq
                            d_15_stoppedOnOpenSpan_: bool
                            d_16_stoppedOnEos_: bool
                            d_17_stepsUsed_: int
                            out3_: _dafny.Seq
                            out4_: bool
                            out5_: bool
                            out6_: int
                            out3_, out4_, out5_, out6_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_13_actualChunk_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_14_chunkGenerated_ = out3_
                            d_15_stoppedOnOpenSpan_ = out4_
                            d_16_stoppedOnEos_ = out5_
                            d_17_stepsUsed_ = out6_
                            generated = d_14_chunkGenerated_
                            d_1_steps_ = (d_1_steps_) + (d_17_stepsUsed_)
                            if d_16_stoppedOnEos_:
                                raise _dafny.Break("0")
                            elif d_15_stoppedOnOpenSpan_:
                                d_18_g2_: _dafny.Seq
                                d_19_inside2_: bool
                                d_20_current2_: _dafny.Seq
                                out7_: _dafny.Seq
                                out8_: bool
                                out9_: _dafny.Seq
                                out7_, out8_, out9_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_18_g2_ = out7_
                                d_19_inside2_ = out8_
                                d_20_current2_ = out9_
                                generated = d_18_g2_
                                insideConstrainedOut = d_19_inside2_
                                currentConstrainedOut = d_20_current2_
                                d_2_spanSteps_ = 0
                                d_5_rollbackCount_ = 0
                                d_7_chunksWithoutSpan_ = 0
                            elif True:
                                d_7_chunksWithoutSpan_ = (d_7_chunksWithoutSpan_) + (1)
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_21_closedGenerated_: _dafny.Seq
                        d_22_closedInside_: bool
                        d_23_closedCurrent_: _dafny.Seq
                        out10_: _dafny.Seq
                        out11_: bool
                        out12_: _dafny.Seq
                        out10_, out11_, out12_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_21_closedGenerated_ = out10_
                        d_22_closedInside_ = out11_
                        d_23_closedCurrent_ = out12_
                        generated = d_21_closedGenerated_
                        insideConstrainedOut = d_22_closedInside_
                        currentConstrainedOut = d_23_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_2_spanSteps_ = 0
                        d_5_rollbackCount_ = 0
                        d_7_chunksWithoutSpan_ = 0
                    elif ((d_2_spanSteps_) >= (d_3_maxSpanSteps_)) or ((d_5_rollbackCount_) >= (d_6_maxRollbacks_)):
                        d_24_rolledGenerated_: _dafny.Seq
                        d_25_rolledCurrent_: _dafny.Seq
                        out13_: _dafny.Seq
                        out14_: _dafny.Seq
                        out13_, out14_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                        d_24_rolledGenerated_ = out13_
                        d_25_rolledCurrent_ = out14_
                        if (len(d_24_rolledGenerated_)) > (0):
                            generated = _dafny.SeqWithoutIsStrInference((d_24_rolledGenerated_)[:(len(d_24_rolledGenerated_)) - (1):])
                        elif True:
                            generated = d_24_rolledGenerated_
                        insideConstrainedOut = False
                        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_2_spanSteps_ = 0
                        d_5_rollbackCount_ = 0
                    elif True:
                        d_26_validCount_: int
                        out15_: int
                        out15_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                        d_26_validCount_ = out15_
                        if (d_26_validCount_) == (0):
                            d_27_rolledGenerated_: _dafny.Seq
                            d_28_rolledCurrent_: _dafny.Seq
                            out16_: _dafny.Seq
                            out17_: _dafny.Seq
                            out16_, out17_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                            d_27_rolledGenerated_ = out16_
                            d_28_rolledCurrent_ = out17_
                            if (len(d_28_rolledCurrent_)) < (len(currentConstrainedOut)):
                                generated = d_27_rolledGenerated_
                                currentConstrainedOut = d_28_rolledCurrent_
                                d_5_rollbackCount_ = (d_5_rollbackCount_) + (1)
                            elif True:
                                if (len(d_27_rolledGenerated_)) > (0):
                                    generated = _dafny.SeqWithoutIsStrInference((d_27_rolledGenerated_)[:(len(d_27_rolledGenerated_)) - (1):])
                                elif True:
                                    generated = d_27_rolledGenerated_
                                insideConstrainedOut = False
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                d_1_steps_ = (d_1_steps_) + (1)
                                d_2_spanSteps_ = 0
                                d_5_rollbackCount_ = 0
                        elif True:
                            d_29_constrainedPrompt_: _dafny.Seq
                            d_29_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_30_next_: _dafny.Seq
                            d_31_wasConstrained_: bool
                            out18_: _dafny.Seq
                            out19_: bool
                            out18_, out19_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, d_29_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_30_next_ = out18_
                            d_31_wasConstrained_ = out19_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_2_spanSteps_ = (d_2_spanSteps_) + (1)
                            if (d_30_next_) == (eosToken):
                                if (parser).IsCompletePrefix(currentConstrainedOut):
                                    d_32_closedGenerated_: _dafny.Seq
                                    d_33_closedInside_: bool
                                    d_34_closedCurrent_: _dafny.Seq
                                    out20_: _dafny.Seq
                                    out21_: bool
                                    out22_: _dafny.Seq
                                    out20_, out21_, out22_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_32_closedGenerated_ = out20_
                                    d_33_closedInside_ = out21_
                                    d_34_closedCurrent_ = out22_
                                    generated = d_32_closedGenerated_
                                    insideConstrainedOut = d_33_closedInside_
                                    currentConstrainedOut = d_34_closedCurrent_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    d_2_spanSteps_ = 0
                                    d_5_rollbackCount_ = 0
                                elif True:
                                    d_35_rolledGenerated_: _dafny.Seq
                                    d_36_rolledCurrent_: _dafny.Seq
                                    out23_: _dafny.Seq
                                    out24_: _dafny.Seq
                                    out23_, out24_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                    d_35_rolledGenerated_ = out23_
                                    d_36_rolledCurrent_ = out24_
                                    if (len(d_35_rolledGenerated_)) > (0):
                                        generated = _dafny.SeqWithoutIsStrInference((d_35_rolledGenerated_)[:(len(d_35_rolledGenerated_)) - (1):])
                                    elif True:
                                        generated = d_35_rolledGenerated_
                                    insideConstrainedOut = False
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                    d_2_spanSteps_ = 0
                                    d_5_rollbackCount_ = 0
                                raise _dafny.Break("0")
                            elif True:
                                d_37_valid_: bool
                                out25_: bool
                                out25_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_30_next_)
                                d_37_valid_ = out25_
                                if d_37_valid_:
                                    d_38_appendedGenerated_: _dafny.Seq
                                    d_39_appendedInside_: bool
                                    d_40_appendedCurrent_: _dafny.Seq
                                    out26_: _dafny.Seq
                                    out27_: bool
                                    out28_: _dafny.Seq
                                    out26_, out27_, out28_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_30_next_)
                                    d_38_appendedGenerated_ = out26_
                                    d_39_appendedInside_ = out27_
                                    d_40_appendedCurrent_ = out28_
                                    generated = d_38_appendedGenerated_
                                    insideConstrainedOut = d_39_appendedInside_
                                    currentConstrainedOut = d_40_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

