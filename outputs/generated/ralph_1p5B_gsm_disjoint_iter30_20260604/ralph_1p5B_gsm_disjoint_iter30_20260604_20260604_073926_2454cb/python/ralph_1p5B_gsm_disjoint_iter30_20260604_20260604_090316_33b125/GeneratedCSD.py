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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. For each calculation, wrap the symbolic expression in << >>. Use only variable names without curly braces inside << >>. The final answer must be inside << >>.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_spanSteps_: int
        d_2_spanSteps_ = 0
        d_3_maxSpanSteps_: int
        d_3_maxSpanSteps_ = 15
        d_4_chunkBudget_: int
        d_4_chunkBudget_ = 6
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_5_remaining_: int
                        d_5_remaining_ = (maxSteps) - (d_1_steps_)
                        if ((d_5_remaining_) <= (12)) and ((d_5_remaining_) >= (2)):
                            d_6_g2_: _dafny.Seq
                            d_7_inside2_: bool
                            d_8_current2_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_6_g2_ = out0_
                            d_7_inside2_ = out1_
                            d_8_current2_ = out2_
                            generated = d_6_g2_
                            insideConstrainedOut = d_7_inside2_
                            currentConstrainedOut = d_8_current2_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_2_spanSteps_ = 0
                        elif True:
                            d_9_actualChunk_: int
                            if (d_5_remaining_) < (d_4_chunkBudget_):
                                d_9_actualChunk_ = d_5_remaining_
                            elif True:
                                d_9_actualChunk_ = d_4_chunkBudget_
                            if (d_9_actualChunk_) == (0):
                                raise _dafny.Break("0")
                            d_10_chunkGenerated_: _dafny.Seq
                            d_11_stoppedOnOpenSpan_: bool
                            d_12_stoppedOnEos_: bool
                            d_13_stepsUsed_: int
                            out3_: _dafny.Seq
                            out4_: bool
                            out5_: bool
                            out6_: int
                            out3_, out4_, out5_, out6_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_9_actualChunk_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_10_chunkGenerated_ = out3_
                            d_11_stoppedOnOpenSpan_ = out4_
                            d_12_stoppedOnEos_ = out5_
                            d_13_stepsUsed_ = out6_
                            generated = d_10_chunkGenerated_
                            d_1_steps_ = (d_1_steps_) + (d_13_stepsUsed_)
                            if d_12_stoppedOnEos_:
                                raise _dafny.Break("0")
                            elif d_11_stoppedOnOpenSpan_:
                                d_14_g2_: _dafny.Seq
                                d_15_inside2_: bool
                                d_16_current2_: _dafny.Seq
                                out7_: _dafny.Seq
                                out8_: bool
                                out9_: _dafny.Seq
                                out7_, out8_, out9_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_14_g2_ = out7_
                                d_15_inside2_ = out8_
                                d_16_current2_ = out9_
                                generated = d_14_g2_
                                insideConstrainedOut = d_15_inside2_
                                currentConstrainedOut = d_16_current2_
                                d_2_spanSteps_ = 0
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_17_closedGenerated_: _dafny.Seq
                        d_18_closedInside_: bool
                        d_19_closedCurrent_: _dafny.Seq
                        out10_: _dafny.Seq
                        out11_: bool
                        out12_: _dafny.Seq
                        out10_, out11_, out12_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_17_closedGenerated_ = out10_
                        d_18_closedInside_ = out11_
                        d_19_closedCurrent_ = out12_
                        generated = d_17_closedGenerated_
                        insideConstrainedOut = d_18_closedInside_
                        currentConstrainedOut = d_19_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_2_spanSteps_ = 0
                    elif (d_2_spanSteps_) >= (d_3_maxSpanSteps_):
                        d_20_rolledGenerated_: _dafny.Seq
                        d_21_rolledCurrent_: _dafny.Seq
                        out13_: _dafny.Seq
                        out14_: _dafny.Seq
                        out13_, out14_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                        d_20_rolledGenerated_ = out13_
                        d_21_rolledCurrent_ = out14_
                        if (len(d_20_rolledGenerated_)) > (0):
                            generated = _dafny.SeqWithoutIsStrInference((d_20_rolledGenerated_)[:(len(d_20_rolledGenerated_)) - (1):])
                        elif True:
                            generated = d_20_rolledGenerated_
                        insideConstrainedOut = False
                        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_2_spanSteps_ = 0
                    elif True:
                        d_22_validCount_: int
                        out15_: int
                        out15_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                        d_22_validCount_ = out15_
                        if (d_22_validCount_) == (0):
                            d_23_rolledGenerated_: _dafny.Seq
                            d_24_rolledCurrent_: _dafny.Seq
                            out16_: _dafny.Seq
                            out17_: _dafny.Seq
                            out16_, out17_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                            d_23_rolledGenerated_ = out16_
                            d_24_rolledCurrent_ = out17_
                            if (len(d_23_rolledGenerated_)) > (0):
                                generated = _dafny.SeqWithoutIsStrInference((d_23_rolledGenerated_)[:(len(d_23_rolledGenerated_)) - (1):])
                            elif True:
                                generated = d_23_rolledGenerated_
                            insideConstrainedOut = False
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_2_spanSteps_ = 0
                        elif True:
                            d_25_constrainedPrompt_: _dafny.Seq
                            d_25_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_26_next_: _dafny.Seq
                            d_27_wasConstrained_: bool
                            out18_: _dafny.Seq
                            out19_: bool
                            out18_, out19_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, d_25_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_26_next_ = out18_
                            d_27_wasConstrained_ = out19_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_2_spanSteps_ = (d_2_spanSteps_) + (1)
                            if (d_26_next_) == (eosToken):
                                d_28_rolledGenerated_: _dafny.Seq
                                d_29_rolledCurrent_: _dafny.Seq
                                out20_: _dafny.Seq
                                out21_: _dafny.Seq
                                out20_, out21_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                d_28_rolledGenerated_ = out20_
                                d_29_rolledCurrent_ = out21_
                                if (len(d_28_rolledGenerated_)) > (0):
                                    generated = _dafny.SeqWithoutIsStrInference((d_28_rolledGenerated_)[:(len(d_28_rolledGenerated_)) - (1):])
                                elif True:
                                    generated = d_28_rolledGenerated_
                                insideConstrainedOut = False
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                d_2_spanSteps_ = 0
                                raise _dafny.Break("0")
                            elif True:
                                d_30_valid_: bool
                                out22_: bool
                                out22_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_26_next_)
                                d_30_valid_ = out22_
                                if d_30_valid_:
                                    d_31_appendedGenerated_: _dafny.Seq
                                    d_32_appendedInside_: bool
                                    d_33_appendedCurrent_: _dafny.Seq
                                    out23_: _dafny.Seq
                                    out24_: bool
                                    out25_: _dafny.Seq
                                    out23_, out24_, out25_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_26_next_)
                                    d_31_appendedGenerated_ = out23_
                                    d_32_appendedInside_ = out24_
                                    d_33_appendedCurrent_ = out25_
                                    generated = d_31_appendedGenerated_
                                    insideConstrainedOut = d_32_appendedInside_
                                    currentConstrainedOut = d_33_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

