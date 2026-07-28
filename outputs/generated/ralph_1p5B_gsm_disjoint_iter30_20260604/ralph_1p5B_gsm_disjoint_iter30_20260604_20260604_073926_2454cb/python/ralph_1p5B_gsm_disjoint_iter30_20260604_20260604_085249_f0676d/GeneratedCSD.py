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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. For each calculation, write the symbolic expression inside << >>. The final answer must also be inside << >>. Keep expressions concise.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_spanSteps_: int
        d_2_spanSteps_ = 0
        d_3_maxSpanSteps_: int
        d_3_maxSpanSteps_ = 10
        d_4_chunkBudget_: int
        d_4_chunkBudget_ = 8
        d_5_penaltyTokens_: _dafny.Seq
        d_5_penaltyTokens_ = _dafny.SeqWithoutIsStrInference([])
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_6_remaining_: int
                        d_6_remaining_ = (maxSteps) - (d_1_steps_)
                        if ((d_6_remaining_) <= (15)) and ((d_6_remaining_) >= (2)):
                            d_7_g2_: _dafny.Seq
                            d_8_inside2_: bool
                            d_9_current2_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_7_g2_ = out0_
                            d_8_inside2_ = out1_
                            d_9_current2_ = out2_
                            generated = d_7_g2_
                            insideConstrainedOut = d_8_inside2_
                            currentConstrainedOut = d_9_current2_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_2_spanSteps_ = 0
                            d_5_penaltyTokens_ = _dafny.SeqWithoutIsStrInference([])
                        elif True:
                            d_10_actualChunk_: int
                            if (d_6_remaining_) < (d_4_chunkBudget_):
                                d_10_actualChunk_ = d_6_remaining_
                            elif True:
                                d_10_actualChunk_ = d_4_chunkBudget_
                            if (d_10_actualChunk_) == (0):
                                raise _dafny.Break("0")
                            d_11_chunkGenerated_: _dafny.Seq
                            d_12_stoppedOnOpenSpan_: bool
                            d_13_stoppedOnEos_: bool
                            d_14_stepsUsed_: int
                            out3_: _dafny.Seq
                            out4_: bool
                            out5_: bool
                            out6_: int
                            out3_, out4_, out5_, out6_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_10_actualChunk_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_11_chunkGenerated_ = out3_
                            d_12_stoppedOnOpenSpan_ = out4_
                            d_13_stoppedOnEos_ = out5_
                            d_14_stepsUsed_ = out6_
                            generated = d_11_chunkGenerated_
                            d_1_steps_ = (d_1_steps_) + (d_14_stepsUsed_)
                            if d_13_stoppedOnEos_:
                                raise _dafny.Break("0")
                            elif d_12_stoppedOnOpenSpan_:
                                d_15_g2_: _dafny.Seq
                                d_16_inside2_: bool
                                d_17_current2_: _dafny.Seq
                                out7_: _dafny.Seq
                                out8_: bool
                                out9_: _dafny.Seq
                                out7_, out8_, out9_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_15_g2_ = out7_
                                d_16_inside2_ = out8_
                                d_17_current2_ = out9_
                                generated = d_15_g2_
                                insideConstrainedOut = d_16_inside2_
                                currentConstrainedOut = d_17_current2_
                                d_2_spanSteps_ = 0
                                d_5_penaltyTokens_ = _dafny.SeqWithoutIsStrInference([])
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_18_closedGenerated_: _dafny.Seq
                        d_19_closedInside_: bool
                        d_20_closedCurrent_: _dafny.Seq
                        out10_: _dafny.Seq
                        out11_: bool
                        out12_: _dafny.Seq
                        out10_, out11_, out12_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_18_closedGenerated_ = out10_
                        d_19_closedInside_ = out11_
                        d_20_closedCurrent_ = out12_
                        generated = d_18_closedGenerated_
                        insideConstrainedOut = d_19_closedInside_
                        currentConstrainedOut = d_20_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_2_spanSteps_ = 0
                        d_5_penaltyTokens_ = _dafny.SeqWithoutIsStrInference([])
                    elif (d_2_spanSteps_) >= (d_3_maxSpanSteps_):
                        d_21_rolledGenerated_: _dafny.Seq
                        d_22_rolledCurrent_: _dafny.Seq
                        out13_: _dafny.Seq
                        out14_: _dafny.Seq
                        out13_, out14_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                        d_21_rolledGenerated_ = out13_
                        d_22_rolledCurrent_ = out14_
                        if (len(d_21_rolledGenerated_)) > (0):
                            generated = _dafny.SeqWithoutIsStrInference((d_21_rolledGenerated_)[:(len(d_21_rolledGenerated_)) - (1):])
                        elif True:
                            generated = d_21_rolledGenerated_
                        insideConstrainedOut = False
                        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_2_spanSteps_ = 0
                        d_5_penaltyTokens_ = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        d_23_validCount_: int
                        out15_: int
                        out15_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                        d_23_validCount_ = out15_
                        if (d_23_validCount_) == (0):
                            d_24_rolledGenerated_: _dafny.Seq
                            d_25_rolledCurrent_: _dafny.Seq
                            out16_: _dafny.Seq
                            out17_: _dafny.Seq
                            out16_, out17_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                            d_24_rolledGenerated_ = out16_
                            d_25_rolledCurrent_ = out17_
                            if (len(d_24_rolledGenerated_)) > (0):
                                generated = _dafny.SeqWithoutIsStrInference((d_24_rolledGenerated_)[:(len(d_24_rolledGenerated_)) - (1):])
                            elif True:
                                generated = d_24_rolledGenerated_
                            insideConstrainedOut = False
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_2_spanSteps_ = 0
                            d_5_penaltyTokens_ = _dafny.SeqWithoutIsStrInference([])
                        elif True:
                            d_26_constrainedPrompt_: _dafny.Seq
                            d_26_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_27_next_: _dafny.Seq
                            out18_: _dafny.Seq
                            out18_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_26_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_5_penaltyTokens_, _dafny.BigRational('3e0'), 8, eosToken)
                            d_27_next_ = out18_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_2_spanSteps_ = (d_2_spanSteps_) + (1)
                            if (d_27_next_) == (eosToken):
                                d_28_rolledGenerated_: _dafny.Seq
                                d_29_rolledCurrent_: _dafny.Seq
                                out19_: _dafny.Seq
                                out20_: _dafny.Seq
                                out19_, out20_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                d_28_rolledGenerated_ = out19_
                                d_29_rolledCurrent_ = out20_
                                if (len(d_28_rolledGenerated_)) > (0):
                                    generated = _dafny.SeqWithoutIsStrInference((d_28_rolledGenerated_)[:(len(d_28_rolledGenerated_)) - (1):])
                                elif True:
                                    generated = d_28_rolledGenerated_
                                insideConstrainedOut = False
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                d_2_spanSteps_ = 0
                                d_5_penaltyTokens_ = _dafny.SeqWithoutIsStrInference([])
                                raise _dafny.Break("0")
                            elif True:
                                d_30_valid_: bool
                                out21_: bool
                                out21_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_27_next_)
                                d_30_valid_ = out21_
                                if d_30_valid_:
                                    d_31_appendedGenerated_: _dafny.Seq
                                    d_32_appendedInside_: bool
                                    d_33_appendedCurrent_: _dafny.Seq
                                    out22_: _dafny.Seq
                                    out23_: bool
                                    out24_: _dafny.Seq
                                    out22_, out23_, out24_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_27_next_)
                                    d_31_appendedGenerated_ = out22_
                                    d_32_appendedInside_ = out23_
                                    d_33_appendedCurrent_ = out24_
                                    generated = d_31_appendedGenerated_
                                    insideConstrainedOut = d_32_appendedInside_
                                    currentConstrainedOut = d_33_appendedCurrent_
                                    if (len(d_5_penaltyTokens_)) >= (2):
                                        d_5_penaltyTokens_ = (_dafny.SeqWithoutIsStrInference((d_5_penaltyTokens_)[1::])) + (_dafny.SeqWithoutIsStrInference([d_27_next_]))
                                    elif True:
                                        d_5_penaltyTokens_ = (d_5_penaltyTokens_) + (_dafny.SeqWithoutIsStrInference([d_27_next_]))
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

