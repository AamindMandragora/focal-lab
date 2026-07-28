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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step. Put every arithmetic computation inside visible << and >> delimiters.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_rollbackLimit_: int
        d_2_rollbackLimit_ = 24
        d_3_forcedOpenUsed_: bool
        d_3_forcedOpenUsed_ = insideConstrained
        d_4_completeSpanCount_: int
        d_4_completeSpanCount_ = 0
        d_5_initialOpenCount_: int
        out0_: int
        out0_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(generatedPrefix, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))
        d_5_initialOpenCount_ = out0_
        d_6_initialCloseCount_: int
        out1_: int
        out1_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(generatedPrefix, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")))
        d_6_initialCloseCount_ = out1_
        if ((d_6_initialCloseCount_) > (0)) and ((d_5_initialOpenCount_) >= (d_6_initialCloseCount_)):
            d_4_completeSpanCount_ = d_6_initialCloseCount_
            d_3_forcedOpenUsed_ = True
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_7_sinceOpen_: int
                        out2_: int
                        out2_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))
                        d_7_sinceOpen_ = out2_
                        if ((not(d_3_forcedOpenUsed_)) and ((d_4_completeSpanCount_) == (0))) and ((d_7_sinceOpen_) >= (48)):
                            d_8_openedGenerated_: _dafny.Seq
                            d_9_openedInside_: bool
                            d_10_openedCurrent_: _dafny.Seq
                            out3_: _dafny.Seq
                            out4_: bool
                            out5_: _dafny.Seq
                            out3_, out4_, out5_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_8_openedGenerated_ = out3_
                            d_9_openedInside_ = out4_
                            d_10_openedCurrent_ = out5_
                            generated = d_8_openedGenerated_
                            insideConstrainedOut = d_9_openedInside_
                            currentConstrainedOut = d_10_openedCurrent_
                            d_3_forcedOpenUsed_ = True
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_11_chunkBudget_: int
                            d_11_chunkBudget_ = (maxSteps) - (d_1_steps_)
                            if (d_11_chunkBudget_) > (6):
                                d_11_chunkBudget_ = 6
                            d_12_chunkedGenerated_: _dafny.Seq
                            d_13_stoppedOnOpenSpan_: bool
                            d_14_stoppedOnEos_: bool
                            d_15_stepsUsed_: int
                            out6_: _dafny.Seq
                            out7_: bool
                            out8_: bool
                            out9_: int
                            out6_, out7_, out8_, out9_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_11_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_12_chunkedGenerated_ = out6_
                            d_13_stoppedOnOpenSpan_ = out7_
                            d_14_stoppedOnEos_ = out8_
                            d_15_stepsUsed_ = out9_
                            generated = d_12_chunkedGenerated_
                            d_1_steps_ = (d_1_steps_) + (d_15_stepsUsed_)
                            if d_14_stoppedOnEos_:
                                raise _dafny.Break("0")
                            elif d_13_stoppedOnOpenSpan_:
                                d_16_observedGenerated_: _dafny.Seq
                                d_17_observedInside_: bool
                                d_18_observedCurrent_: _dafny.Seq
                                out10_: _dafny.Seq
                                out11_: bool
                                out12_: _dafny.Seq
                                out10_, out11_, out12_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_16_observedGenerated_ = out10_
                                d_17_observedInside_ = out11_
                                d_18_observedCurrent_ = out12_
                                generated = d_16_observedGenerated_
                                insideConstrainedOut = d_17_observedInside_
                                currentConstrainedOut = d_18_observedCurrent_
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_19_closedGenerated_: _dafny.Seq
                        d_20_closedInside_: bool
                        d_21_closedCurrent_: _dafny.Seq
                        out13_: _dafny.Seq
                        out14_: bool
                        out15_: _dafny.Seq
                        out13_, out14_, out15_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_19_closedGenerated_ = out13_
                        d_20_closedInside_ = out14_
                        d_21_closedCurrent_ = out15_
                        generated = d_19_closedGenerated_
                        insideConstrainedOut = d_20_closedInside_
                        currentConstrainedOut = d_21_closedCurrent_
                        d_4_completeSpanCount_ = (d_4_completeSpanCount_) + (1)
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif (len(currentConstrainedOut)) >= (d_2_rollbackLimit_):
                        d_22_rolledGenerated_: _dafny.Seq
                        d_23_rolledCurrent_: _dafny.Seq
                        out16_: _dafny.Seq
                        out17_: _dafny.Seq
                        out16_, out17_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                        d_22_rolledGenerated_ = out16_
                        d_23_rolledCurrent_ = out17_
                        generated = d_22_rolledGenerated_
                        insideConstrainedOut = True
                        currentConstrainedOut = d_23_rolledCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_24_stablePrefix_: _dafny.Seq
                        d_24_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_25_constrainedPrompt_: _dafny.Seq
                        d_25_constrainedPrompt_ = (prompt) + (d_24_stablePrefix_)
                        d_26_nextIn_: _dafny.Seq
                        d_26_nextIn_ = eosToken
                        if (len(currentConstrainedOut)) == (0):
                            out18_: _dafny.Seq
                            out18_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_25_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                            d_26_nextIn_ = out18_
                        elif True:
                            out19_: _dafny.Seq
                            out19_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_25_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('2e0'), eosToken)
                            d_26_nextIn_ = out19_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_26_nextIn_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_27_appendedGenerated_: _dafny.Seq
                            d_28_appendedInside_: bool
                            d_29_appendedCurrent_: _dafny.Seq
                            out20_: _dafny.Seq
                            out21_: bool
                            out22_: _dafny.Seq
                            out20_, out21_, out22_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_26_nextIn_)
                            d_27_appendedGenerated_ = out20_
                            d_28_appendedInside_ = out21_
                            d_29_appendedCurrent_ = out22_
                            generated = d_27_appendedGenerated_
                            insideConstrainedOut = d_28_appendedInside_
                            currentConstrainedOut = d_29_appendedCurrent_
                    pass
            pass
        if ((insideConstrainedOut) and ((d_1_steps_) < (maxSteps))) and ((parser).IsCompletePrefix(currentConstrainedOut)):
            d_30_finalGenerated_: _dafny.Seq
            d_31_finalInside_: bool
            d_32_finalCurrent_: _dafny.Seq
            out23_: _dafny.Seq
            out24_: bool
            out25_: _dafny.Seq
            out23_, out24_, out25_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
            d_30_finalGenerated_ = out23_
            d_31_finalInside_ = out24_
            d_32_finalCurrent_ = out25_
            generated = d_30_finalGenerated_
            insideConstrainedOut = d_31_finalInside_
            currentConstrainedOut = d_32_finalCurrent_
            d_4_completeSpanCount_ = (d_4_completeSpanCount_) + (1)
            d_1_steps_ = (d_1_steps_) + (1)
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

