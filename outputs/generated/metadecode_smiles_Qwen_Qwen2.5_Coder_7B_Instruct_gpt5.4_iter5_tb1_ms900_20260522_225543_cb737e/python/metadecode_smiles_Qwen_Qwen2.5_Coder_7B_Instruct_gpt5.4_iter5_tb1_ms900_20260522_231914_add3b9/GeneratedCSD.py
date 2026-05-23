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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Produce exactly one valid SMILES string for the requested molecular class. Use a short natural lead-in before the molecule if helpful, then give one final molecule in a single constrained span. Prefer chemically plausible, diverse, non-repetitive SMILES.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_usedChunkPrelude_: bool
        d_2_usedChunkPrelude_ = False
        d_3_chunkPreludeBudget_: int
        if (maxSteps) < (8):
            d_3_chunkPreludeBudget_ = maxSteps
        elif True:
            d_3_chunkPreludeBudget_ = 8
        d_4_narrowThreshold_: int
        d_4_narrowThreshold_ = 12
        d_5_repeatThreshold_: int
        d_5_repeatThreshold_ = 2
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if not(d_2_usedChunkPrelude_):
                            d_6_remainingPrelude_: int
                            if (d_3_chunkPreludeBudget_) <= ((maxSteps) - (d_1_steps_)):
                                d_6_remainingPrelude_ = d_3_chunkPreludeBudget_
                            elif True:
                                d_6_remainingPrelude_ = (maxSteps) - (d_1_steps_)
                            d_7_chunkedGenerated_: _dafny.Seq
                            d_8_stoppedOnOpenSpan_: bool
                            d_9_stoppedOnEos_: bool
                            d_10_stepsUsed_: int
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: bool
                            out3_: int
                            out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_6_remainingPrelude_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_7_chunkedGenerated_ = out0_
                            d_8_stoppedOnOpenSpan_ = out1_
                            d_9_stoppedOnEos_ = out2_
                            d_10_stepsUsed_ = out3_
                            generated = d_7_chunkedGenerated_
                            d_2_usedChunkPrelude_ = True
                            d_1_steps_ = (d_1_steps_) + (d_10_stepsUsed_)
                            if d_9_stoppedOnEos_:
                                raise _dafny.Break("0")
                            elif d_8_stoppedOnOpenSpan_:
                                d_11_observedGenerated_: _dafny.Seq
                                d_12_observedInside_: bool
                                d_13_observedCurrent_: _dafny.Seq
                                out4_: _dafny.Seq
                                out5_: bool
                                out6_: _dafny.Seq
                                out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_11_observedGenerated_ = out4_
                                d_12_observedInside_ = out5_
                                d_13_observedCurrent_ = out6_
                                generated = d_11_observedGenerated_
                                insideConstrainedOut = d_12_observedInside_
                                currentConstrainedOut = d_13_observedCurrent_
                        elif True:
                            d_14_openedGenerated_: _dafny.Seq
                            d_15_openedInside_: bool
                            d_16_openedCurrent_: _dafny.Seq
                            out7_: _dafny.Seq
                            out8_: bool
                            out9_: _dafny.Seq
                            out7_, out8_, out9_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_14_openedGenerated_ = out7_
                            d_15_openedInside_ = out8_
                            d_16_openedCurrent_ = out9_
                            generated = d_14_openedGenerated_
                            insideConstrainedOut = d_15_openedInside_
                            currentConstrainedOut = d_16_openedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
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
                    elif True:
                        d_20_stablePrefix_: _dafny.Seq
                        d_20_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_21_constrainedPrompt_: _dafny.Seq
                        d_21_constrainedPrompt_ = (prompt) + (d_20_stablePrefix_)
                        d_22_nextConstrained_: _dafny.Seq
                        d_22_nextConstrained_ = eosToken
                        d_23_repeatedCount_: int
                        d_23_repeatedCount_ = 0
                        if (len(currentConstrainedOut)) > (0):
                            out13_: int
                            out13_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(currentConstrainedOut, (currentConstrainedOut)[(len(currentConstrainedOut)) - (1)])
                            d_23_repeatedCount_ = out13_
                        if ((len(currentConstrainedOut)) > (0)) and ((d_23_repeatedCount_) >= (d_5_repeatThreshold_)):
                            out14_: _dafny.Seq
                            out14_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_21_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('2e0'), eosToken)
                            d_22_nextConstrained_ = out14_
                        elif True:
                            d_24_validCount_: int
                            out15_: int
                            out15_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                            d_24_validCount_ = out15_
                            if (d_24_validCount_) <= (4):
                                out16_: _dafny.Seq
                                out16_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, d_21_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), eosToken)
                                d_22_nextConstrained_ = out16_
                            elif (d_24_validCount_) <= (d_4_narrowThreshold_):
                                out17_: _dafny.Seq
                                out17_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_21_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                                d_22_nextConstrained_ = out17_
                            elif True:
                                d_25_nextSoft_: _dafny.Seq
                                d_26_usedFallback_: bool
                                out18_: _dafny.Seq
                                out19_: bool
                                out18_, out19_ = (d_0_helpers_).SafeSoftConstrainedStep(lm, parser, d_21_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('8e0'), eosToken)
                                d_25_nextSoft_ = out18_
                                d_26_usedFallback_ = out19_
                                d_22_nextConstrained_ = d_25_nextSoft_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_22_nextConstrained_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_27_appendedGenerated_: _dafny.Seq
                            d_28_appendedInside_: bool
                            d_29_appendedCurrent_: _dafny.Seq
                            out20_: _dafny.Seq
                            out21_: bool
                            out22_: _dafny.Seq
                            out20_, out21_, out22_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_22_nextConstrained_)
                            d_27_appendedGenerated_ = out20_
                            d_28_appendedInside_ = out21_
                            d_29_appendedCurrent_ = out22_
                            generated = d_27_appendedGenerated_
                            insideConstrainedOut = d_28_appendedInside_
                            currentConstrainedOut = d_29_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

