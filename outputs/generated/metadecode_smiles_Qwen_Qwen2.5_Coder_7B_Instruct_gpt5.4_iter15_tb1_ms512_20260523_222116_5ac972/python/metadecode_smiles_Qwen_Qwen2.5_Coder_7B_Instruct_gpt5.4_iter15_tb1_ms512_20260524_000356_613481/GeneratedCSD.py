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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate a valid SMILES string for the requested molecular class. Prefer one complete <<SMILES>> span containing exactly one full valid SMILES. Do not open the span immediately; once inside, keep every prefix valid and continue until the molecule is complete, then close the span immediately.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_forcedOpenDone_: bool
        d_2_forcedOpenDone_ = insideConstrained
        d_3_preludeLimit_: int
        d_3_preludeLimit_ = 12
        d_4_outsideChunkSize_: int
        d_4_outsideChunkSize_ = 4
        d_5_rollbackLimit_: int
        d_5_rollbackLimit_ = 32
        d_6_narrowThreshold_: int
        d_6_narrowThreshold_ = 6
        d_7_repeatThreshold_: int
        d_7_repeatThreshold_ = 2
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if (not(d_2_forcedOpenDone_)) and (((len(generated)) - (len(generatedPrefix))) >= (d_3_preludeLimit_)):
                            d_8_openedGenerated_: _dafny.Seq
                            d_9_openedInside_: bool
                            d_10_openedCurrent_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_8_openedGenerated_ = out0_
                            d_9_openedInside_ = out1_
                            d_10_openedCurrent_ = out2_
                            generated = d_8_openedGenerated_
                            insideConstrainedOut = d_9_openedInside_
                            currentConstrainedOut = d_10_openedCurrent_
                            d_2_forcedOpenDone_ = True
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_11_remainingOutside_: int
                            d_11_remainingOutside_ = (maxSteps) - (d_1_steps_)
                            d_12_chunkBudget_: int
                            if (d_4_outsideChunkSize_) > (d_11_remainingOutside_):
                                d_12_chunkBudget_ = d_11_remainingOutside_
                            elif True:
                                d_12_chunkBudget_ = d_4_outsideChunkSize_
                            d_13_chunkedGenerated_: _dafny.Seq
                            d_14_stoppedOnOpenSpan_: bool
                            d_15_stoppedOnEos_: bool
                            d_16_stepsUsed_: int
                            out3_: _dafny.Seq
                            out4_: bool
                            out5_: bool
                            out6_: int
                            out3_, out4_, out5_, out6_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_12_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_13_chunkedGenerated_ = out3_
                            d_14_stoppedOnOpenSpan_ = out4_
                            d_15_stoppedOnEos_ = out5_
                            d_16_stepsUsed_ = out6_
                            generated = d_13_chunkedGenerated_
                            d_1_steps_ = (d_1_steps_) + (d_16_stepsUsed_)
                            if d_15_stoppedOnEos_:
                                raise _dafny.Break("0")
                            elif d_14_stoppedOnOpenSpan_:
                                d_17_enteredGenerated_: _dafny.Seq
                                d_18_enteredInside_: bool
                                d_19_enteredCurrent_: _dafny.Seq
                                out7_: _dafny.Seq
                                out8_: bool
                                out9_: _dafny.Seq
                                out7_, out8_, out9_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_17_enteredGenerated_ = out7_
                                d_18_enteredInside_ = out8_
                                d_19_enteredCurrent_ = out9_
                                generated = d_17_enteredGenerated_
                                insideConstrainedOut = d_18_enteredInside_
                                currentConstrainedOut = d_19_enteredCurrent_
                                d_2_forcedOpenDone_ = True
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_20_closedGenerated_: _dafny.Seq
                        d_21_closedInside_: bool
                        d_22_closedCurrent_: _dafny.Seq
                        out10_: _dafny.Seq
                        out11_: bool
                        out12_: _dafny.Seq
                        out10_, out11_, out12_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_20_closedGenerated_ = out10_
                        d_21_closedInside_ = out11_
                        d_22_closedCurrent_ = out12_
                        generated = d_20_closedGenerated_
                        insideConstrainedOut = d_21_closedInside_
                        currentConstrainedOut = d_22_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif (len(currentConstrainedOut)) >= (d_5_rollbackLimit_):
                        d_23_rolledGenerated_: _dafny.Seq
                        d_24_rolledCurrent_: _dafny.Seq
                        out13_: _dafny.Seq
                        out14_: _dafny.Seq
                        out13_, out14_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                        d_23_rolledGenerated_ = out13_
                        d_24_rolledCurrent_ = out14_
                        generated = d_23_rolledGenerated_
                        insideConstrainedOut = True
                        currentConstrainedOut = d_24_rolledCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_25_stablePrefix_: _dafny.Seq
                        d_25_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_26_constrainedPrompt_: _dafny.Seq
                        d_26_constrainedPrompt_ = (prompt) + (d_25_stablePrefix_)
                        d_27_validCount_: int
                        out15_: int
                        out15_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                        d_27_validCount_ = out15_
                        d_28_repeatedRecently_: bool
                        d_28_repeatedRecently_ = False
                        if (len(currentConstrainedOut)) > (0):
                            d_29_lastTok_: _dafny.Seq
                            d_29_lastTok_ = (currentConstrainedOut)[(len(currentConstrainedOut)) - (1)]
                            d_30_occ_: int = int(0)
                            out16_: int
                            out16_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(currentConstrainedOut, d_29_lastTok_)
                            d_30_occ_ = out16_
                            if (d_30_occ_) >= (d_7_repeatThreshold_):
                                d_28_repeatedRecently_ = True
                        d_31_nextIn_: _dafny.Seq
                        d_31_nextIn_ = eosToken
                        if d_28_repeatedRecently_:
                            out17_: _dafny.Seq
                            out17_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_26_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('2e0'), eosToken)
                            d_31_nextIn_ = out17_
                        elif (d_27_validCount_) <= (d_6_narrowThreshold_):
                            out18_: _dafny.Seq
                            out18_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_26_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_31_nextIn_ = out18_
                        elif True:
                            out19_: _dafny.Seq
                            out19_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_26_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                            d_31_nextIn_ = out19_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_31_nextIn_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_32_appendedGenerated_: _dafny.Seq
                            d_33_appendedInside_: bool
                            d_34_appendedCurrent_: _dafny.Seq
                            out20_: _dafny.Seq
                            out21_: bool
                            out22_: _dafny.Seq
                            out20_, out21_, out22_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_31_nextIn_)
                            d_32_appendedGenerated_ = out20_
                            d_33_appendedInside_ = out21_
                            d_34_appendedCurrent_ = out22_
                            generated = d_32_appendedGenerated_
                            insideConstrainedOut = d_33_appendedInside_
                            currentConstrainedOut = d_34_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

