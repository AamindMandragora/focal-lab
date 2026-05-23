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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "When producing the molecular answer, prefer a chemically plausible valid SMILES for the requested class. Do not start the constrained molecular segment until the prompt context naturally indicates it.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_narrowThreshold_: int
        d_2_narrowThreshold_ = 12
        d_3_rollbackLimit_: int
        d_3_rollbackLimit_ = 64
        d_4_outsideChunkLimit_: int
        d_4_outsideChunkLimit_ = 4
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_5_remainingOutside_: int
                        d_5_remainingOutside_ = (maxSteps) - (d_1_steps_)
                        d_6_chunkBudget_: int
                        if (d_4_outsideChunkLimit_) > (d_5_remainingOutside_):
                            d_6_chunkBudget_ = d_5_remainingOutside_
                        elif True:
                            d_6_chunkBudget_ = d_4_outsideChunkLimit_
                        d_7_chunkedGenerated_: _dafny.Seq
                        d_8_stoppedOnOpenSpan_: bool
                        d_9_stoppedOnEos_: bool
                        d_10_stepsUsed_: int
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: bool
                        out3_: int
                        out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_6_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_7_chunkedGenerated_ = out0_
                        d_8_stoppedOnOpenSpan_ = out1_
                        d_9_stoppedOnEos_ = out2_
                        d_10_stepsUsed_ = out3_
                        generated = d_7_chunkedGenerated_
                        d_1_steps_ = (d_1_steps_) + (d_10_stepsUsed_)
                        if d_9_stoppedOnEos_:
                            raise _dafny.Break("0")
                        elif d_8_stoppedOnOpenSpan_:
                            d_11_enteredGenerated_: _dafny.Seq
                            d_12_enteredInside_: bool
                            d_13_enteredCurrent_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                            d_11_enteredGenerated_ = out4_
                            d_12_enteredInside_ = out5_
                            d_13_enteredCurrent_ = out6_
                            generated = d_11_enteredGenerated_
                            insideConstrainedOut = d_12_enteredInside_
                            currentConstrainedOut = d_13_enteredCurrent_
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_14_closedGenerated_: _dafny.Seq
                        d_15_closedInside_: bool
                        d_16_closedCurrent_: _dafny.Seq
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: _dafny.Seq
                        out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_14_closedGenerated_ = out7_
                        d_15_closedInside_ = out8_
                        d_16_closedCurrent_ = out9_
                        generated = d_14_closedGenerated_
                        insideConstrainedOut = d_15_closedInside_
                        currentConstrainedOut = d_16_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif (len(currentConstrainedOut)) >= (d_3_rollbackLimit_):
                        d_17_rolledGenerated_: _dafny.Seq
                        d_18_rolledCurrent_: _dafny.Seq
                        out10_: _dafny.Seq
                        out11_: _dafny.Seq
                        out10_, out11_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                        d_17_rolledGenerated_ = out10_
                        d_18_rolledCurrent_ = out11_
                        generated = d_17_rolledGenerated_
                        insideConstrainedOut = True
                        currentConstrainedOut = d_18_rolledCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_19_stablePrefix_: _dafny.Seq
                        d_19_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_20_constrainedPrompt_: _dafny.Seq
                        d_20_constrainedPrompt_ = (prompt) + (d_19_stablePrefix_)
                        d_21_validCount_: int
                        out12_: int
                        out12_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                        d_21_validCount_ = out12_
                        d_22_recentRepeat_: bool
                        d_22_recentRepeat_ = False
                        if (len(currentConstrainedOut)) > (0):
                            d_23_lastTok_: _dafny.Seq
                            d_23_lastTok_ = (currentConstrainedOut)[(len(currentConstrainedOut)) - (1)]
                            d_24_since_: int
                            out13_: int
                            out13_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(currentConstrainedOut, d_23_lastTok_)
                            d_24_since_ = out13_
                            d_22_recentRepeat_ = (d_24_since_) <= (1)
                        if ((stepTokenBudget) > (1)) and ((d_21_validCount_) > (d_2_narrowThreshold_)):
                            d_25_remaining_: int
                            d_25_remaining_ = (maxSteps) - (d_1_steps_)
                            d_26_symbolBudget_: int
                            if (stepTokenBudget) > (d_25_remaining_):
                                d_26_symbolBudget_ = d_25_remaining_
                            elif True:
                                d_26_symbolBudget_ = stepTokenBudget
                            d_27_symbolGenerated_: _dafny.Seq
                            d_28_symbolOut_: _dafny.Seq
                            d_29_hitEos_: bool
                            d_30_innerStepsUsed_: int
                            out14_: _dafny.Seq
                            out15_: _dafny.Seq
                            out16_: bool
                            out17_: int
                            out14_, out15_, out16_, out17_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_20_constrainedPrompt_, generated, currentConstrainedOut, d_26_symbolBudget_, eosToken)
                            d_27_symbolGenerated_ = out14_
                            d_28_symbolOut_ = out15_
                            d_29_hitEos_ = out16_
                            d_30_innerStepsUsed_ = out17_
                            generated = d_27_symbolGenerated_
                            currentConstrainedOut = d_28_symbolOut_
                            d_1_steps_ = (d_1_steps_) + (d_30_innerStepsUsed_)
                            if d_29_hitEos_:
                                raise _dafny.Break("0")
                        elif True:
                            d_31_nextIn_: _dafny.Seq
                            d_31_nextIn_ = eosToken
                            if d_22_recentRepeat_:
                                out18_: _dafny.Seq
                                out18_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_20_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('2e0'), eosToken)
                                d_31_nextIn_ = out18_
                            elif (d_21_validCount_) <= (d_2_narrowThreshold_):
                                out19_: _dafny.Seq
                                out19_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_20_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_2_narrowThreshold_, eosToken)
                                d_31_nextIn_ = out19_
                            elif True:
                                out20_: _dafny.Seq
                                out20_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, d_20_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), eosToken)
                                d_31_nextIn_ = out20_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_31_nextIn_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_32_appendedGenerated_: _dafny.Seq
                                d_33_appendedInside_: bool
                                d_34_appendedCurrent_: _dafny.Seq
                                out21_: _dafny.Seq
                                out22_: bool
                                out23_: _dafny.Seq
                                out21_, out22_, out23_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_31_nextIn_)
                                d_32_appendedGenerated_ = out21_
                                d_33_appendedInside_ = out22_
                                d_34_appendedCurrent_ = out23_
                                generated = d_32_appendedGenerated_
                                insideConstrainedOut = d_33_appendedInside_
                                currentConstrainedOut = d_34_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

