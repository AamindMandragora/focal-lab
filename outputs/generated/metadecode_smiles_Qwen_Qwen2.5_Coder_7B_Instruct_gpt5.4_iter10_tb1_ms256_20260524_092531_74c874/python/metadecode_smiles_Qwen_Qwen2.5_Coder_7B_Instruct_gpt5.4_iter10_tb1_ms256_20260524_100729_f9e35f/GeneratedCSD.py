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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate a valid SMILES string for the requested molecular class. If the model emits <<, place the full SMILES inside that span and continue until the molecule is complete before closing with >>.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_rollbackLimit_: int
        d_2_rollbackLimit_ = 64
        d_3_penaltyTokens_: _dafny.Seq
        d_3_penaltyTokens_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))])
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_4_chunkBudget_: int
                        d_4_chunkBudget_ = (maxSteps) - (d_1_steps_)
                        d_5_generatedOut_: _dafny.Seq
                        d_6_stoppedOnOpenSpan_: bool
                        d_7_stoppedOnEos_: bool
                        d_8_stepsUsed_: int
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: bool
                        out3_: int
                        out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_4_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_5_generatedOut_ = out0_
                        d_6_stoppedOnOpenSpan_ = out1_
                        d_7_stoppedOnEos_ = out2_
                        d_8_stepsUsed_ = out3_
                        generated = d_5_generatedOut_
                        d_1_steps_ = (d_1_steps_) + (d_8_stepsUsed_)
                        if d_7_stoppedOnEos_:
                            raise _dafny.Break("0")
                        elif d_6_stoppedOnOpenSpan_:
                            d_9_enteredGenerated_: _dafny.Seq
                            d_10_enteredInside_: bool
                            d_11_enteredCurrent_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                            d_9_enteredGenerated_ = out4_
                            d_10_enteredInside_ = out5_
                            d_11_enteredCurrent_ = out6_
                            generated = d_9_enteredGenerated_
                            insideConstrainedOut = d_10_enteredInside_
                            currentConstrainedOut = d_11_enteredCurrent_
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_12_closedGenerated_: _dafny.Seq
                        d_13_closedInside_: bool
                        d_14_closedCurrent_: _dafny.Seq
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: _dafny.Seq
                        out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_12_closedGenerated_ = out7_
                        d_13_closedInside_ = out8_
                        d_14_closedCurrent_ = out9_
                        generated = d_12_closedGenerated_
                        insideConstrainedOut = d_13_closedInside_
                        currentConstrainedOut = d_14_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif (len(currentConstrainedOut)) >= (d_2_rollbackLimit_):
                        d_15_rolledGenerated_: _dafny.Seq
                        d_16_rolledCurrent_: _dafny.Seq
                        out10_: _dafny.Seq
                        out11_: _dafny.Seq
                        out10_, out11_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                        d_15_rolledGenerated_ = out10_
                        d_16_rolledCurrent_ = out11_
                        generated = d_15_rolledGenerated_
                        insideConstrainedOut = True
                        currentConstrainedOut = d_16_rolledCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_17_stablePrefix_: _dafny.Seq
                        d_17_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_18_constrainedPrompt_: _dafny.Seq
                        d_18_constrainedPrompt_ = (prompt) + (d_17_stablePrefix_)
                        d_19_next_: _dafny.Seq
                        d_19_next_ = eosToken
                        d_20_validCount_: int
                        out12_: int
                        out12_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                        d_20_validCount_ = out12_
                        d_21_recentRepeat_: bool
                        d_21_recentRepeat_ = False
                        if ((len(currentConstrainedOut)) > (0)) and ((len(generated)) > (0)):
                            d_22_lastTok_: _dafny.Seq
                            d_22_lastTok_ = (generated)[(len(generated)) - (1)]
                            d_23_lastTokCount_: int = int(0)
                            out13_: int
                            out13_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(generated, d_22_lastTok_)
                            d_23_lastTokCount_ = out13_
                            d_21_recentRepeat_ = (d_23_lastTokCount_) >= (2)
                        if d_21_recentRepeat_:
                            out14_: _dafny.Seq
                            out14_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_18_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('2e0'), eosToken)
                            d_19_next_ = out14_
                        elif (d_20_validCount_) <= (12):
                            out15_: _dafny.Seq
                            out15_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_18_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_3_penaltyTokens_, _dafny.BigRational('4e0'), 12, eosToken)
                            d_19_next_ = out15_
                        elif True:
                            d_24_nextSoft_: _dafny.Seq
                            d_25_usedFallback_: bool
                            out16_: _dafny.Seq
                            out17_: bool
                            out16_, out17_ = (d_0_helpers_).SafeSoftConstrainedStep(lm, parser, d_18_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('8e0'), eosToken)
                            d_24_nextSoft_ = out16_
                            d_25_usedFallback_ = out17_
                            d_19_next_ = d_24_nextSoft_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_19_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_26_appendedGenerated_: _dafny.Seq
                            d_27_appendedInside_: bool
                            d_28_appendedCurrent_: _dafny.Seq
                            out18_: _dafny.Seq
                            out19_: bool
                            out20_: _dafny.Seq
                            out18_, out19_, out20_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_19_next_)
                            d_26_appendedGenerated_ = out18_
                            d_27_appendedInside_ = out19_
                            d_28_appendedCurrent_ = out20_
                            generated = d_26_appendedGenerated_
                            insideConstrainedOut = d_27_appendedInside_
                            currentConstrainedOut = d_28_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

