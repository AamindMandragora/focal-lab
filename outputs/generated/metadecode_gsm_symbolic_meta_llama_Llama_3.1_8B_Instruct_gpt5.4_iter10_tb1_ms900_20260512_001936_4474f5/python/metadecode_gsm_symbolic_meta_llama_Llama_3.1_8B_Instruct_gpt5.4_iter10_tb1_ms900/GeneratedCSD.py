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
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_triggerArmed_: bool
        d_2_triggerArmed_ = False
        d_3_rollbackLimit_: int
        d_3_rollbackLimit_ = 32
        d_4_narrowThreshold_: int
        d_4_narrowThreshold_ = 10
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if d_2_triggerArmed_:
                            d_5_openedGenerated_: _dafny.Seq
                            d_6_openedInside_: bool
                            d_7_openedCurrent_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_5_openedGenerated_ = out0_
                            d_6_openedInside_ = out1_
                            d_7_openedCurrent_ = out2_
                            generated = d_5_openedGenerated_
                            insideConstrainedOut = d_6_openedInside_
                            currentConstrainedOut = d_7_openedCurrent_
                            d_2_triggerArmed_ = False
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_8_remainingChunk_: int
                            d_8_remainingChunk_ = (maxSteps) - (d_1_steps_)
                            d_9_chunkBudget_: int
                            if (d_8_remainingChunk_) > (3):
                                d_9_chunkBudget_ = 3
                            elif True:
                                d_9_chunkBudget_ = d_8_remainingChunk_
                            d_10_chunkedGenerated_: _dafny.Seq
                            d_11_stoppedOnOpenSpan_: bool
                            d_12_stoppedOnEos_: bool
                            d_13_stepsUsed_: int
                            out3_: _dafny.Seq
                            out4_: bool
                            out5_: bool
                            out6_: int
                            out3_, out4_, out5_, out6_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_9_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_10_chunkedGenerated_ = out3_
                            d_11_stoppedOnOpenSpan_ = out4_
                            d_12_stoppedOnEos_ = out5_
                            d_13_stepsUsed_ = out6_
                            generated = d_10_chunkedGenerated_
                            d_1_steps_ = (d_1_steps_) + (d_13_stepsUsed_)
                            if d_12_stoppedOnEos_:
                                raise _dafny.Break("0")
                            elif d_11_stoppedOnOpenSpan_:
                                d_14_enteredGenerated_: _dafny.Seq
                                d_15_enteredInside_: bool
                                d_16_enteredCurrent_: _dafny.Seq
                                out7_: _dafny.Seq
                                out8_: bool
                                out9_: _dafny.Seq
                                out7_, out8_, out9_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_14_enteredGenerated_ = out7_
                                d_15_enteredInside_ = out8_
                                d_16_enteredCurrent_ = out9_
                                generated = d_14_enteredGenerated_
                                insideConstrainedOut = d_15_enteredInside_
                                currentConstrainedOut = d_16_enteredCurrent_
                                d_2_triggerArmed_ = False
                            elif True:
                                d_17_prevEq_: _dafny.Seq
                                d_18_foundEq_: bool
                                out10_: _dafny.Seq
                                out11_: bool
                                out10_, out11_ = (d_0_helpers_).LastTokenBefore(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")))
                                d_17_prevEq_ = out10_
                                d_18_foundEq_ = out11_
                                d_19_prevColon_: _dafny.Seq
                                d_20_foundColon_: bool
                                out12_: _dafny.Seq
                                out13_: bool
                                out12_, out13_ = (d_0_helpers_).LastTokenBefore(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ":")))
                                d_19_prevColon_ = out12_
                                d_20_foundColon_ = out13_
                                d_21_sinceEq_: int
                                out14_: int
                                out14_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")))
                                d_21_sinceEq_ = out14_
                                d_22_sinceColon_: int
                                out15_: int
                                out15_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ":")))
                                d_22_sinceColon_ = out15_
                                if (((d_18_foundEq_) or (d_20_foundColon_)) or ((d_21_sinceEq_) <= (2))) or ((d_22_sinceColon_) <= (2)):
                                    d_2_triggerArmed_ = True
                                elif True:
                                    d_2_triggerArmed_ = False
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_23_closedGenerated_: _dafny.Seq
                        d_24_closedInside_: bool
                        d_25_closedCurrent_: _dafny.Seq
                        out16_: _dafny.Seq
                        out17_: bool
                        out18_: _dafny.Seq
                        out16_, out17_, out18_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_23_closedGenerated_ = out16_
                        d_24_closedInside_ = out17_
                        d_25_closedCurrent_ = out18_
                        generated = d_23_closedGenerated_
                        insideConstrainedOut = d_24_closedInside_
                        currentConstrainedOut = d_25_closedCurrent_
                        d_2_triggerArmed_ = False
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif (len(currentConstrainedOut)) >= (d_3_rollbackLimit_):
                        d_26_rolledGenerated_: _dafny.Seq
                        d_27_rolledCurrent_: _dafny.Seq
                        out19_: _dafny.Seq
                        out20_: _dafny.Seq
                        out19_, out20_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                        d_26_rolledGenerated_ = out19_
                        d_27_rolledCurrent_ = out20_
                        generated = d_26_rolledGenerated_
                        insideConstrainedOut = True
                        currentConstrainedOut = d_27_rolledCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_28_stablePrefix_: _dafny.Seq
                        d_28_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_29_constrainedPrompt_: _dafny.Seq
                        d_29_constrainedPrompt_ = (prompt) + (d_28_stablePrefix_)
                        d_30_validCount_: int
                        out21_: int
                        out21_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                        d_30_validCount_ = out21_
                        if ((stepTokenBudget) > (1)) and ((d_30_validCount_) > (d_4_narrowThreshold_)):
                            d_31_remaining_: int
                            d_31_remaining_ = (maxSteps) - (d_1_steps_)
                            d_32_symbolBudget_: int
                            if (stepTokenBudget) > (d_31_remaining_):
                                d_32_symbolBudget_ = d_31_remaining_
                            elif True:
                                d_32_symbolBudget_ = stepTokenBudget
                            d_33_symbolGenerated_: _dafny.Seq
                            d_34_symbolCurrent_: _dafny.Seq
                            d_35_hitEos_: bool
                            d_36_stepsUsed_: int
                            out22_: _dafny.Seq
                            out23_: _dafny.Seq
                            out24_: bool
                            out25_: int
                            out22_, out23_, out24_, out25_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_29_constrainedPrompt_, generated, currentConstrainedOut, d_32_symbolBudget_, eosToken)
                            d_33_symbolGenerated_ = out22_
                            d_34_symbolCurrent_ = out23_
                            d_35_hitEos_ = out24_
                            d_36_stepsUsed_ = out25_
                            generated = d_33_symbolGenerated_
                            insideConstrainedOut = True
                            currentConstrainedOut = d_34_symbolCurrent_
                            d_1_steps_ = (d_1_steps_) + (d_36_stepsUsed_)
                            if d_35_hitEos_:
                                raise _dafny.Break("0")
                        elif True:
                            d_37_next_: _dafny.Seq
                            d_37_next_ = eosToken
                            d_38_isDeadEnd_: bool
                            out26_: bool
                            out26_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 2)
                            d_38_isDeadEnd_ = out26_
                            d_39_openCount_: int
                            out27_: int
                            out27_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))
                            d_39_openCount_ = out27_
                            d_40_closeCount_: int
                            out28_: int
                            out28_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")))
                            d_40_closeCount_ = out28_
                            if d_38_isDeadEnd_:
                                out29_: _dafny.Seq
                                out29_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_29_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_37_next_ = out29_
                            elif (len(currentConstrainedOut)) < (2):
                                out30_: _dafny.Seq
                                out30_ = (d_0_helpers_).SafePenalizedConstrainedStep(lm, parser, d_29_constrainedPrompt_, currentConstrainedOut, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]), _dafny.BigRational('6e0'), eosToken)
                                d_37_next_ = out30_
                            elif (d_39_openCount_) > (d_40_closeCount_):
                                out31_: _dafny.Seq
                                out31_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_29_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('2e0'), eosToken)
                                d_37_next_ = out31_
                            elif (d_30_validCount_) <= (d_4_narrowThreshold_):
                                out32_: _dafny.Seq
                                out32_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_29_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                                d_37_next_ = out32_
                            elif True:
                                out33_: _dafny.Seq
                                out33_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, d_29_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), eosToken)
                                d_37_next_ = out33_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_37_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_41_appendedGenerated_: _dafny.Seq
                                d_42_appendedInside_: bool
                                d_43_appendedCurrent_: _dafny.Seq
                                out34_: _dafny.Seq
                                out35_: bool
                                out36_: _dafny.Seq
                                out34_, out35_, out36_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_37_next_)
                                d_41_appendedGenerated_ = out34_
                                d_42_appendedInside_ = out35_
                                d_43_appendedCurrent_ = out36_
                                generated = d_41_appendedGenerated_
                                insideConstrainedOut = d_42_appendedInside_
                                currentConstrainedOut = d_43_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

