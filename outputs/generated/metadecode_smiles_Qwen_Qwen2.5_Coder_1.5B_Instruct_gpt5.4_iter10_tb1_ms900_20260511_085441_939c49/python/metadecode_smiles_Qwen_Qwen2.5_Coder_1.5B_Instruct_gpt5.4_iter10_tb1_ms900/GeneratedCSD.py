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
        d_2_narrowThreshold_: int
        d_2_narrowThreshold_ = 12
        d_3_rollbackLimit_: int
        d_3_rollbackLimit_ = 64
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_4_chunkBudget_: int
                        d_4_chunkBudget_ = (maxSteps) - (d_1_steps_)
                        d_5_chunkGenerated_: _dafny.Seq
                        d_6_stoppedOnOpenSpan_: bool
                        d_7_stoppedOnEos_: bool
                        d_8_stepsUsed_: int
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: bool
                        out3_: int
                        out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_4_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_5_chunkGenerated_ = out0_
                        d_6_stoppedOnOpenSpan_ = out1_
                        d_7_stoppedOnEos_ = out2_
                        d_8_stepsUsed_ = out3_
                        generated = d_5_chunkGenerated_
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
                    elif (len(currentConstrainedOut)) >= (d_3_rollbackLimit_):
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
                        d_19_validCount_: int
                        out12_: int
                        out12_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                        d_19_validCount_ = out12_
                        d_20_openParens_: int
                        out13_: int
                        out13_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "(")))
                        d_20_openParens_ = out13_
                        d_21_closeParens_: int
                        out14_: int
                        out14_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ")")))
                        d_21_closeParens_ = out14_
                        d_22_useRepPenalty_: bool
                        d_22_useRepPenalty_ = False
                        if (len(currentConstrainedOut)) > (8):
                            if (d_20_openParens_) > (d_21_closeParens_):
                                d_22_useRepPenalty_ = True
                        if d_22_useRepPenalty_:
                            d_23_nextRep_: _dafny.Seq
                            out15_: _dafny.Seq
                            out15_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_18_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('2e0'), eosToken)
                            d_23_nextRep_ = out15_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_23_nextRep_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_24_appendedGenerated_: _dafny.Seq
                                d_25_appendedInside_: bool
                                d_26_appendedCurrent_: _dafny.Seq
                                out16_: _dafny.Seq
                                out17_: bool
                                out18_: _dafny.Seq
                                out16_, out17_, out18_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_23_nextRep_)
                                d_24_appendedGenerated_ = out16_
                                d_25_appendedInside_ = out17_
                                d_26_appendedCurrent_ = out18_
                                generated = d_24_appendedGenerated_
                                insideConstrainedOut = d_25_appendedInside_
                                currentConstrainedOut = d_26_appendedCurrent_
                        elif (d_19_validCount_) <= (d_2_narrowThreshold_):
                            d_27_nextAdaptive_: _dafny.Seq
                            out19_: _dafny.Seq
                            out19_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_18_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_2_narrowThreshold_, eosToken)
                            d_27_nextAdaptive_ = out19_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_27_nextAdaptive_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_28_appendedGenerated2_: _dafny.Seq
                                d_29_appendedInside2_: bool
                                d_30_appendedCurrent2_: _dafny.Seq
                                out20_: _dafny.Seq
                                out21_: bool
                                out22_: _dafny.Seq
                                out20_, out21_, out22_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_27_nextAdaptive_)
                                d_28_appendedGenerated2_ = out20_
                                d_29_appendedInside2_ = out21_
                                d_30_appendedCurrent2_ = out22_
                                generated = d_28_appendedGenerated2_
                                insideConstrainedOut = d_29_appendedInside2_
                                currentConstrainedOut = d_30_appendedCurrent2_
                        elif True:
                            d_31_remaining_: int
                            d_31_remaining_ = (maxSteps) - (d_1_steps_)
                            d_32_symbolBudget_: int
                            if ((stepTokenBudget) == (0)) or ((stepTokenBudget) > (d_31_remaining_)):
                                d_32_symbolBudget_ = d_31_remaining_
                            elif True:
                                d_32_symbolBudget_ = stepTokenBudget
                            d_33_symbolGenerated_: _dafny.Seq
                            d_34_symbolOut_: _dafny.Seq
                            d_35_hitEos_: bool
                            d_36_symbolSteps_: int
                            out23_: _dafny.Seq
                            out24_: _dafny.Seq
                            out25_: bool
                            out26_: int
                            out23_, out24_, out25_, out26_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_18_constrainedPrompt_, generated, currentConstrainedOut, d_32_symbolBudget_, eosToken)
                            d_33_symbolGenerated_ = out23_
                            d_34_symbolOut_ = out24_
                            d_35_hitEos_ = out25_
                            d_36_symbolSteps_ = out26_
                            generated = d_33_symbolGenerated_
                            insideConstrainedOut = True
                            currentConstrainedOut = d_34_symbolOut_
                            d_1_steps_ = (d_1_steps_) + (d_36_symbolSteps_)
                            if d_35_hitEos_:
                                raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

