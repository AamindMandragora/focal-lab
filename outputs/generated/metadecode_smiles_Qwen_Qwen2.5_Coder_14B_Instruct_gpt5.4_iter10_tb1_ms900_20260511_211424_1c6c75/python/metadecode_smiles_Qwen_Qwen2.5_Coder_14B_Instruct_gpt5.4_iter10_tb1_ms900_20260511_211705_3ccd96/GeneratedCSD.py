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
        d_2_freeTokensSinceSpan_: int
        d_2_freeTokensSinceSpan_ = 0
        d_3_narrowThreshold_: int
        d_3_narrowThreshold_ = 12
        d_4_openAfterFree_: int
        d_4_openAfterFree_ = 8
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if (d_2_freeTokensSinceSpan_) >= (d_4_openAfterFree_):
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
                            d_2_freeTokensSinceSpan_ = 0
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_8_remainingOutside_: int
                            d_8_remainingOutside_ = (maxSteps) - (d_1_steps_)
                            d_9_chunkBudget_: int
                            if (d_8_remainingOutside_) > (2):
                                d_9_chunkBudget_ = 2
                            elif True:
                                d_9_chunkBudget_ = d_8_remainingOutside_
                            d_10_chunkedGenerated_: _dafny.Seq
                            d_11_stoppedOpen_: bool
                            d_12_stoppedEos_: bool
                            d_13_stepsUsed_: int
                            out3_: _dafny.Seq
                            out4_: bool
                            out5_: bool
                            out6_: int
                            out3_, out4_, out5_, out6_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_9_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_10_chunkedGenerated_ = out3_
                            d_11_stoppedOpen_ = out4_
                            d_12_stoppedEos_ = out5_
                            d_13_stepsUsed_ = out6_
                            generated = d_10_chunkedGenerated_
                            d_1_steps_ = (d_1_steps_) + (d_13_stepsUsed_)
                            d_2_freeTokensSinceSpan_ = (d_2_freeTokensSinceSpan_) + (d_13_stepsUsed_)
                            if d_12_stoppedEos_:
                                raise _dafny.Break("0")
                            elif d_11_stoppedOpen_:
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
                                d_2_freeTokensSinceSpan_ = 0
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
                        d_2_freeTokensSinceSpan_ = 0
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_20_stablePrefix_: _dafny.Seq
                        d_20_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_21_constrainedPrompt_: _dafny.Seq
                        d_21_constrainedPrompt_ = (prompt) + (d_20_stablePrefix_)
                        d_22_validCount_: int
                        out13_: int
                        out13_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                        d_22_validCount_ = out13_
                        d_23_canUseSymbol_: bool
                        d_23_canUseSymbol_ = ((stepTokenBudget) > (1)) and ((d_22_validCount_) > (d_3_narrowThreshold_))
                        if d_23_canUseSymbol_:
                            d_24_remainingInside_: int
                            d_24_remainingInside_ = (maxSteps) - (d_1_steps_)
                            d_25_symbolBudget_: int
                            if (stepTokenBudget) > (d_24_remainingInside_):
                                d_25_symbolBudget_ = d_24_remainingInside_
                            elif True:
                                d_25_symbolBudget_ = stepTokenBudget
                            d_26_symbolGenerated_: _dafny.Seq
                            d_27_symbolCurrent_: _dafny.Seq
                            d_28_hitEos_: bool
                            d_29_stepsUsed_: int
                            out14_: _dafny.Seq
                            out15_: _dafny.Seq
                            out16_: bool
                            out17_: int
                            out14_, out15_, out16_, out17_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_21_constrainedPrompt_, generated, currentConstrainedOut, d_25_symbolBudget_, eosToken)
                            d_26_symbolGenerated_ = out14_
                            d_27_symbolCurrent_ = out15_
                            d_28_hitEos_ = out16_
                            d_29_stepsUsed_ = out17_
                            generated = d_26_symbolGenerated_
                            currentConstrainedOut = d_27_symbolCurrent_
                            d_1_steps_ = (d_1_steps_) + (d_29_stepsUsed_)
                            if d_28_hitEos_:
                                raise _dafny.Break("0")
                        elif True:
                            d_30_next_: _dafny.Seq
                            d_30_next_ = eosToken
                            d_31_repeatedGap_: int
                            out18_: int
                            out18_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ")")))
                            d_31_repeatedGap_ = out18_
                            d_32_deadEnd_: bool
                            out19_: bool
                            out19_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                            d_32_deadEnd_ = out19_
                            if d_32_deadEnd_:
                                d_33_nextSoft_: _dafny.Seq
                                d_34_usedFallback_: bool
                                out20_: _dafny.Seq
                                out21_: bool
                                out20_, out21_ = (d_0_helpers_).SafeSoftConstrainedStep(lm, parser, d_21_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('8e0'), eosToken)
                                d_33_nextSoft_ = out20_
                                d_34_usedFallback_ = out21_
                                d_30_next_ = d_33_nextSoft_
                            elif (d_31_repeatedGap_) < (3):
                                out22_: _dafny.Seq
                                out22_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_21_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('2e0'), eosToken)
                                d_30_next_ = out22_
                            elif (d_22_validCount_) <= (4):
                                d_35_gatedNext_: _dafny.Seq
                                d_36_wasConstrained_: bool
                                out23_: _dafny.Seq
                                out24_: bool
                                out23_, out24_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, d_21_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_35_gatedNext_ = out23_
                                d_36_wasConstrained_ = out24_
                                d_30_next_ = d_35_gatedNext_
                            elif (len(validTokenGroups)) > (0):
                                out25_: _dafny.Seq
                                out25_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_21_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_3_narrowThreshold_, eosToken)
                                d_30_next_ = out25_
                            elif True:
                                out26_: _dafny.Seq
                                out26_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_21_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_30_next_ = out26_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_30_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_37_appendedGenerated_: _dafny.Seq
                                d_38_appendedInside_: bool
                                d_39_appendedCurrent_: _dafny.Seq
                                out27_: _dafny.Seq
                                out28_: bool
                                out29_: _dafny.Seq
                                out27_, out28_, out29_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_30_next_)
                                d_37_appendedGenerated_ = out27_
                                d_38_appendedInside_ = out28_
                                d_39_appendedCurrent_ = out29_
                                generated = d_37_appendedGenerated_
                                insideConstrainedOut = d_38_appendedInside_
                                currentConstrainedOut = d_39_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

