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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. Put each arithmetic computation inside << >> delimiters, and close every computation span before continuing.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_openArmed_: bool
        d_2_openArmed_ = False
        d_3_narrowThreshold_: int
        d_3_narrowThreshold_ = 8
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if d_2_openArmed_:
                            d_4_openedGenerated_: _dafny.Seq
                            d_5_openedInside_: bool
                            d_6_openedCurrent_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_4_openedGenerated_ = out0_
                            d_5_openedInside_ = out1_
                            d_6_openedCurrent_ = out2_
                            generated = d_4_openedGenerated_
                            insideConstrainedOut = d_5_openedInside_
                            currentConstrainedOut = d_6_openedCurrent_
                            d_2_openArmed_ = False
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_7_remainingOutside_: int
                            d_7_remainingOutside_ = (maxSteps) - (d_1_steps_)
                            d_8_eqCountBefore_: int
                            out3_: int
                            out3_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")))
                            d_8_eqCountBefore_ = out3_
                            d_9_colonCountBefore_: int
                            out4_: int
                            out4_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ":")))
                            d_9_colonCountBefore_ = out4_
                            d_10_openCountBefore_: int
                            out5_: int
                            out5_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))
                            d_10_openCountBefore_ = out5_
                            d_11_cueSeenBefore_: bool
                            d_11_cueSeenBefore_ = ((d_8_eqCountBefore_) + (d_9_colonCountBefore_)) > (d_10_openCountBefore_)
                            d_12_chunkBudget_: int
                            if d_11_cueSeenBefore_:
                                if (d_7_remainingOutside_) > (2):
                                    d_12_chunkBudget_ = 2
                                elif True:
                                    d_12_chunkBudget_ = d_7_remainingOutside_
                            elif True:
                                if (d_7_remainingOutside_) > (6):
                                    d_12_chunkBudget_ = 6
                                elif True:
                                    d_12_chunkBudget_ = d_7_remainingOutside_
                            d_13_chunkedGenerated_: _dafny.Seq
                            d_14_stoppedOnOpenSpan_: bool
                            d_15_stoppedOnEos_: bool
                            d_16_stepsUsed_: int
                            out6_: _dafny.Seq
                            out7_: bool
                            out8_: bool
                            out9_: int
                            out6_, out7_, out8_, out9_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_12_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_13_chunkedGenerated_ = out6_
                            d_14_stoppedOnOpenSpan_ = out7_
                            d_15_stoppedOnEos_ = out8_
                            d_16_stepsUsed_ = out9_
                            generated = d_13_chunkedGenerated_
                            d_1_steps_ = (d_1_steps_) + (d_16_stepsUsed_)
                            if d_15_stoppedOnEos_:
                                raise _dafny.Break("0")
                            elif d_14_stoppedOnOpenSpan_:
                                d_17_eqCountAfter_: int
                                out10_: int
                                out10_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")))
                                d_17_eqCountAfter_ = out10_
                                d_18_colonCountAfter_: int
                                out11_: int
                                out11_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ":")))
                                d_18_colonCountAfter_ = out11_
                                d_19_openCountAfter_: int
                                out12_: int
                                out12_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))
                                d_19_openCountAfter_ = out12_
                                if ((d_17_eqCountAfter_) + (d_18_colonCountAfter_)) >= (d_19_openCountAfter_):
                                    d_20_observedGenerated_: _dafny.Seq
                                    d_21_observedInside_: bool
                                    d_22_observedCurrent_: _dafny.Seq
                                    out13_: _dafny.Seq
                                    out14_: bool
                                    out15_: _dafny.Seq
                                    out13_, out14_, out15_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                    d_20_observedGenerated_ = out13_
                                    d_21_observedInside_ = out14_
                                    d_22_observedCurrent_ = out15_
                                    generated = d_20_observedGenerated_
                                    insideConstrainedOut = d_21_observedInside_
                                    currentConstrainedOut = d_22_observedCurrent_
                                    d_2_openArmed_ = False
                                elif True:
                                    d_2_openArmed_ = False
                            elif True:
                                d_23_eqCount_: int
                                out16_: int
                                out16_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")))
                                d_23_eqCount_ = out16_
                                d_24_colonCount_: int
                                out17_: int
                                out17_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ":")))
                                d_24_colonCount_ = out17_
                                d_25_openCount_: int
                                out18_: int
                                out18_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))
                                d_25_openCount_ = out18_
                                if ((d_23_eqCount_) + (d_24_colonCount_)) > (d_25_openCount_):
                                    d_2_openArmed_ = True
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_26_closedGenerated_: _dafny.Seq
                        d_27_closedInside_: bool
                        d_28_closedCurrent_: _dafny.Seq
                        out19_: _dafny.Seq
                        out20_: bool
                        out21_: _dafny.Seq
                        out19_, out20_, out21_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_26_closedGenerated_ = out19_
                        d_27_closedInside_ = out20_
                        d_28_closedCurrent_ = out21_
                        generated = d_26_closedGenerated_
                        insideConstrainedOut = d_27_closedInside_
                        currentConstrainedOut = d_28_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_29_constrainedPrompt_: _dafny.Seq
                        d_29_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_30_validCount_: int
                        out22_: int
                        out22_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                        d_30_validCount_ = out22_
                        if (d_30_validCount_) <= (d_3_narrowThreshold_):
                            d_31_next_: _dafny.Seq
                            out23_: _dafny.Seq
                            out23_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_29_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_31_next_ = out23_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_31_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_32_appendedGenerated_: _dafny.Seq
                                d_33_appendedInside_: bool
                                d_34_appendedCurrent_: _dafny.Seq
                                out24_: _dafny.Seq
                                out25_: bool
                                out26_: _dafny.Seq
                                out24_, out25_, out26_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_31_next_)
                                d_32_appendedGenerated_ = out24_
                                d_33_appendedInside_ = out25_
                                d_34_appendedCurrent_ = out26_
                                generated = d_32_appendedGenerated_
                                insideConstrainedOut = d_33_appendedInside_
                                currentConstrainedOut = d_34_appendedCurrent_
                        elif True:
                            d_35_remainingInside_: int
                            d_35_remainingInside_ = (maxSteps) - (d_1_steps_)
                            d_36_symbolBudget_: int
                            if (stepTokenBudget) == (0):
                                d_36_symbolBudget_ = 1
                            elif (stepTokenBudget) > (d_35_remainingInside_):
                                d_36_symbolBudget_ = d_35_remainingInside_
                            elif True:
                                d_36_symbolBudget_ = stepTokenBudget
                            d_37_symbolGenerated_: _dafny.Seq
                            d_38_symbolCurrent_: _dafny.Seq
                            d_39_hitEos_: bool
                            d_40_stepsUsed_: int
                            out27_: _dafny.Seq
                            out28_: _dafny.Seq
                            out29_: bool
                            out30_: int
                            out27_, out28_, out29_, out30_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_29_constrainedPrompt_, generated, currentConstrainedOut, d_36_symbolBudget_, eosToken)
                            d_37_symbolGenerated_ = out27_
                            d_38_symbolCurrent_ = out28_
                            d_39_hitEos_ = out29_
                            d_40_stepsUsed_ = out30_
                            generated = d_37_symbolGenerated_
                            insideConstrainedOut = True
                            currentConstrainedOut = d_38_symbolCurrent_
                            d_1_steps_ = (d_1_steps_) + (d_40_stepsUsed_)
                            if d_39_hitEos_:
                                raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

