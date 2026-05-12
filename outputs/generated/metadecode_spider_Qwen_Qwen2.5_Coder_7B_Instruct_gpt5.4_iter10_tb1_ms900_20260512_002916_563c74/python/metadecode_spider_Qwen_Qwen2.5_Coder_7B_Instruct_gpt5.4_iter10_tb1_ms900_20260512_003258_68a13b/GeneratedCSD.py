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
        d_2_openedCount_: int
        out0_: int
        out0_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))
        d_2_openedCount_ = out0_
        d_3_closedCount_: int
        out1_: int
        out1_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")))
        d_3_closedCount_ = out1_
        d_4_needExplicitOpen_: bool
        d_4_needExplicitOpen_ = (d_2_openedCount_) == (d_3_closedCount_)
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if d_4_needExplicitOpen_:
                            d_5_openedGenerated_: _dafny.Seq
                            d_6_openedInside_: bool
                            d_7_openedCurrent_: _dafny.Seq
                            out2_: _dafny.Seq
                            out3_: bool
                            out4_: _dafny.Seq
                            out2_, out3_, out4_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_5_openedGenerated_ = out2_
                            d_6_openedInside_ = out3_
                            d_7_openedCurrent_ = out4_
                            generated = d_5_openedGenerated_
                            insideConstrainedOut = d_6_openedInside_
                            currentConstrainedOut = d_7_openedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_2_openedCount_ = (d_2_openedCount_) + (1)
                            d_4_needExplicitOpen_ = False
                        elif True:
                            d_8_remainingOutside_: int
                            d_8_remainingOutside_ = (maxSteps) - (d_1_steps_)
                            d_9_chunkBudget_: int
                            if (d_8_remainingOutside_) > (3):
                                d_9_chunkBudget_ = 3
                            elif True:
                                d_9_chunkBudget_ = d_8_remainingOutside_
                            d_10_chunkedGenerated_: _dafny.Seq
                            d_11_stoppedOnOpenSpan_: bool
                            d_12_stoppedOnEos_: bool
                            d_13_stepsUsed_: int
                            out5_: _dafny.Seq
                            out6_: bool
                            out7_: bool
                            out8_: int
                            out5_, out6_, out7_, out8_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_9_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_10_chunkedGenerated_ = out5_
                            d_11_stoppedOnOpenSpan_ = out6_
                            d_12_stoppedOnEos_ = out7_
                            d_13_stepsUsed_ = out8_
                            generated = d_10_chunkedGenerated_
                            d_1_steps_ = (d_1_steps_) + (d_13_stepsUsed_)
                            if d_12_stoppedOnEos_:
                                raise _dafny.Break("0")
                            elif d_11_stoppedOnOpenSpan_:
                                d_14_enteredGenerated_: _dafny.Seq
                                d_15_enteredInside_: bool
                                d_16_enteredCurrent_: _dafny.Seq
                                out9_: _dafny.Seq
                                out10_: bool
                                out11_: _dafny.Seq
                                out9_, out10_, out11_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_14_enteredGenerated_ = out9_
                                d_15_enteredInside_ = out10_
                                d_16_enteredCurrent_ = out11_
                                generated = d_14_enteredGenerated_
                                insideConstrainedOut = d_15_enteredInside_
                                currentConstrainedOut = d_16_enteredCurrent_
                                d_2_openedCount_ = (d_2_openedCount_) + (1)
                                d_4_needExplicitOpen_ = False
                            elif True:
                                out12_: int
                                out12_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))
                                d_2_openedCount_ = out12_
                                out13_: int
                                out13_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")))
                                d_3_closedCount_ = out13_
                                d_4_needExplicitOpen_ = (d_2_openedCount_) == (d_3_closedCount_)
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_17_closedGenerated2_: _dafny.Seq
                        d_18_closedInside2_: bool
                        d_19_closedCurrent2_: _dafny.Seq
                        out14_: _dafny.Seq
                        out15_: bool
                        out16_: _dafny.Seq
                        out14_, out15_, out16_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_17_closedGenerated2_ = out14_
                        d_18_closedInside2_ = out15_
                        d_19_closedCurrent2_ = out16_
                        generated = d_17_closedGenerated2_
                        insideConstrainedOut = d_18_closedInside2_
                        currentConstrainedOut = d_19_closedCurrent2_
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_3_closedCount_ = (d_3_closedCount_) + (1)
                        d_4_needExplicitOpen_ = False
                    elif True:
                        d_20_stablePrefix_: _dafny.Seq
                        d_20_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_21_constrainedPrompt_: _dafny.Seq
                        d_21_constrainedPrompt_ = (prompt) + (d_20_stablePrefix_)
                        d_22_validCount_: int
                        out17_: int
                        out17_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                        d_22_validCount_ = out17_
                        d_23_deadEndLike_: bool
                        out18_: bool
                        out18_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                        d_23_deadEndLike_ = out18_
                        if ((d_23_deadEndLike_) or ((d_22_validCount_) <= (12))) or ((stepTokenBudget) == (1)):
                            d_24_next_: _dafny.Seq
                            out19_: _dafny.Seq
                            out19_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_21_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                            d_24_next_ = out19_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_24_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_25_appendedGenerated_: _dafny.Seq
                                d_26_appendedInside_: bool
                                d_27_appendedCurrent_: _dafny.Seq
                                out20_: _dafny.Seq
                                out21_: bool
                                out22_: _dafny.Seq
                                out20_, out21_, out22_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_24_next_)
                                d_25_appendedGenerated_ = out20_
                                d_26_appendedInside_ = out21_
                                d_27_appendedCurrent_ = out22_
                                generated = d_25_appendedGenerated_
                                insideConstrainedOut = d_26_appendedInside_
                                currentConstrainedOut = d_27_appendedCurrent_
                        elif True:
                            d_28_remainingInside_: int
                            d_28_remainingInside_ = (maxSteps) - (d_1_steps_)
                            d_29_symbolBudget_: int
                            if (stepTokenBudget) == (0):
                                d_29_symbolBudget_ = d_28_remainingInside_
                            elif (stepTokenBudget) > (d_28_remainingInside_):
                                d_29_symbolBudget_ = d_28_remainingInside_
                            elif True:
                                d_29_symbolBudget_ = stepTokenBudget
                            d_30_symbolGenerated_: _dafny.Seq
                            d_31_symbolCurrent_: _dafny.Seq
                            d_32_hitEos_: bool
                            d_33_stepsUsed2_: int
                            out23_: _dafny.Seq
                            out24_: _dafny.Seq
                            out25_: bool
                            out26_: int
                            out23_, out24_, out25_, out26_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_21_constrainedPrompt_, generated, currentConstrainedOut, d_29_symbolBudget_, eosToken)
                            d_30_symbolGenerated_ = out23_
                            d_31_symbolCurrent_ = out24_
                            d_32_hitEos_ = out25_
                            d_33_stepsUsed2_ = out26_
                            generated = d_30_symbolGenerated_
                            currentConstrainedOut = d_31_symbolCurrent_
                            insideConstrainedOut = True
                            d_1_steps_ = (d_1_steps_) + (d_33_stepsUsed2_)
                            if d_32_hitEos_:
                                raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

