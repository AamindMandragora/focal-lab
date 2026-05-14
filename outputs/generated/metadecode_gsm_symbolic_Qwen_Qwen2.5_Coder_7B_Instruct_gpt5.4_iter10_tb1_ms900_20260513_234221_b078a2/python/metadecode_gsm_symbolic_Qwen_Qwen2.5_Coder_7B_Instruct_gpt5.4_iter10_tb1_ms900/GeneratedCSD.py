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
        d_2_cueArmed_: bool
        d_2_cueArmed_ = False
        d_3_rollbackLimit_: int
        d_3_rollbackLimit_ = 32
        d_4_narrowThreshold_: int
        d_4_narrowThreshold_ = 12
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if d_2_cueArmed_:
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
                            d_2_cueArmed_ = False
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
                            d_11_stoppedOnOpen_: bool
                            d_12_stoppedOnEos_: bool
                            d_13_stepsUsed_: int
                            out3_: _dafny.Seq
                            out4_: bool
                            out5_: bool
                            out6_: int
                            out3_, out4_, out5_, out6_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_9_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_10_chunkedGenerated_ = out3_
                            d_11_stoppedOnOpen_ = out4_
                            d_12_stoppedOnEos_ = out5_
                            d_13_stepsUsed_ = out6_
                            generated = d_10_chunkedGenerated_
                            d_1_steps_ = (d_1_steps_) + (d_13_stepsUsed_)
                            if d_12_stoppedOnEos_:
                                raise _dafny.Break("0")
                            elif d_11_stoppedOnOpen_:
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
                                d_2_cueArmed_ = False
                            elif True:
                                d_17_lastEq_: _dafny.Seq
                                d_18_foundEq_: bool
                                out10_: _dafny.Seq
                                out11_: bool
                                out10_, out11_ = (d_0_helpers_).LastTokenBefore(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")))
                                d_17_lastEq_ = out10_
                                d_18_foundEq_ = out11_
                                d_19_lastColon_: _dafny.Seq
                                d_20_foundColon_: bool
                                out12_: _dafny.Seq
                                out13_: bool
                                out12_, out13_ = (d_0_helpers_).LastTokenBefore(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ":")))
                                d_19_lastColon_ = out12_
                                d_20_foundColon_ = out13_
                                d_2_cueArmed_ = (d_18_foundEq_) or (d_20_foundColon_)
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_21_closedGenerated_: _dafny.Seq
                        d_22_closedInside_: bool
                        d_23_closedCurrent_: _dafny.Seq
                        out14_: _dafny.Seq
                        out15_: bool
                        out16_: _dafny.Seq
                        out14_, out15_, out16_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_21_closedGenerated_ = out14_
                        d_22_closedInside_ = out15_
                        d_23_closedCurrent_ = out16_
                        generated = d_21_closedGenerated_
                        insideConstrainedOut = d_22_closedInside_
                        currentConstrainedOut = d_23_closedCurrent_
                        d_2_cueArmed_ = False
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_24_shouldRepair_: bool
                        d_24_shouldRepair_ = False
                        if (len(currentConstrainedOut)) >= (d_3_rollbackLimit_):
                            d_25_deadEnd_: bool
                            out17_: bool
                            out17_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                            d_25_deadEnd_ = out17_
                            d_24_shouldRepair_ = d_25_deadEnd_
                        if d_24_shouldRepair_:
                            d_26_rolledGenerated_: _dafny.Seq
                            d_27_rolledCurrent_: _dafny.Seq
                            out18_: _dafny.Seq
                            out19_: _dafny.Seq
                            out18_, out19_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                            d_26_rolledGenerated_ = out18_
                            d_27_rolledCurrent_ = out19_
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
                            out20_: int
                            out20_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                            d_30_validCount_ = out20_
                            if ((d_30_validCount_) <= (d_4_narrowThreshold_)) or ((stepTokenBudget) <= (1)):
                                d_31_next_: _dafny.Seq
                                out21_: _dafny.Seq
                                out21_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_29_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_4_narrowThreshold_, eosToken)
                                d_31_next_ = out21_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_31_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_32_appendedGenerated_: _dafny.Seq
                                    d_33_appendedInside_: bool
                                    d_34_appendedCurrent_: _dafny.Seq
                                    out22_: _dafny.Seq
                                    out23_: bool
                                    out24_: _dafny.Seq
                                    out22_, out23_, out24_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_31_next_)
                                    d_32_appendedGenerated_ = out22_
                                    d_33_appendedInside_ = out23_
                                    d_34_appendedCurrent_ = out24_
                                    generated = d_32_appendedGenerated_
                                    insideConstrainedOut = d_33_appendedInside_
                                    currentConstrainedOut = d_34_appendedCurrent_
                            elif True:
                                d_35_remainingInside_: int
                                d_35_remainingInside_ = (maxSteps) - (d_1_steps_)
                                d_36_symbolBudget_: int
                                if ((stepTokenBudget) == (0)) or ((stepTokenBudget) > (d_35_remainingInside_)):
                                    d_36_symbolBudget_ = d_35_remainingInside_
                                elif True:
                                    d_36_symbolBudget_ = stepTokenBudget
                                d_37_symbolGenerated_: _dafny.Seq
                                d_38_symbolCurrent_: _dafny.Seq
                                d_39_hitEos_: bool
                                d_40_stepsUsed_: int
                                out25_: _dafny.Seq
                                out26_: _dafny.Seq
                                out27_: bool
                                out28_: int
                                out25_, out26_, out27_, out28_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_29_constrainedPrompt_, generated, currentConstrainedOut, d_36_symbolBudget_, eosToken)
                                d_37_symbolGenerated_ = out25_
                                d_38_symbolCurrent_ = out26_
                                d_39_hitEos_ = out27_
                                d_40_stepsUsed_ = out28_
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

