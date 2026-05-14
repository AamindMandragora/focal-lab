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
        d_2_rollbackLimit_: int
        d_2_rollbackLimit_ = 48
        d_3_narrowThreshold_: int
        d_3_narrowThreshold_ = 10
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if (stepTokenBudget) > (1):
                            d_4_remainingChunk_: int
                            d_4_remainingChunk_ = (maxSteps) - (d_1_steps_)
                            d_5_chunkBudget_: int
                            if (stepTokenBudget) <= (d_4_remainingChunk_):
                                d_5_chunkBudget_ = stepTokenBudget
                            elif True:
                                d_5_chunkBudget_ = d_4_remainingChunk_
                            d_6_chunkedGenerated_: _dafny.Seq
                            d_7_stoppedOnOpenSpan_: bool
                            d_8_stoppedOnEos_: bool
                            d_9_stepsUsed_: int
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: bool
                            out3_: int
                            out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_5_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_6_chunkedGenerated_ = out0_
                            d_7_stoppedOnOpenSpan_ = out1_
                            d_8_stoppedOnEos_ = out2_
                            d_9_stepsUsed_ = out3_
                            generated = d_6_chunkedGenerated_
                            d_1_steps_ = (d_1_steps_) + (d_9_stepsUsed_)
                            if d_8_stoppedOnEos_:
                                raise _dafny.Break("0")
                            elif d_7_stoppedOnOpenSpan_:
                                d_10_enteredGenerated_: _dafny.Seq
                                d_11_enteredInside_: bool
                                d_12_enteredCurrent_: _dafny.Seq
                                out4_: _dafny.Seq
                                out5_: bool
                                out6_: _dafny.Seq
                                out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_10_enteredGenerated_ = out4_
                                d_11_enteredInside_ = out5_
                                d_12_enteredCurrent_ = out6_
                                generated = d_10_enteredGenerated_
                                insideConstrainedOut = d_11_enteredInside_
                                currentConstrainedOut = d_12_enteredCurrent_
                        elif True:
                            d_13_shouldOpen_: bool
                            d_13_shouldOpen_ = False
                            if (len(generated)) == (0):
                                d_13_shouldOpen_ = True
                            elif True:
                                d_14_lastBeforeClose_: _dafny.Seq
                                d_15_foundBeforeClose_: bool
                                out7_: _dafny.Seq
                                out8_: bool
                                out7_, out8_ = (d_0_helpers_).LastTokenBefore(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")))
                                d_14_lastBeforeClose_ = out7_
                                d_15_foundBeforeClose_ = out8_
                                if (d_15_foundBeforeClose_) and ((d_14_lastBeforeClose_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ":")))):
                                    d_13_shouldOpen_ = True
                            if d_13_shouldOpen_:
                                d_16_openedGenerated_: _dafny.Seq
                                d_17_openedInside_: bool
                                d_18_openedCurrent_: _dafny.Seq
                                out9_: _dafny.Seq
                                out10_: bool
                                out11_: _dafny.Seq
                                out9_, out10_, out11_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                d_16_openedGenerated_ = out9_
                                d_17_openedInside_ = out10_
                                d_18_openedCurrent_ = out11_
                                generated = d_16_openedGenerated_
                                insideConstrainedOut = d_17_openedInside_
                                currentConstrainedOut = d_18_openedCurrent_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                d_19_next_: _dafny.Seq
                                out12_: _dafny.Seq
                                out12_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                                d_19_next_ = out12_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_19_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_19_next_]))
                                    if (d_19_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                        d_20_enteredGenerated2_: _dafny.Seq
                                        d_21_enteredInside2_: bool
                                        d_22_enteredCurrent2_: _dafny.Seq
                                        out13_: _dafny.Seq
                                        out14_: bool
                                        out15_: _dafny.Seq
                                        out13_, out14_, out15_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                        d_20_enteredGenerated2_ = out13_
                                        d_21_enteredInside2_ = out14_
                                        d_22_enteredCurrent2_ = out15_
                                        generated = d_20_enteredGenerated2_
                                        insideConstrainedOut = d_21_enteredInside2_
                                        currentConstrainedOut = d_22_enteredCurrent2_
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
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif (len(currentConstrainedOut)) >= (d_2_rollbackLimit_):
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
                        d_31_isNarrow_: bool
                        out22_: bool
                        out22_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, d_3_narrowThreshold_)
                        d_31_isNarrow_ = out22_
                        if (d_31_isNarrow_) or ((d_30_validCount_) <= (d_3_narrowThreshold_)):
                            d_32_nextConstrained_: _dafny.Seq
                            out23_: _dafny.Seq
                            out23_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_29_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                            d_32_nextConstrained_ = out23_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_32_nextConstrained_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_33_appendedGenerated_: _dafny.Seq
                                d_34_appendedInside_: bool
                                d_35_appendedCurrent_: _dafny.Seq
                                out24_: _dafny.Seq
                                out25_: bool
                                out26_: _dafny.Seq
                                out24_, out25_, out26_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_32_nextConstrained_)
                                d_33_appendedGenerated_ = out24_
                                d_34_appendedInside_ = out25_
                                d_35_appendedCurrent_ = out26_
                                generated = d_33_appendedGenerated_
                                insideConstrainedOut = d_34_appendedInside_
                                currentConstrainedOut = d_35_appendedCurrent_
                        elif True:
                            d_36_remaining_: int
                            d_36_remaining_ = (maxSteps) - (d_1_steps_)
                            d_37_symbolBudget_: int
                            if ((stepTokenBudget) == (0)) or ((stepTokenBudget) > (d_36_remaining_)):
                                d_37_symbolBudget_ = d_36_remaining_
                            elif True:
                                d_37_symbolBudget_ = stepTokenBudget
                            d_38_symbolGenerated_: _dafny.Seq
                            d_39_symbolCurrent_: _dafny.Seq
                            d_40_hitEos_: bool
                            d_41_stepsUsed2_: int
                            out27_: _dafny.Seq
                            out28_: _dafny.Seq
                            out29_: bool
                            out30_: int
                            out27_, out28_, out29_, out30_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_29_constrainedPrompt_, generated, currentConstrainedOut, d_37_symbolBudget_, eosToken)
                            d_38_symbolGenerated_ = out27_
                            d_39_symbolCurrent_ = out28_
                            d_40_hitEos_ = out29_
                            d_41_stepsUsed2_ = out30_
                            generated = d_38_symbolGenerated_
                            currentConstrainedOut = d_39_symbolCurrent_
                            d_1_steps_ = (d_1_steps_) + (d_41_stepsUsed2_)
                            if d_40_hitEos_:
                                raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

