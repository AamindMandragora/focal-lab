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
        d_2_didInitialProbe_: bool
        d_2_didInitialProbe_ = False
        d_3_narrowThreshold_: int
        d_3_narrowThreshold_ = 10
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if not(d_2_didInitialProbe_):
                            d_4_remainingProbe_: int
                            d_4_remainingProbe_ = (maxSteps) - (d_1_steps_)
                            d_5_probeBudget_: int
                            if (d_4_remainingProbe_) > (1):
                                d_5_probeBudget_ = 1
                            elif True:
                                d_5_probeBudget_ = d_4_remainingProbe_
                            if (d_5_probeBudget_) == (0):
                                raise _dafny.Break("0")
                            d_6_chunkedGenerated_: _dafny.Seq
                            d_7_stoppedOpen_: bool
                            d_8_stoppedEos_: bool
                            d_9_stepsUsed_: int
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: bool
                            out3_: int
                            out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_5_probeBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_6_chunkedGenerated_ = out0_
                            d_7_stoppedOpen_ = out1_
                            d_8_stoppedEos_ = out2_
                            d_9_stepsUsed_ = out3_
                            generated = d_6_chunkedGenerated_
                            d_1_steps_ = (d_1_steps_) + (d_9_stepsUsed_)
                            d_2_didInitialProbe_ = True
                            if d_8_stoppedEos_:
                                raise _dafny.Break("0")
                            elif d_7_stoppedOpen_:
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
                            d_13_openedGenerated_: _dafny.Seq
                            d_14_openedInside_: bool
                            d_15_openedCurrent_: _dafny.Seq
                            out7_: _dafny.Seq
                            out8_: bool
                            out9_: _dafny.Seq
                            out7_, out8_, out9_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_13_openedGenerated_ = out7_
                            d_14_openedInside_ = out8_
                            d_15_openedCurrent_ = out9_
                            generated = d_13_openedGenerated_
                            insideConstrainedOut = d_14_openedInside_
                            currentConstrainedOut = d_15_openedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_16_closedGenerated_: _dafny.Seq
                        d_17_closedInside_: bool
                        d_18_closedCurrent_: _dafny.Seq
                        out10_: _dafny.Seq
                        out11_: bool
                        out12_: _dafny.Seq
                        out10_, out11_, out12_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_16_closedGenerated_ = out10_
                        d_17_closedInside_ = out11_
                        d_18_closedCurrent_ = out12_
                        generated = d_16_closedGenerated_
                        insideConstrainedOut = d_17_closedInside_
                        currentConstrainedOut = d_18_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_19_stablePrefix_: _dafny.Seq
                        d_19_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_20_constrainedPrompt_: _dafny.Seq
                        d_20_constrainedPrompt_ = (prompt) + (d_19_stablePrefix_)
                        d_21_validCount_: int
                        out13_: int
                        out13_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                        d_21_validCount_ = out13_
                        if (d_21_validCount_) <= (d_3_narrowThreshold_):
                            d_22_next_: _dafny.Seq
                            d_22_next_ = eosToken
                            if (len(currentConstrainedOut)) == (0):
                                out14_: _dafny.Seq
                                out14_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, d_20_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), eosToken)
                                d_22_next_ = out14_
                            elif (len(currentConstrainedOut)) >= (6):
                                out15_: _dafny.Seq
                                out15_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_20_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('2e0'), eosToken)
                                d_22_next_ = out15_
                            elif True:
                                out16_: _dafny.Seq
                                out16_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_20_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_3_narrowThreshold_, eosToken)
                                d_22_next_ = out16_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_22_next_) == (eosToken):
                                d_23_repairedGenerated_: _dafny.Seq
                                d_24_repairedCurrent_: _dafny.Seq
                                out17_: _dafny.Seq
                                out18_: _dafny.Seq
                                out17_, out18_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                d_23_repairedGenerated_ = out17_
                                d_24_repairedCurrent_ = out18_
                                generated = d_23_repairedGenerated_
                                insideConstrainedOut = True
                                currentConstrainedOut = d_24_repairedCurrent_
                                if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                    d_25_closedGenerated2_: _dafny.Seq
                                    d_26_closedInside2_: bool
                                    d_27_closedCurrent2_: _dafny.Seq
                                    out19_: _dafny.Seq
                                    out20_: bool
                                    out21_: _dafny.Seq
                                    out19_, out20_, out21_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_25_closedGenerated2_ = out19_
                                    d_26_closedInside2_ = out20_
                                    d_27_closedCurrent2_ = out21_
                                    generated = d_25_closedGenerated2_
                                    insideConstrainedOut = d_26_closedInside2_
                                    currentConstrainedOut = d_27_closedCurrent2_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                elif True:
                                    raise _dafny.Break("0")
                            elif True:
                                d_28_appendedGenerated_: _dafny.Seq
                                d_29_appendedInside_: bool
                                d_30_appendedCurrent_: _dafny.Seq
                                out22_: _dafny.Seq
                                out23_: bool
                                out24_: _dafny.Seq
                                out22_, out23_, out24_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_22_next_)
                                d_28_appendedGenerated_ = out22_
                                d_29_appendedInside_ = out23_
                                d_30_appendedCurrent_ = out24_
                                generated = d_28_appendedGenerated_
                                insideConstrainedOut = d_29_appendedInside_
                                currentConstrainedOut = d_30_appendedCurrent_
                        elif True:
                            d_31_remaining_: int
                            d_31_remaining_ = (maxSteps) - (d_1_steps_)
                            d_32_symbolBudget_: int
                            if ((stepTokenBudget) == (0)) or ((stepTokenBudget) > (d_31_remaining_)):
                                d_32_symbolBudget_ = d_31_remaining_
                            elif True:
                                d_32_symbolBudget_ = stepTokenBudget
                            d_33_symbolGenerated_: _dafny.Seq
                            d_34_symbolCurrent_: _dafny.Seq
                            d_35_hitEos_: bool
                            d_36_stepsUsed2_: int
                            out25_: _dafny.Seq
                            out26_: _dafny.Seq
                            out27_: bool
                            out28_: int
                            out25_, out26_, out27_, out28_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_20_constrainedPrompt_, generated, currentConstrainedOut, d_32_symbolBudget_, eosToken)
                            d_33_symbolGenerated_ = out25_
                            d_34_symbolCurrent_ = out26_
                            d_35_hitEos_ = out27_
                            d_36_stepsUsed2_ = out28_
                            generated = d_33_symbolGenerated_
                            insideConstrainedOut = True
                            currentConstrainedOut = d_34_symbolCurrent_
                            d_1_steps_ = (d_1_steps_) + (d_36_stepsUsed2_)
                            if d_35_hitEos_:
                                d_37_repairedGenerated2_: _dafny.Seq
                                d_38_repairedCurrent2_: _dafny.Seq
                                out29_: _dafny.Seq
                                out30_: _dafny.Seq
                                out29_, out30_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                d_37_repairedGenerated2_ = out29_
                                d_38_repairedCurrent2_ = out30_
                                generated = d_37_repairedGenerated2_
                                insideConstrainedOut = True
                                currentConstrainedOut = d_38_repairedCurrent2_
                                if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                    d_39_closedGenerated3_: _dafny.Seq
                                    d_40_closedInside3_: bool
                                    d_41_closedCurrent3_: _dafny.Seq
                                    out31_: _dafny.Seq
                                    out32_: bool
                                    out33_: _dafny.Seq
                                    out31_, out32_, out33_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_39_closedGenerated3_ = out31_
                                    d_40_closedInside3_ = out32_
                                    d_41_closedCurrent3_ = out33_
                                    generated = d_39_closedGenerated3_
                                    insideConstrainedOut = d_40_closedInside3_
                                    currentConstrainedOut = d_41_closedCurrent3_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                elif True:
                                    raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

