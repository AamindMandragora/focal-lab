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
        if True:
            generated = generatedPrefix
            insideConstrainedOut = insideConstrained
            currentConstrainedOut = currentConstrained
            cost = 0
            d_1_steps_: int
            d_1_steps_ = 0
            d_2_changed_: bool
            d_2_changed_ = False
            d_3_finished_: bool
            d_3_finished_ = False
            d_4_narrowThreshold_: int
            d_4_narrowThreshold_ = 2
            if (maxSteps) > (0):
                if not(insideConstrainedOut):
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
                    d_1_steps_ = (d_1_steps_) + (1)
                    d_2_changed_ = True
                elif (parser).IsCompletePrefix(currentConstrainedOut):
                    d_8_closedGenerated_: _dafny.Seq
                    d_9_closedInside_: bool
                    d_10_closedCurrent_: _dafny.Seq
                    out3_: _dafny.Seq
                    out4_: bool
                    out5_: _dafny.Seq
                    out3_, out4_, out5_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                    d_8_closedGenerated_ = out3_
                    d_9_closedInside_ = out4_
                    d_10_closedCurrent_ = out5_
                    generated = d_8_closedGenerated_
                    insideConstrainedOut = d_9_closedInside_
                    currentConstrainedOut = d_10_closedCurrent_
                    d_1_steps_ = (d_1_steps_) + (1)
                    d_3_finished_ = True
                    d_2_changed_ = True
                elif True:
                    d_11_stablePrefix_: _dafny.Seq
                    d_11_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                    d_12_constrainedPrompt_: _dafny.Seq
                    d_12_constrainedPrompt_ = (prompt) + (d_11_stablePrefix_)
                    d_13_validCount_: int
                    out6_: int
                    out6_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                    d_13_validCount_ = out6_
                    d_14_deadEndRisk_: bool
                    out7_: bool
                    out7_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 2)
                    d_14_deadEndRisk_ = out7_
                    if (((d_14_deadEndRisk_) or ((d_13_validCount_) <= (d_4_narrowThreshold_))) or ((stepTokenBudget) <= (1))) or (((maxSteps) - (d_1_steps_)) == (1)):
                        d_15_next_: _dafny.Seq
                        out8_: _dafny.Seq
                        out8_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_12_constrainedPrompt_, currentConstrainedOut, eosToken)
                        d_15_next_ = out8_
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_2_changed_ = True
                        if (d_15_next_) == (eosToken):
                            d_3_finished_ = True
                        elif True:
                            d_16_appendedGenerated_: _dafny.Seq
                            d_17_appendedInside_: bool
                            d_18_appendedCurrent_: _dafny.Seq
                            out9_: _dafny.Seq
                            out10_: bool
                            out11_: _dafny.Seq
                            out9_, out10_, out11_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_15_next_)
                            d_16_appendedGenerated_ = out9_
                            d_17_appendedInside_ = out10_
                            d_18_appendedCurrent_ = out11_
                            generated = d_16_appendedGenerated_
                            insideConstrainedOut = d_17_appendedInside_
                            currentConstrainedOut = d_18_appendedCurrent_
                    elif True:
                        d_19_remaining_: int
                        d_19_remaining_ = (maxSteps) - (d_1_steps_)
                        d_20_symbolBudget_: int
                        if (stepTokenBudget) > (d_19_remaining_):
                            d_20_symbolBudget_ = d_19_remaining_
                        elif True:
                            d_20_symbolBudget_ = stepTokenBudget
                        d_21_symbolGenerated_: _dafny.Seq
                        d_22_symbolCurrent_: _dafny.Seq
                        d_23_hitEos_: bool
                        d_24_stepsUsed_: int
                        out12_: _dafny.Seq
                        out13_: _dafny.Seq
                        out14_: bool
                        out15_: int
                        out12_, out13_, out14_, out15_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_12_constrainedPrompt_, generated, currentConstrainedOut, d_20_symbolBudget_, eosToken)
                        d_21_symbolGenerated_ = out12_
                        d_22_symbolCurrent_ = out13_
                        d_23_hitEos_ = out14_
                        d_24_stepsUsed_ = out15_
                        generated = d_21_symbolGenerated_
                        insideConstrainedOut = True
                        currentConstrainedOut = d_22_symbolCurrent_
                        d_1_steps_ = (d_1_steps_) + (d_24_stepsUsed_)
                        if (d_24_stepsUsed_) > (0):
                            d_2_changed_ = True
                        if d_23_hitEos_:
                            d_3_finished_ = True
            if ((maxSteps) > (0)) and (not(d_2_changed_)):
                d_25_openedGenerated2_: _dafny.Seq
                d_26_openedInside2_: bool
                d_27_openedCurrent2_: _dafny.Seq
                out16_: _dafny.Seq
                out17_: bool
                out18_: _dafny.Seq
                out16_, out17_, out18_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                d_25_openedGenerated2_ = out16_
                d_26_openedInside2_ = out17_
                d_27_openedCurrent2_ = out18_
                generated = d_25_openedGenerated2_
                insideConstrainedOut = d_26_openedInside2_
                currentConstrainedOut = d_27_openedCurrent2_
                cost = 1
            elif True:
                cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

