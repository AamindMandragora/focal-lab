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
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_3_openedGenerated_: _dafny.Seq
                        d_4_openedInside_: bool
                        d_5_openedCurrent_: _dafny.Seq
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: _dafny.Seq
                        out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                        d_3_openedGenerated_ = out0_
                        d_4_openedInside_ = out1_
                        d_5_openedCurrent_ = out2_
                        generated = d_3_openedGenerated_
                        insideConstrainedOut = d_4_openedInside_
                        currentConstrainedOut = d_5_openedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_6_closedGenerated_: _dafny.Seq
                        d_7_closedInside_: bool
                        d_8_closedCurrent_: _dafny.Seq
                        out3_: _dafny.Seq
                        out4_: bool
                        out5_: _dafny.Seq
                        out3_, out4_, out5_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_6_closedGenerated_ = out3_
                        d_7_closedInside_ = out4_
                        d_8_closedCurrent_ = out5_
                        generated = d_6_closedGenerated_
                        insideConstrainedOut = d_7_closedInside_
                        currentConstrainedOut = d_8_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_9_deadEnd_: bool
                        out6_: bool
                        out6_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                        d_9_deadEnd_ = out6_
                        if d_9_deadEnd_:
                            d_10_repairedGenerated_: _dafny.Seq
                            d_11_repairedCurrent_: _dafny.Seq
                            out7_: _dafny.Seq
                            out8_: _dafny.Seq
                            out7_, out8_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                            d_10_repairedGenerated_ = out7_
                            d_11_repairedCurrent_ = out8_
                            generated = d_10_repairedGenerated_
                            insideConstrainedOut = True
                            currentConstrainedOut = d_11_repairedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_12_stablePrefix_: _dafny.Seq
                            d_12_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_13_constrainedPrompt_: _dafny.Seq
                            d_13_constrainedPrompt_ = (prompt) + (d_12_stablePrefix_)
                            d_14_validCount_: int
                            out9_: int
                            out9_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                            d_14_validCount_ = out9_
                            if ((d_14_validCount_) <= (d_2_narrowThreshold_)) or ((stepTokenBudget) <= (1)):
                                d_15_next_: _dafny.Seq
                                out10_: _dafny.Seq
                                out10_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_13_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_15_next_ = out10_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_15_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_16_appendedGenerated_: _dafny.Seq
                                    d_17_appendedInside_: bool
                                    d_18_appendedCurrent_: _dafny.Seq
                                    out11_: _dafny.Seq
                                    out12_: bool
                                    out13_: _dafny.Seq
                                    out11_, out12_, out13_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_15_next_)
                                    d_16_appendedGenerated_ = out11_
                                    d_17_appendedInside_ = out12_
                                    d_18_appendedCurrent_ = out13_
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
                                d_24_symbolSteps_: int
                                out14_: _dafny.Seq
                                out15_: _dafny.Seq
                                out16_: bool
                                out17_: int
                                out14_, out15_, out16_, out17_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_13_constrainedPrompt_, generated, currentConstrainedOut, d_20_symbolBudget_, eosToken)
                                d_21_symbolGenerated_ = out14_
                                d_22_symbolCurrent_ = out15_
                                d_23_hitEos_ = out16_
                                d_24_symbolSteps_ = out17_
                                generated = d_21_symbolGenerated_
                                insideConstrainedOut = True
                                currentConstrainedOut = d_22_symbolCurrent_
                                d_1_steps_ = (d_1_steps_) + (d_24_symbolSteps_)
                                if d_23_hitEos_:
                                    raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

