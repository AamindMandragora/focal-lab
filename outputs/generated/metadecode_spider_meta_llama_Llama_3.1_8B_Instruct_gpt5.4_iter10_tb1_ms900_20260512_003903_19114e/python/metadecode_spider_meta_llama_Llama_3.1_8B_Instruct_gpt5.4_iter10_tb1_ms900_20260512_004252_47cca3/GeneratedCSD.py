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
        d_2_openedAny_: bool
        d_2_openedAny_ = insideConstrainedOut
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if d_2_openedAny_:
                            d_3_nextOutside_: _dafny.Seq
                            out0_: _dafny.Seq
                            out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_3_nextOutside_ = out0_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_3_nextOutside_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_3_nextOutside_]))
                                if (d_3_nextOutside_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    insideConstrainedOut = True
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                        elif True:
                            d_4_openedGenerated_: _dafny.Seq
                            d_5_openedInside_: bool
                            d_6_openedCurrent_: _dafny.Seq
                            out1_: _dafny.Seq
                            out2_: bool
                            out3_: _dafny.Seq
                            out1_, out2_, out3_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_4_openedGenerated_ = out1_
                            d_5_openedInside_ = out2_
                            d_6_openedCurrent_ = out3_
                            generated = d_4_openedGenerated_
                            insideConstrainedOut = d_5_openedInside_
                            currentConstrainedOut = d_6_openedCurrent_
                            d_2_openedAny_ = True
                            d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            d_7_closedGenerated_: _dafny.Seq
                            d_8_closedInside_: bool
                            d_9_closedCurrent_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_7_closedGenerated_ = out4_
                            d_8_closedInside_ = out5_
                            d_9_closedCurrent_ = out6_
                            generated = d_7_closedGenerated_
                            insideConstrainedOut = d_8_closedInside_
                            currentConstrainedOut = d_9_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_10_remaining_: int
                            d_10_remaining_ = (maxSteps) - (d_1_steps_)
                            if (d_10_remaining_) == (1):
                                d_11_rolledGenerated_: _dafny.Seq
                                d_12_rolledCurrent_: _dafny.Seq
                                out7_: _dafny.Seq
                                out8_: _dafny.Seq
                                out7_, out8_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                d_11_rolledGenerated_ = out7_
                                d_12_rolledCurrent_ = out8_
                                generated = d_11_rolledGenerated_
                                insideConstrainedOut = True
                                currentConstrainedOut = d_12_rolledCurrent_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                d_13_stablePrefix_: _dafny.Seq
                                d_13_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                d_14_constrainedPrompt_: _dafny.Seq
                                d_14_constrainedPrompt_ = (prompt) + (d_13_stablePrefix_)
                                d_15_validCount_: int
                                out9_: int
                                out9_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                                d_15_validCount_ = out9_
                                d_16_deadEndSoon_: bool
                                out10_: bool
                                out10_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 2)
                                d_16_deadEndSoon_ = out10_
                                if (((stepTokenBudget) > (1)) and ((d_15_validCount_) > (8))) and (not(d_16_deadEndSoon_)):
                                    d_17_usableBudget_: int
                                    d_17_usableBudget_ = (d_10_remaining_) - (1)
                                    d_18_symbolBudget_: int
                                    if (stepTokenBudget) > (d_17_usableBudget_):
                                        d_18_symbolBudget_ = d_17_usableBudget_
                                    elif True:
                                        d_18_symbolBudget_ = stepTokenBudget
                                    d_19_symbolGenerated_: _dafny.Seq
                                    d_20_symbolOut_: _dafny.Seq
                                    d_21_hitEos_: bool
                                    d_22_stepsUsed_: int
                                    out11_: _dafny.Seq
                                    out12_: _dafny.Seq
                                    out13_: bool
                                    out14_: int
                                    out11_, out12_, out13_, out14_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_14_constrainedPrompt_, generated, currentConstrainedOut, d_18_symbolBudget_, eosToken)
                                    d_19_symbolGenerated_ = out11_
                                    d_20_symbolOut_ = out12_
                                    d_21_hitEos_ = out13_
                                    d_22_stepsUsed_ = out14_
                                    generated = d_19_symbolGenerated_
                                    insideConstrainedOut = True
                                    currentConstrainedOut = d_20_symbolOut_
                                    d_1_steps_ = (d_1_steps_) + (d_22_stepsUsed_)
                                    if d_21_hitEos_:
                                        raise _dafny.Break("0")
                                elif True:
                                    d_23_nextHard_: _dafny.Seq
                                    out15_: _dafny.Seq
                                    out15_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_14_constrainedPrompt_, currentConstrainedOut, eosToken)
                                    d_23_nextHard_ = out15_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    if (d_23_nextHard_) == (eosToken):
                                        raise _dafny.Break("0")
                                    elif True:
                                        d_24_appendedGenerated_: _dafny.Seq
                                        d_25_appendedInside_: bool
                                        d_26_appendedCurrent_: _dafny.Seq
                                        out16_: _dafny.Seq
                                        out17_: bool
                                        out18_: _dafny.Seq
                                        out16_, out17_, out18_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_23_nextHard_)
                                        d_24_appendedGenerated_ = out16_
                                        d_25_appendedInside_ = out17_
                                        d_26_appendedCurrent_ = out18_
                                        generated = d_24_appendedGenerated_
                                        insideConstrainedOut = d_25_appendedInside_
                                        currentConstrainedOut = d_26_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

