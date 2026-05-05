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
        d_2_narrowThreshold_ = 16
        d_3_minCloseLen_: int
        d_3_minCloseLen_ = 8
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
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
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_7_isComplete_: bool
                        d_7_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if (d_7_isComplete_) and (((len(currentConstrainedOut)) >= (d_3_minCloseLen_)) or (((maxSteps) - (d_1_steps_)) <= (1))):
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
                            raise _dafny.Break("0")
                        elif True:
                            d_11_stablePrefix_: _dafny.Seq
                            d_11_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_12_constrainedPrompt_: _dafny.Seq
                            d_12_constrainedPrompt_ = (prompt) + (d_11_stablePrefix_)
                            if d_7_isComplete_:
                                d_13_remainingComplete_: int
                                d_13_remainingComplete_ = (maxSteps) - (d_1_steps_)
                                d_14_symbolBudgetComplete_: int
                                if ((stepTokenBudget) == (0)) or ((stepTokenBudget) > (d_13_remainingComplete_)):
                                    d_14_symbolBudgetComplete_ = d_13_remainingComplete_
                                elif True:
                                    d_14_symbolBudgetComplete_ = stepTokenBudget
                                d_15_symbolGeneratedComplete_: _dafny.Seq
                                d_16_symbolOutComplete_: _dafny.Seq
                                d_17_hitEosComplete_: bool
                                d_18_stepsUsedComplete_: int
                                out6_: _dafny.Seq
                                out7_: _dafny.Seq
                                out8_: bool
                                out9_: int
                                out6_, out7_, out8_, out9_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_12_constrainedPrompt_, generated, currentConstrainedOut, d_14_symbolBudgetComplete_, eosToken)
                                d_15_symbolGeneratedComplete_ = out6_
                                d_16_symbolOutComplete_ = out7_
                                d_17_hitEosComplete_ = out8_
                                d_18_stepsUsedComplete_ = out9_
                                generated = d_15_symbolGeneratedComplete_
                                insideConstrainedOut = True
                                currentConstrainedOut = d_16_symbolOutComplete_
                                d_1_steps_ = (d_1_steps_) + (d_18_stepsUsedComplete_)
                                if d_17_hitEosComplete_:
                                    raise _dafny.Break("0")
                            elif True:
                                d_19_validCount_: int
                                out10_: int
                                out10_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                                d_19_validCount_ = out10_
                                if (d_19_validCount_) <= (d_2_narrowThreshold_):
                                    d_20_next_: _dafny.Seq
                                    out11_: _dafny.Seq
                                    out11_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_12_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_2_narrowThreshold_, eosToken)
                                    d_20_next_ = out11_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    if (d_20_next_) == (eosToken):
                                        raise _dafny.Break("0")
                                    elif True:
                                        d_21_appendedGenerated_: _dafny.Seq
                                        d_22_appendedInside_: bool
                                        d_23_appendedCurrent_: _dafny.Seq
                                        out12_: _dafny.Seq
                                        out13_: bool
                                        out14_: _dafny.Seq
                                        out12_, out13_, out14_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_20_next_)
                                        d_21_appendedGenerated_ = out12_
                                        d_22_appendedInside_ = out13_
                                        d_23_appendedCurrent_ = out14_
                                        generated = d_21_appendedGenerated_
                                        insideConstrainedOut = d_22_appendedInside_
                                        currentConstrainedOut = d_23_appendedCurrent_
                                elif True:
                                    d_24_remaining_: int
                                    d_24_remaining_ = (maxSteps) - (d_1_steps_)
                                    d_25_symbolBudget_: int
                                    if ((stepTokenBudget) == (0)) or ((stepTokenBudget) > (d_24_remaining_)):
                                        d_25_symbolBudget_ = d_24_remaining_
                                    elif True:
                                        d_25_symbolBudget_ = stepTokenBudget
                                    d_26_symbolGenerated_: _dafny.Seq
                                    d_27_symbolOut_: _dafny.Seq
                                    d_28_hitEos_: bool
                                    d_29_stepsUsed_: int
                                    out15_: _dafny.Seq
                                    out16_: _dafny.Seq
                                    out17_: bool
                                    out18_: int
                                    out15_, out16_, out17_, out18_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_12_constrainedPrompt_, generated, currentConstrainedOut, d_25_symbolBudget_, eosToken)
                                    d_26_symbolGenerated_ = out15_
                                    d_27_symbolOut_ = out16_
                                    d_28_hitEos_ = out17_
                                    d_29_stepsUsed_ = out18_
                                    generated = d_26_symbolGenerated_
                                    insideConstrainedOut = True
                                    currentConstrainedOut = d_27_symbolOut_
                                    d_1_steps_ = (d_1_steps_) + (d_29_stepsUsed_)
                                    if d_28_hitEos_:
                                        raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

