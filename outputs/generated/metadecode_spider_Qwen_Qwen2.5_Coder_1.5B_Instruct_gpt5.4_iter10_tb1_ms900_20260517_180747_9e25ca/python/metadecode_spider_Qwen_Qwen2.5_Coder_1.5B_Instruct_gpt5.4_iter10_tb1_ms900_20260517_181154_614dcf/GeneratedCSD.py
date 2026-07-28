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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Output exactly one SQL query in the required constrained span and avoid extra explanation.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_openedByStrategy_: bool
        d_2_openedByStrategy_ = False
        d_3_narrowThreshold_: int
        d_3_narrowThreshold_ = 12
        if not(insideConstrainedOut):
            d_4_openCount0_: int
            out0_: int
            out0_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))
            d_4_openCount0_ = out0_
            if (d_4_openCount0_) > (0):
                d_5_enteredGenerated0_: _dafny.Seq
                d_6_enteredInside0_: bool
                d_7_enteredCurrent0_: _dafny.Seq
                out1_: _dafny.Seq
                out2_: bool
                out3_: _dafny.Seq
                out1_, out2_, out3_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                d_5_enteredGenerated0_ = out1_
                d_6_enteredInside0_ = out2_
                d_7_enteredCurrent0_ = out3_
                generated = d_5_enteredGenerated0_
                insideConstrainedOut = d_6_enteredInside0_
                currentConstrainedOut = d_7_enteredCurrent0_
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if not(d_2_openedByStrategy_):
                            d_8_openedGenerated_: _dafny.Seq
                            d_9_openedInside_: bool
                            d_10_openedCurrent_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_8_openedGenerated_ = out4_
                            d_9_openedInside_ = out5_
                            d_10_openedCurrent_ = out6_
                            generated = d_8_openedGenerated_
                            insideConstrainedOut = d_9_openedInside_
                            currentConstrainedOut = d_10_openedCurrent_
                            d_2_openedByStrategy_ = True
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_11_next_: _dafny.Seq
                            out7_: _dafny.Seq
                            out7_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_11_next_ = out7_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_11_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_11_next_]))
                                if (d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    d_12_enteredGenerated_: _dafny.Seq
                                    d_13_enteredInside_: bool
                                    d_14_enteredCurrent_: _dafny.Seq
                                    out8_: _dafny.Seq
                                    out9_: bool
                                    out10_: _dafny.Seq
                                    out8_, out9_, out10_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                    d_12_enteredGenerated_ = out8_
                                    d_13_enteredInside_ = out9_
                                    d_14_enteredCurrent_ = out10_
                                    generated = d_12_enteredGenerated_
                                    insideConstrainedOut = d_13_enteredInside_
                                    currentConstrainedOut = d_14_enteredCurrent_
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_15_closedGenerated_: _dafny.Seq
                        d_16_closedInside_: bool
                        d_17_closedCurrent_: _dafny.Seq
                        out11_: _dafny.Seq
                        out12_: bool
                        out13_: _dafny.Seq
                        out11_, out12_, out13_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_15_closedGenerated_ = out11_
                        d_16_closedInside_ = out12_
                        d_17_closedCurrent_ = out13_
                        generated = d_15_closedGenerated_
                        insideConstrainedOut = d_16_closedInside_
                        currentConstrainedOut = d_17_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_18_stablePrefix_: _dafny.Seq
                        d_18_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_19_constrainedPrompt_: _dafny.Seq
                        d_19_constrainedPrompt_ = (prompt) + (d_18_stablePrefix_)
                        d_20_validCount_: int
                        out14_: int
                        out14_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                        d_20_validCount_ = out14_
                        if ((d_20_validCount_) <= (d_3_narrowThreshold_)) or ((stepTokenBudget) <= (1)):
                            d_21_next_: _dafny.Seq
                            out15_: _dafny.Seq
                            out15_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_19_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_3_narrowThreshold_, eosToken)
                            d_21_next_ = out15_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_21_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_22_appendedGenerated_: _dafny.Seq
                                d_23_appendedInside_: bool
                                d_24_appendedCurrent_: _dafny.Seq
                                out16_: _dafny.Seq
                                out17_: bool
                                out18_: _dafny.Seq
                                out16_, out17_, out18_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_21_next_)
                                d_22_appendedGenerated_ = out16_
                                d_23_appendedInside_ = out17_
                                d_24_appendedCurrent_ = out18_
                                generated = d_22_appendedGenerated_
                                insideConstrainedOut = d_23_appendedInside_
                                currentConstrainedOut = d_24_appendedCurrent_
                        elif True:
                            d_25_remaining_: int
                            d_25_remaining_ = (maxSteps) - (d_1_steps_)
                            d_26_symbolBudget_: int
                            if (stepTokenBudget) > (d_25_remaining_):
                                d_26_symbolBudget_ = d_25_remaining_
                            elif True:
                                d_26_symbolBudget_ = stepTokenBudget
                            d_27_symbolGenerated_: _dafny.Seq
                            d_28_symbolOut_: _dafny.Seq
                            d_29_hitEos_: bool
                            d_30_stepsUsed_: int
                            out19_: _dafny.Seq
                            out20_: _dafny.Seq
                            out21_: bool
                            out22_: int
                            out19_, out20_, out21_, out22_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_19_constrainedPrompt_, generated, currentConstrainedOut, d_26_symbolBudget_, eosToken)
                            d_27_symbolGenerated_ = out19_
                            d_28_symbolOut_ = out20_
                            d_29_hitEos_ = out21_
                            d_30_stepsUsed_ = out22_
                            generated = d_27_symbolGenerated_
                            insideConstrainedOut = True
                            currentConstrainedOut = d_28_symbolOut_
                            d_1_steps_ = (d_1_steps_) + (d_30_stepsUsed_)
                            if d_29_hitEos_:
                                raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

