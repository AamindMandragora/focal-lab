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
        d_2_outsidePrimed_: bool
        d_2_outsidePrimed_ = False
        d_3_narrowThreshold_: int
        d_3_narrowThreshold_ = 6
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if not(d_2_outsidePrimed_):
                            d_4_next_: _dafny.Seq
                            out0_: _dafny.Seq
                            out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_4_next_ = out0_
                            d_2_outsidePrimed_ = True
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_4_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_4_next_]))
                                if (d_4_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    d_5_enteredGenerated_: _dafny.Seq
                                    d_6_enteredInside_: bool
                                    d_7_enteredCurrent_: _dafny.Seq
                                    out1_: _dafny.Seq
                                    out2_: bool
                                    out3_: _dafny.Seq
                                    out1_, out2_, out3_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                    d_5_enteredGenerated_ = out1_
                                    d_6_enteredInside_ = out2_
                                    d_7_enteredCurrent_ = out3_
                                    generated = d_5_enteredGenerated_
                                    insideConstrainedOut = d_6_enteredInside_
                                    currentConstrainedOut = d_7_enteredCurrent_
                        elif True:
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
                            d_1_steps_ = (d_1_steps_) + (1)
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_11_closedGenerated_: _dafny.Seq
                        d_12_closedInside_: bool
                        d_13_closedCurrent_: _dafny.Seq
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: _dafny.Seq
                        out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_11_closedGenerated_ = out7_
                        d_12_closedInside_ = out8_
                        d_13_closedCurrent_ = out9_
                        generated = d_11_closedGenerated_
                        insideConstrainedOut = d_12_closedInside_
                        currentConstrainedOut = d_13_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_14_stablePrefix_: _dafny.Seq
                        d_14_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_15_constrainedPrompt_: _dafny.Seq
                        d_15_constrainedPrompt_ = (prompt) + (d_14_stablePrefix_)
                        d_16_validCount_: int
                        out10_: int
                        out10_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                        d_16_validCount_ = out10_
                        d_17_deadEndNear_: bool
                        out11_: bool
                        out11_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 2)
                        d_17_deadEndNear_ = out11_
                        d_18_nextIn_: _dafny.Seq
                        d_18_nextIn_ = eosToken
                        if ((d_17_deadEndNear_) or ((d_16_validCount_) <= (d_3_narrowThreshold_))) or ((stepTokenBudget) <= (1)):
                            out12_: _dafny.Seq
                            out12_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_15_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_18_nextIn_ = out12_
                        elif True:
                            d_19_wasConstrained_: bool = False
                            out13_: _dafny.Seq
                            out14_: bool
                            out13_, out14_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, d_15_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_18_nextIn_ = out13_
                            d_19_wasConstrained_ = out14_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_18_nextIn_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_20_appendedGenerated_: _dafny.Seq
                            d_21_appendedInside_: bool
                            d_22_appendedCurrent_: _dafny.Seq
                            out15_: _dafny.Seq
                            out16_: bool
                            out17_: _dafny.Seq
                            out15_, out16_, out17_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_18_nextIn_)
                            d_20_appendedGenerated_ = out15_
                            d_21_appendedInside_ = out16_
                            d_22_appendedCurrent_ = out17_
                            generated = d_20_appendedGenerated_
                            insideConstrainedOut = d_21_appendedInside_
                            currentConstrainedOut = d_22_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

