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
    def MyCSDStrategy(lm, parser, prompt, generatedPrefix, insideConstrained, currentConstrained, maxSteps, eosToken):
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
        d_2_didWarmup_: bool
        d_2_didWarmup_ = insideConstrained
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if insideConstrainedOut:
                        d_3_isComplete_: bool
                        d_3_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_3_isComplete_:
                            d_4_closedGenerated_: _dafny.Seq
                            d_5_closedInside_: bool
                            d_6_closedCurrent_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_4_closedGenerated_ = out0_
                            d_5_closedInside_ = out1_
                            d_6_closedCurrent_ = out2_
                            generated = d_4_closedGenerated_
                            insideConstrainedOut = d_5_closedInside_
                            currentConstrainedOut = d_6_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_7_constrainedPrompt_: _dafny.Seq
                            d_7_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_8_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_7_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_8_next_ = out3_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_8_next_) == (eosToken):
                                d_9_rolledBackGenerated_: _dafny.Seq
                                d_10_rolledBackCurrent_: _dafny.Seq
                                out4_: _dafny.Seq
                                out5_: _dafny.Seq
                                out4_, out5_ = (d_0_helpers_).RollbackConstrainedSpan(parser, _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]), generated, currentConstrainedOut)
                                d_9_rolledBackGenerated_ = out4_
                                d_10_rolledBackCurrent_ = out5_
                                generated = d_9_rolledBackGenerated_
                                insideConstrainedOut = False
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                raise _dafny.Break("0")
                            elif True:
                                d_11_appendedGenerated_: _dafny.Seq
                                d_12_appendedInside_: bool
                                d_13_appendedCurrent_: _dafny.Seq
                                out6_: _dafny.Seq
                                out7_: bool
                                out8_: _dafny.Seq
                                out6_, out7_, out8_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_8_next_)
                                d_11_appendedGenerated_ = out6_
                                d_12_appendedInside_ = out7_
                                d_13_appendedCurrent_ = out8_
                                generated = d_11_appendedGenerated_
                                insideConstrainedOut = d_12_appendedInside_
                                currentConstrainedOut = d_13_appendedCurrent_
                    elif True:
                        if not(d_2_didWarmup_):
                            d_14_nextWarmup_: _dafny.Seq
                            out9_: _dafny.Seq
                            out9_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_14_nextWarmup_ = out9_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_2_didWarmup_ = True
                            if (d_14_nextWarmup_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_14_nextWarmup_]))
                        elif True:
                            if (d_1_steps_) < (maxSteps):
                                d_15_openedGenerated_: _dafny.Seq
                                d_16_openedInside_: bool
                                d_17_openedCurrent_: _dafny.Seq
                                out10_: _dafny.Seq
                                out11_: bool
                                out12_: _dafny.Seq
                                out10_, out11_, out12_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                d_15_openedGenerated_ = out10_
                                d_16_openedInside_ = out11_
                                d_17_openedCurrent_ = out12_
                                generated = d_15_openedGenerated_
                                insideConstrainedOut = d_16_openedInside_
                                currentConstrainedOut = d_17_openedCurrent_
                                d_1_steps_ = (d_1_steps_) + (1)
                    pass
            pass
        if insideConstrainedOut:
            d_18_completeAtEnd_: bool
            d_18_completeAtEnd_ = (parser).IsCompletePrefix(currentConstrainedOut)
            if (d_18_completeAtEnd_) and ((d_1_steps_) < (maxSteps)):
                d_19_finalGenerated_: _dafny.Seq
                d_20_finalInside_: bool
                d_21_finalCurrent_: _dafny.Seq
                out13_: _dafny.Seq
                out14_: bool
                out15_: _dafny.Seq
                out13_, out14_, out15_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_19_finalGenerated_ = out13_
                d_20_finalInside_ = out14_
                d_21_finalCurrent_ = out15_
                generated = d_19_finalGenerated_
                insideConstrainedOut = d_20_finalInside_
                currentConstrainedOut = d_21_finalCurrent_
                d_1_steps_ = (d_1_steps_) + (1)
            elif True:
                d_22_rbGenerated_: _dafny.Seq
                d_23_rbCurrent_: _dafny.Seq
                out16_: _dafny.Seq
                out17_: _dafny.Seq
                out16_, out17_ = (d_0_helpers_).RollbackConstrainedSpan(parser, _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]), generated, currentConstrainedOut)
                d_22_rbGenerated_ = out16_
                d_23_rbCurrent_ = out17_
                generated = d_22_rbGenerated_
                insideConstrainedOut = False
                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

