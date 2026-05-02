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
        d_1_steps_ = maxSteps
        while (0) < (d_1_steps_):
            if not(insideConstrainedOut):
                if ((len(generated)) > (len(generatedPrefix))) and ((2) <= (d_1_steps_)):
                    d_2_openedGenerated_: _dafny.Seq
                    d_3_openedInside_: bool
                    d_4_openedCurrent_: _dafny.Seq
                    out0_: _dafny.Seq
                    out1_: bool
                    out2_: _dafny.Seq
                    out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                    d_2_openedGenerated_ = out0_
                    d_3_openedInside_ = out1_
                    d_4_openedCurrent_ = out2_
                    generated = d_2_openedGenerated_
                    insideConstrainedOut = d_3_openedInside_
                    currentConstrainedOut = d_4_openedCurrent_
                    d_1_steps_ = (d_1_steps_) - (2)
                elif True:
                    d_5_next_: _dafny.Seq
                    out3_: _dafny.Seq
                    out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                    d_5_next_ = out3_
                    d_1_steps_ = (d_1_steps_) - (1)
                    if (d_5_next_) == (eosToken):
                        pass
                    elif True:
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_5_next_]))
            elif True:
                d_6_isComplete_: bool
                d_6_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                if d_6_isComplete_:
                    if (2) <= (d_1_steps_):
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
                        d_1_steps_ = (d_1_steps_) - (2)
                    elif True:
                        d_1_steps_ = 0
                elif True:
                    d_10_narrow_: bool
                    out7_: bool
                    out7_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                    d_10_narrow_ = out7_
                    if d_10_narrow_:
                        d_11_stablePrefix_: _dafny.Seq
                        d_11_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_12_repairedGenerated_: _dafny.Seq
                        d_13_repairedCurrent_: _dafny.Seq
                        out8_: _dafny.Seq
                        out9_: _dafny.Seq
                        out8_, out9_ = (d_0_helpers_).RollbackConstrainedSpan(parser, d_11_stablePrefix_, generated, currentConstrainedOut)
                        d_12_repairedGenerated_ = out8_
                        d_13_repairedCurrent_ = out9_
                        generated = d_12_repairedGenerated_
                        currentConstrainedOut = d_13_repairedCurrent_
                        d_14_repairedComplete_: bool
                        d_14_repairedComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_14_repairedComplete_:
                            insideConstrainedOut = False
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        d_15_constrainedPrompt_: _dafny.Seq
                        d_15_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_16_next_: _dafny.Seq
                        d_17_isValid_: bool
                        out10_: _dafny.Seq
                        out11_: bool
                        out10_, out11_ = (d_0_helpers_).SoftConstrainedStep(lm, parser, d_15_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('2e0'), eosToken)
                        d_16_next_ = out10_
                        d_17_isValid_ = out11_
                        d_1_steps_ = (d_1_steps_) - (1)
                        if (d_16_next_) == (eosToken):
                            pass
                        elif True:
                            if d_17_isValid_:
                                d_18_appendedGenerated_: _dafny.Seq
                                d_19_appendedInside_: bool
                                d_20_appendedCurrent_: _dafny.Seq
                                out12_: _dafny.Seq
                                out13_: bool
                                out14_: _dafny.Seq
                                out12_, out13_, out14_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_16_next_)
                                d_18_appendedGenerated_ = out12_
                                d_19_appendedInside_ = out13_
                                d_20_appendedCurrent_ = out14_
                                generated = d_18_appendedGenerated_
                                insideConstrainedOut = d_19_appendedInside_
                                currentConstrainedOut = d_20_appendedCurrent_
                            elif True:
                                d_21_stablePrefix2_: _dafny.Seq
                                d_21_stablePrefix2_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                d_22_repairedGenerated2_: _dafny.Seq
                                d_23_repairedCurrent2_: _dafny.Seq
                                out15_: _dafny.Seq
                                out16_: _dafny.Seq
                                out15_, out16_ = (d_0_helpers_).RollbackConstrainedSpan(parser, d_21_stablePrefix2_, generated, currentConstrainedOut)
                                d_22_repairedGenerated2_ = out15_
                                d_23_repairedCurrent2_ = out16_
                                generated = d_22_repairedGenerated2_
                                currentConstrainedOut = d_23_repairedCurrent2_
                                d_24_repairedComplete2_: bool
                                d_24_repairedComplete2_ = (parser).IsCompletePrefix(currentConstrainedOut)
                                if d_24_repairedComplete2_:
                                    insideConstrainedOut = False
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
        cost = d_0_helpers_.cost
        return generated, insideConstrainedOut, currentConstrainedOut, cost

