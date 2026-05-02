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
        while (d_1_steps_) < (maxSteps):
            if not(insideConstrainedOut):
                d_2_next_: _dafny.Seq
                out0_: _dafny.Seq
                out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                d_2_next_ = out0_
                if (d_2_next_) == (eosToken):
                    d_1_steps_ = maxSteps
                elif True:
                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_2_next_]))
                    d_1_steps_ = (d_1_steps_) + (1)
            elif True:
                d_3_isComplete_: bool
                d_3_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                if d_3_isComplete_:
                    d_4_closedGenerated_: _dafny.Seq
                    d_5_closedInside_: bool
                    d_6_closedCurrent_: _dafny.Seq
                    out1_: _dafny.Seq
                    out2_: bool
                    out3_: _dafny.Seq
                    out1_, out2_, out3_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                    d_4_closedGenerated_ = out1_
                    d_5_closedInside_ = out2_
                    d_6_closedCurrent_ = out3_
                    generated = d_4_closedGenerated_
                    insideConstrainedOut = d_5_closedInside_
                    currentConstrainedOut = d_6_closedCurrent_
                    d_1_steps_ = (d_1_steps_) + (1)
                elif True:
                    d_7_narrow_: bool
                    out4_: bool
                    out4_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                    d_7_narrow_ = out4_
                    if d_7_narrow_:
                        d_8_stablePrefix_: _dafny.Seq
                        d_8_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_9_repairedGenerated_: _dafny.Seq
                        d_10_repairedCurrent_: _dafny.Seq
                        out5_: _dafny.Seq
                        out6_: _dafny.Seq
                        out5_, out6_ = (d_0_helpers_).RollbackConstrainedSpan(parser, d_8_stablePrefix_, generated, currentConstrainedOut)
                        d_9_repairedGenerated_ = out5_
                        d_10_repairedCurrent_ = out6_
                        generated = d_9_repairedGenerated_
                        currentConstrainedOut = d_10_repairedCurrent_
                        insideConstrainedOut = True
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_11_next_: _dafny.Seq
                        d_12_isValid_: bool
                        out7_: _dafny.Seq
                        out8_: bool
                        out7_, out8_ = (d_0_helpers_).SoftConstrainedStep(lm, parser, prompt, currentConstrainedOut, _dafny.BigRational('2e0'), eosToken)
                        d_11_next_ = out7_
                        d_12_isValid_ = out8_
                        if (d_11_next_) == (eosToken):
                            d_1_steps_ = maxSteps
                        elif True:
                            if d_12_isValid_:
                                d_13_appendedGenerated_: _dafny.Seq
                                d_14_appendedInside_: bool
                                d_15_appendedCurrent_: _dafny.Seq
                                out9_: _dafny.Seq
                                out10_: bool
                                out11_: _dafny.Seq
                                out9_, out10_, out11_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_11_next_)
                                d_13_appendedGenerated_ = out9_
                                d_14_appendedInside_ = out10_
                                d_15_appendedCurrent_ = out11_
                                generated = d_13_appendedGenerated_
                                insideConstrainedOut = d_14_appendedInside_
                                currentConstrainedOut = d_15_appendedCurrent_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                d_16_stablePrefix2_: _dafny.Seq
                                d_16_stablePrefix2_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                d_17_repairedGenerated2_: _dafny.Seq
                                d_18_repairedCurrent2_: _dafny.Seq
                                out12_: _dafny.Seq
                                out13_: _dafny.Seq
                                out12_, out13_ = (d_0_helpers_).RollbackConstrainedSpan(parser, d_16_stablePrefix2_, generated, currentConstrainedOut)
                                d_17_repairedGenerated2_ = out12_
                                d_18_repairedCurrent2_ = out13_
                                generated = d_17_repairedGenerated2_
                                currentConstrainedOut = d_18_repairedCurrent2_
                                insideConstrainedOut = True
                                d_1_steps_ = (d_1_steps_) + (1)
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

