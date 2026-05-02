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
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if insideConstrainedOut:
                        d_2_isComplete_: bool
                        d_2_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_2_isComplete_:
                            d_3_closedGenerated_: _dafny.Seq
                            d_4_closedInside_: bool
                            d_5_closedCurrent_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_3_closedGenerated_ = out0_
                            d_4_closedInside_ = out1_
                            d_5_closedCurrent_ = out2_
                            generated = d_3_closedGenerated_
                            insideConstrainedOut = d_4_closedInside_
                            currentConstrainedOut = d_5_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_6_narrow_: bool
                            out3_: bool
                            out3_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                            d_6_narrow_ = out3_
                            if d_6_narrow_:
                                d_7_stablePrefix_: _dafny.Seq
                                d_7_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                d_8_rolledGenerated_: _dafny.Seq
                                d_9_rolledCurrent_: _dafny.Seq
                                out4_: _dafny.Seq
                                out5_: _dafny.Seq
                                out4_, out5_ = (d_0_helpers_).RollbackConstrainedSpan(parser, d_7_stablePrefix_, generated, currentConstrainedOut)
                                d_8_rolledGenerated_ = out4_
                                d_9_rolledCurrent_ = out5_
                                generated = d_8_rolledGenerated_
                                currentConstrainedOut = d_9_rolledCurrent_
                            elif True:
                                d_10_stablePrefix2_: _dafny.Seq
                                d_10_stablePrefix2_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                d_11_constrainedPrompt_: _dafny.Seq
                                d_11_constrainedPrompt_ = (prompt) + (d_10_stablePrefix2_)
                                d_12_next_: _dafny.Seq
                                out6_: _dafny.Seq
                                out6_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_11_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_12_next_ = out6_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_12_next_) == (eosToken):
                                    pass
                                elif True:
                                    d_13_appendedGenerated_: _dafny.Seq
                                    d_14_appendedInside_: bool
                                    d_15_appendedCurrent_: _dafny.Seq
                                    out7_: _dafny.Seq
                                    out8_: bool
                                    out9_: _dafny.Seq
                                    out7_, out8_, out9_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_12_next_)
                                    d_13_appendedGenerated_ = out7_
                                    d_14_appendedInside_ = out8_
                                    d_15_appendedCurrent_ = out9_
                                    generated = d_13_appendedGenerated_
                                    insideConstrainedOut = d_14_appendedInside_
                                    currentConstrainedOut = d_15_appendedCurrent_
                                if (d_12_next_) == (eosToken):
                                    d_1_steps_ = d_1_steps_
                                    raise _dafny.Break("0")
                    elif True:
                        d_16_openedGenerated_: _dafny.Seq
                        d_17_openedInside_: bool
                        d_18_openedCurrent_: _dafny.Seq
                        out10_: _dafny.Seq
                        out11_: bool
                        out12_: _dafny.Seq
                        out10_, out11_, out12_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                        d_16_openedGenerated_ = out10_
                        d_17_openedInside_ = out11_
                        d_18_openedCurrent_ = out12_
                        generated = d_16_openedGenerated_
                        insideConstrainedOut = d_17_openedInside_
                        currentConstrainedOut = d_18_openedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

