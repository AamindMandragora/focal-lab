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
    def MyCSDStrategy(lm, parser, prompt, maxSteps, eosToken):
        generated: _dafny.Seq = _dafny.Seq({})
        remainingSteps: int = int(0)
        d_0_helpers_: VerifiedDecoderAgent.CSDHelpers
        nw0_ = VerifiedDecoderAgent.CSDHelpers()
        nw0_.ctor__(lm, parser)
        d_0_helpers_ = nw0_
        generated = _dafny.SeqWithoutIsStrInference([])
        d_1_stepsLeft_: int
        d_1_stepsLeft_ = maxSteps
        d_2_phase_: _dafny.Seq
        d_2_phase_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "reason"))
        d_3_inside__span_: bool
        d_3_inside__span_ = False
        d_4_closed__spans_: int
        d_4_closed__spans_ = 0
        d_5_reason__steps_: int
        d_5_reason__steps_ = 0
        d_6_wrap__steps_: int
        d_6_wrap__steps_ = 0
        d_7_nudge__steps_: int
        d_7_nudge__steps_ = 0
        with _dafny.label("0"):
            while (d_1_stepsLeft_) > (0):
                with _dafny.c_label("0"):
                    if d_3_inside__span_:
                        if ((d_0_helpers_).IsComplete(generated)) or ((d_0_helpers_).CanConstrain(generated)):
                            out0_: _dafny.Seq
                            out1_: int
                            out0_, out1_ = (d_0_helpers_).AppendConstrainedOrRightDelimiterStep(prompt, generated, d_1_stepsLeft_)
                            generated = out0_
                            d_1_stepsLeft_ = out1_
                            if (d_0_helpers_).EndsWithRightDelimiter(generated):
                                d_3_inside__span_ = False
                                d_4_closed__spans_ = (d_4_closed__spans_) + (1)
                                d_2_phase_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "done"))
                        elif True:
                            raise _dafny.Break("0")
                    elif (d_2_phase_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "done"))):
                        raise _dafny.Break("0")
                    elif (d_2_phase_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "reason"))):
                        if (d_5_reason__steps_) < (40):
                            out2_: _dafny.Seq
                            out3_: int
                            out2_, out3_ = (d_0_helpers_).AppendUnconstrainedStep(prompt, generated, d_1_stepsLeft_)
                            generated = out2_
                            d_1_stepsLeft_ = out3_
                            d_5_reason__steps_ = (d_5_reason__steps_) + (1)
                        elif True:
                            d_2_phase_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "wrap"))
                            out4_: _dafny.Seq
                            out5_: int
                            out4_, out5_ = (d_0_helpers_).AppendUnconstrainedStep(prompt, generated, d_1_stepsLeft_)
                            generated = out4_
                            d_1_stepsLeft_ = out5_
                            d_6_wrap__steps_ = 1
                    elif (d_2_phase_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "wrap"))):
                        if (d_6_wrap__steps_) < (3):
                            out6_: _dafny.Seq
                            out7_: int
                            out6_, out7_ = (d_0_helpers_).AppendUnconstrainedStep(prompt, generated, d_1_stepsLeft_)
                            generated = out6_
                            d_1_stepsLeft_ = out7_
                            d_6_wrap__steps_ = (d_6_wrap__steps_) + (1)
                        elif True:
                            d_2_phase_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "nudge"))
                            out8_: _dafny.Seq
                            out9_: int
                            out8_, out9_ = (d_0_helpers_).AppendUnconstrainedNudgeLeftDelimiterStep(prompt, generated, d_1_stepsLeft_)
                            generated = out8_
                            d_1_stepsLeft_ = out9_
                            d_7_nudge__steps_ = 1
                            if (d_0_helpers_).EndsWithLeftDelimiter(generated):
                                d_3_inside__span_ = True
                                d_2_phase_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "span"))
                    elif (d_2_phase_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "nudge"))):
                        out10_: _dafny.Seq
                        out11_: int
                        out10_, out11_ = (d_0_helpers_).AppendUnconstrainedNudgeLeftDelimiterStep(prompt, generated, d_1_stepsLeft_)
                        generated = out10_
                        d_1_stepsLeft_ = out11_
                        d_7_nudge__steps_ = (d_7_nudge__steps_) + (1)
                        if (d_0_helpers_).EndsWithLeftDelimiter(generated):
                            d_3_inside__span_ = True
                            d_2_phase_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "span"))
                    elif True:
                        out12_: _dafny.Seq
                        out13_: int
                        out12_, out13_ = (d_0_helpers_).AppendUnconstrainedNudgeLeftDelimiterStep(prompt, generated, d_1_stepsLeft_)
                        generated = out12_
                        d_1_stepsLeft_ = out13_
                        if (d_0_helpers_).EndsWithLeftDelimiter(generated):
                            d_3_inside__span_ = True
                            d_2_phase_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "span"))
                    pass
            pass
        remainingSteps = d_1_stepsLeft_
        return generated, remainingSteps

