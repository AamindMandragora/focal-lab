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
        d_2_producedSpan_: bool
        d_2_producedSpan_ = False
        if insideConstrained:
            d_2_producedSpan_ = True
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if insideConstrainedOut:
                        d_3_completeNow_: bool
                        d_3_completeNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_3_completeNow_:
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
                            d_2_producedSpan_ = True
                        elif True:
                            if ((d_1_steps_) + (2)) > (maxSteps):
                                raise _dafny.Break("0")
                            elif True:
                                d_7_nextc_: _dafny.Seq
                                out3_: _dafny.Seq
                                out3_ = (d_0_helpers_).ConstrainedStep(lm, parser, prompt, currentConstrainedOut, eosToken)
                                d_7_nextc_ = out3_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_7_nextc_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_8_appendedGenerated_: _dafny.Seq
                                    d_9_appendedInside_: bool
                                    d_10_appendedCurrent_: _dafny.Seq
                                    out4_: _dafny.Seq
                                    out5_: bool
                                    out6_: _dafny.Seq
                                    out4_, out5_, out6_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_7_nextc_)
                                    d_8_appendedGenerated_ = out4_
                                    d_9_appendedInside_ = out5_
                                    d_10_appendedCurrent_ = out6_
                                    generated = d_8_appendedGenerated_
                                    insideConstrainedOut = d_9_appendedInside_
                                    currentConstrainedOut = d_10_appendedCurrent_
                    elif True:
                        if not(d_2_producedSpan_):
                            if ((d_1_steps_) + (2)) > (maxSteps):
                                raise _dafny.Break("0")
                            elif True:
                                d_11_openedGenerated_: _dafny.Seq
                                d_12_openedInside_: bool
                                d_13_openedCurrent_: _dafny.Seq
                                out7_: _dafny.Seq
                                out8_: bool
                                out9_: _dafny.Seq
                                out7_, out8_, out9_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                d_11_openedGenerated_ = out7_
                                d_12_openedInside_ = out8_
                                d_13_openedCurrent_ = out9_
                                generated = d_11_openedGenerated_
                                insideConstrainedOut = d_12_openedInside_
                                currentConstrainedOut = d_13_openedCurrent_
                                d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_14_next_: _dafny.Seq
                            out10_: _dafny.Seq
                            out10_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_14_next_ = out10_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_14_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif (d_14_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                if ((d_1_steps_) + (1)) > (maxSteps):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_15_openedGenerated2_: _dafny.Seq
                                    d_16_openedInside2_: bool
                                    d_17_openedCurrent2_: _dafny.Seq
                                    out11_: _dafny.Seq
                                    out12_: bool
                                    out13_: _dafny.Seq
                                    out11_, out12_, out13_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                    d_15_openedGenerated2_ = out11_
                                    d_16_openedInside2_ = out12_
                                    d_17_openedCurrent2_ = out13_
                                    generated = d_15_openedGenerated2_
                                    insideConstrainedOut = d_16_openedInside2_
                                    currentConstrainedOut = d_17_openedCurrent2_
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_14_next_]))
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

