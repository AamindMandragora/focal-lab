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
        d_2_stop_: bool
        d_2_stop_ = False
        while ((d_1_steps_) < (maxSteps)) and (not(d_2_stop_)):
            if not(insideConstrainedOut):
                d_3_openedGenerated_: _dafny.Seq
                d_4_openedInside_: bool
                d_5_openedCurrent_: _dafny.Seq
                out0_: _dafny.Seq
                out1_: bool
                out2_: _dafny.Seq
                out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                d_3_openedGenerated_ = out0_
                d_4_openedInside_ = out1_
                d_5_openedCurrent_ = out2_
                generated = d_3_openedGenerated_
                insideConstrainedOut = d_4_openedInside_
                currentConstrainedOut = d_5_openedCurrent_
                d_1_steps_ = (d_1_steps_) + (1)
            elif True:
                d_6_completeNow_: bool
                d_6_completeNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                d_7_validCount_: int
                out3_: int
                out3_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                d_7_validCount_ = out3_
                if d_6_completeNow_:
                    if (d_7_validCount_) == (0):
                        d_2_stop_ = True
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_8_next_: _dafny.Seq
                        out4_: _dafny.Seq
                        out4_ = (d_0_helpers_).ConstrainedStep(lm, parser, prompt, currentConstrainedOut, eosToken)
                        d_8_next_ = out4_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_8_next_) == (eosToken):
                            d_2_stop_ = True
                elif True:
                    d_9_next_: _dafny.Seq
                    out5_: _dafny.Seq
                    out5_ = (d_0_helpers_).ConstrainedStep(lm, parser, prompt, currentConstrainedOut, eosToken)
                    d_9_next_ = out5_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if (d_9_next_) == (eosToken):
                        d_2_stop_ = True
                    elif True:
                        d_10_appendedGenerated_: _dafny.Seq
                        d_11_appendedInside_: bool
                        d_12_appendedCurrent_: _dafny.Seq
                        out6_: _dafny.Seq
                        out7_: bool
                        out8_: _dafny.Seq
                        out6_, out7_, out8_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_9_next_)
                        d_10_appendedGenerated_ = out6_
                        d_11_appendedInside_ = out7_
                        d_12_appendedCurrent_ = out8_
                        generated = d_10_appendedGenerated_
                        insideConstrainedOut = d_11_appendedInside_
                        currentConstrainedOut = d_12_appendedCurrent_
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_13_completeAtEnd_: bool
            d_13_completeAtEnd_ = (parser).IsCompletePrefix(currentConstrainedOut)
            if d_13_completeAtEnd_:
                d_14_closedGenerated_: _dafny.Seq
                d_15_closedInside_: bool
                d_16_closedCurrent_: _dafny.Seq
                out9_: _dafny.Seq
                out10_: bool
                out11_: _dafny.Seq
                out9_, out10_, out11_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_14_closedGenerated_ = out9_
                d_15_closedInside_ = out10_
                d_16_closedCurrent_ = out11_
                generated = d_14_closedGenerated_
                insideConstrainedOut = d_15_closedInside_
                currentConstrainedOut = d_16_closedCurrent_
                d_1_steps_ = (d_1_steps_) + (1)
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

