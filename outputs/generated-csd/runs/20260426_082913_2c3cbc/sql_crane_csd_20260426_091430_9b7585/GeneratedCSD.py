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
        if True:
            generated = generatedPrefix
            insideConstrainedOut = insideConstrained
            currentConstrainedOut = currentConstrained
            cost = 0
            d_1_steps_: int
            d_1_steps_ = 0
            d_2_stopped_: bool
            d_2_stopped_ = False
            while (((d_1_steps_) + (1)) < (maxSteps)) and (not(d_2_stopped_)):
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
                    d_6_stableGenerated_: _dafny.Seq
                    d_6_stableGenerated_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                    d_7_next_: _dafny.Seq
                    out3_: _dafny.Seq
                    out3_ = (d_0_helpers_).ConstrainedStep(lm, parser, (prompt) + (d_6_stableGenerated_), currentConstrainedOut, eosToken)
                    d_7_next_ = out3_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if (d_7_next_) == (eosToken):
                        d_2_stopped_ = True
                    elif True:
                        d_8_completeNow_: bool
                        d_8_completeNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_8_completeNow_:
                            d_2_stopped_ = True
                        elif True:
                            d_9_appendedGenerated_: _dafny.Seq
                            d_10_appendedInside_: bool
                            d_11_appendedCurrent_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_7_next_)
                            d_9_appendedGenerated_ = out4_
                            d_10_appendedInside_ = out5_
                            d_11_appendedCurrent_ = out6_
                            generated = d_9_appendedGenerated_
                            insideConstrainedOut = d_10_appendedInside_
                            currentConstrainedOut = d_11_appendedCurrent_
            if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
                d_12_completeEnd_: bool
                d_12_completeEnd_ = (parser).IsCompletePrefix(currentConstrainedOut)
                if d_12_completeEnd_:
                    d_13_closedGenerated_: _dafny.Seq
                    d_14_closedInside_: bool
                    d_15_closedCurrent_: _dafny.Seq
                    out7_: _dafny.Seq
                    out8_: bool
                    out9_: _dafny.Seq
                    out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                    d_13_closedGenerated_ = out7_
                    d_14_closedInside_ = out8_
                    d_15_closedCurrent_ = out9_
                    generated = d_13_closedGenerated_
                    insideConstrainedOut = d_14_closedInside_
                    currentConstrainedOut = d_15_closedCurrent_
                    d_1_steps_ = (d_1_steps_) + (1)
            cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

