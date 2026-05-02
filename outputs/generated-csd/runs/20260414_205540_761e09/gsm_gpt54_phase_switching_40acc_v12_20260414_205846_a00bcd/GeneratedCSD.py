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
        d_2_emitProseOnce_: bool
        d_2_emitProseOnce_ = False
        if not(insideConstrainedOut):
            d_2_emitProseOnce_ = False
        elif True:
            d_2_emitProseOnce_ = False
        while (d_1_steps_) < (maxSteps):
            if insideConstrainedOut:
                d_3_complete_: bool
                d_3_complete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                if d_3_complete_:
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
                    d_2_emitProseOnce_ = True
                    d_1_steps_ = (d_1_steps_) + (1)
                elif True:
                    d_7_stablePrefix_: _dafny.Seq
                    d_7_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                    d_8_next_: _dafny.Seq
                    out3_: _dafny.Seq
                    out3_ = (d_0_helpers_).ConstrainedStep(lm, parser, (prompt) + (d_7_stablePrefix_), currentConstrainedOut, eosToken)
                    d_8_next_ = out3_
                    if (d_8_next_) == (eosToken):
                        d_1_steps_ = maxSteps
                    elif True:
                        d_9_appendedGenerated_: _dafny.Seq
                        d_10_appendedInside_: bool
                        d_11_appendedCurrent_: _dafny.Seq
                        out4_: _dafny.Seq
                        out5_: bool
                        out6_: _dafny.Seq
                        out4_, out5_, out6_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_8_next_)
                        d_9_appendedGenerated_ = out4_
                        d_10_appendedInside_ = out5_
                        d_11_appendedCurrent_ = out6_
                        generated = d_9_appendedGenerated_
                        insideConstrainedOut = d_10_appendedInside_
                        currentConstrainedOut = d_11_appendedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
            elif True:
                if d_2_emitProseOnce_:
                    d_12_next_: _dafny.Seq
                    out7_: _dafny.Seq
                    out7_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                    d_12_next_ = out7_
                    if (d_12_next_) == (eosToken):
                        d_1_steps_ = maxSteps
                    elif True:
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_12_next_]))
                        d_2_emitProseOnce_ = False
                        d_1_steps_ = (d_1_steps_) + (1)
                elif True:
                    d_13_openedGenerated_: _dafny.Seq
                    d_14_openedInside_: bool
                    d_15_openedCurrent_: _dafny.Seq
                    out8_: _dafny.Seq
                    out9_: bool
                    out10_: _dafny.Seq
                    out8_, out9_, out10_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                    d_13_openedGenerated_ = out8_
                    d_14_openedInside_ = out9_
                    d_15_openedCurrent_ = out10_
                    generated = d_13_openedGenerated_
                    insideConstrainedOut = d_14_openedInside_
                    currentConstrainedOut = d_15_openedCurrent_
                    d_2_emitProseOnce_ = False
                    d_1_steps_ = (d_1_steps_) + (1)
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

