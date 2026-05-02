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
        d_2_done_: bool
        d_2_done_ = False
        d_3_usedSpan_: bool
        d_3_usedSpan_ = insideConstrained
        d_4_unconstrainedTokensBeforeOpen_: int
        d_4_unconstrainedTokensBeforeOpen_ = 0
        while ((d_1_steps_) < (maxSteps)) and (not(d_2_done_)):
            if insideConstrainedOut:
                d_5_completeNow_: bool
                d_5_completeNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                if d_5_completeNow_:
                    d_6_closedGenerated_: _dafny.Seq
                    d_7_closedInside_: bool
                    d_8_closedCurrent_: _dafny.Seq
                    out0_: _dafny.Seq
                    out1_: bool
                    out2_: _dafny.Seq
                    out0_, out1_, out2_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                    d_6_closedGenerated_ = out0_
                    d_7_closedInside_ = out1_
                    d_8_closedCurrent_ = out2_
                    generated = d_6_closedGenerated_
                    insideConstrainedOut = d_7_closedInside_
                    currentConstrainedOut = d_8_closedCurrent_
                    d_3_usedSpan_ = True
                    d_1_steps_ = (d_1_steps_) + (1)
                elif True:
                    if ((maxSteps) - (d_1_steps_)) <= (1):
                        d_2_done_ = True
                    elif True:
                        d_9_next_: _dafny.Seq
                        out3_: _dafny.Seq
                        out3_ = (d_0_helpers_).ConstrainedStep(lm, parser, prompt, currentConstrainedOut, eosToken)
                        d_9_next_ = out3_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_9_next_) == (eosToken):
                            d_2_done_ = True
                        elif True:
                            d_10_appendedGenerated_: _dafny.Seq
                            d_11_appendedInside_: bool
                            d_12_appendedCurrent_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_9_next_)
                            d_10_appendedGenerated_ = out4_
                            d_11_appendedInside_ = out5_
                            d_12_appendedCurrent_ = out6_
                            generated = d_10_appendedGenerated_
                            insideConstrainedOut = d_11_appendedInside_
                            currentConstrainedOut = d_12_appendedCurrent_
                            d_13_completeAfterAppend_: bool
                            d_13_completeAfterAppend_ = (parser).IsCompletePrefix(currentConstrainedOut)
                            if not(d_13_completeAfterAppend_):
                                d_2_done_ = True
            elif True:
                d_14_remaining_: int
                d_14_remaining_ = (maxSteps) - (d_1_steps_)
                if ((not(d_3_usedSpan_)) and ((d_4_unconstrainedTokensBeforeOpen_) == (0))) and ((d_14_remaining_) > (3)):
                    d_15_next0_: _dafny.Seq
                    out7_: _dafny.Seq
                    out7_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                    d_15_next0_ = out7_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if (d_15_next0_) == (eosToken):
                        d_2_done_ = True
                    elif True:
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_15_next0_]))
                        d_4_unconstrainedTokensBeforeOpen_ = (d_4_unconstrainedTokensBeforeOpen_) + (1)
                elif ((not(d_3_usedSpan_)) and ((d_4_unconstrainedTokensBeforeOpen_) > (0))) and ((d_14_remaining_) >= (3)):
                    d_16_openedGenerated_: _dafny.Seq
                    d_17_openedInside_: bool
                    d_18_openedCurrent_: _dafny.Seq
                    out8_: _dafny.Seq
                    out9_: bool
                    out10_: _dafny.Seq
                    out8_, out9_, out10_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                    d_16_openedGenerated_ = out8_
                    d_17_openedInside_ = out9_
                    d_18_openedCurrent_ = out10_
                    generated = d_16_openedGenerated_
                    insideConstrainedOut = d_17_openedInside_
                    currentConstrainedOut = d_18_openedCurrent_
                    d_3_usedSpan_ = True
                    d_1_steps_ = (d_1_steps_) + (1)
                elif True:
                    d_19_next2_: _dafny.Seq
                    out11_: _dafny.Seq
                    out11_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                    d_19_next2_ = out11_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if (d_19_next2_) == (eosToken):
                        d_2_done_ = True
                    elif True:
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_19_next2_]))
                        if not(d_3_usedSpan_):
                            d_4_unconstrainedTokensBeforeOpen_ = (d_4_unconstrainedTokensBeforeOpen_) + (1)
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

