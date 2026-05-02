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
            if insideConstrainedOut:
                d_2_complete_: bool
                d_2_complete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                if d_2_complete_:
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
                    d_6_stablePrefix_: _dafny.Seq
                    d_6_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                    d_7_next_: _dafny.Seq
                    out3_: _dafny.Seq
                    out3_ = (d_0_helpers_).ConstrainedStep(lm, parser, (prompt) + (d_6_stablePrefix_), currentConstrainedOut, eosToken)
                    d_7_next_ = out3_
                    if (d_7_next_) == (eosToken):
                        d_8_completeAtEos_: bool
                        d_8_completeAtEos_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_8_completeAtEos_:
                            d_9_closedGenerated2_: _dafny.Seq
                            d_10_closedInside2_: bool
                            d_11_closedCurrent2_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_9_closedGenerated2_ = out4_
                            d_10_closedInside2_ = out5_
                            d_11_closedCurrent2_ = out6_
                            generated = d_9_closedGenerated2_
                            insideConstrainedOut = d_10_closedInside2_
                            currentConstrainedOut = d_11_closedCurrent2_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_12_stablePrefix2_: _dafny.Seq
                            d_12_stablePrefix2_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_13_rolledGenerated_: _dafny.Seq
                            d_14_rolledCurrent_: _dafny.Seq
                            out7_: _dafny.Seq
                            out8_: _dafny.Seq
                            out7_, out8_ = (d_0_helpers_).RollbackConstrainedSpan(parser, d_12_stablePrefix2_, generated, currentConstrainedOut)
                            d_13_rolledGenerated_ = out7_
                            d_14_rolledCurrent_ = out8_
                            generated = d_13_rolledGenerated_
                            insideConstrainedOut = False
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                            d_1_steps_ = maxSteps
                    elif True:
                        d_15_appendedGenerated_: _dafny.Seq
                        d_16_appendedInside_: bool
                        d_17_appendedCurrent_: _dafny.Seq
                        out9_: _dafny.Seq
                        out10_: bool
                        out11_: _dafny.Seq
                        out9_, out10_, out11_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_7_next_)
                        d_15_appendedGenerated_ = out9_
                        d_16_appendedInside_ = out10_
                        d_17_appendedCurrent_ = out11_
                        generated = d_15_appendedGenerated_
                        insideConstrainedOut = d_16_appendedInside_
                        currentConstrainedOut = d_17_appendedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
            elif True:
                d_18_next2_: _dafny.Seq
                out12_: _dafny.Seq
                out12_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                d_18_next2_ = out12_
                if (d_18_next2_) == (eosToken):
                    d_1_steps_ = maxSteps
                elif True:
                    d_19_hasOpen_: bool
                    d_19_hasOpen_ = VerifiedDecoderAgent.default__.Contains(d_18_next2_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))
                    if d_19_hasOpen_:
                        d_20_openedGenerated_: _dafny.Seq
                        d_21_openedInside_: bool
                        d_22_openedCurrent_: _dafny.Seq
                        out13_: _dafny.Seq
                        out14_: bool
                        out15_: _dafny.Seq
                        out13_, out14_, out15_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                        d_20_openedGenerated_ = out13_
                        d_21_openedInside_ = out14_
                        d_22_openedCurrent_ = out15_
                        generated = d_20_openedGenerated_
                        insideConstrainedOut = d_21_openedInside_
                        currentConstrainedOut = d_22_openedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_18_next2_]))
                        d_1_steps_ = (d_1_steps_) + (1)
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

