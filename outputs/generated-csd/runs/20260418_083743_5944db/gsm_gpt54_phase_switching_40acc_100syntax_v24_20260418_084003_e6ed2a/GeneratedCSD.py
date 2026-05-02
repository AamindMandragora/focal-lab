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
        if (0) < (maxSteps):
            if not(insideConstrainedOut):
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
                d_1_steps_ = 1
            elif True:
                d_5_isComplete_: bool
                d_5_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                if d_5_isComplete_:
                    d_6_closedGenerated_: _dafny.Seq
                    d_7_closedInside_: bool
                    d_8_closedCurrent_: _dafny.Seq
                    out3_: _dafny.Seq
                    out4_: bool
                    out5_: _dafny.Seq
                    out3_, out4_, out5_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                    d_6_closedGenerated_ = out3_
                    d_7_closedInside_ = out4_
                    d_8_closedCurrent_ = out5_
                    generated = d_6_closedGenerated_
                    insideConstrainedOut = d_7_closedInside_
                    currentConstrainedOut = d_8_closedCurrent_
                    d_1_steps_ = 1
                elif True:
                    d_9_next_: _dafny.Seq
                    out6_: _dafny.Seq
                    out6_ = (d_0_helpers_).ConstrainedStep(lm, parser, prompt, currentConstrainedOut, eosToken)
                    d_9_next_ = out6_
                    d_1_steps_ = 1
                    if (d_9_next_) == (eosToken):
                        pass
                    elif True:
                        d_10_isValid_: bool
                        out7_: bool
                        out7_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_9_next_)
                        d_10_isValid_ = out7_
                        if d_10_isValid_:
                            d_11_appendedGenerated_: _dafny.Seq
                            d_12_appendedInside_: bool
                            d_13_appendedCurrent_: _dafny.Seq
                            out8_: _dafny.Seq
                            out9_: bool
                            out10_: _dafny.Seq
                            out8_, out9_, out10_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_9_next_)
                            d_11_appendedGenerated_ = out8_
                            d_12_appendedInside_ = out9_
                            d_13_appendedCurrent_ = out10_
                            generated = d_11_appendedGenerated_
                            insideConstrainedOut = d_12_appendedInside_
                            currentConstrainedOut = d_13_appendedCurrent_
                        elif True:
                            d_14_stablePrefix_: _dafny.Seq
                            d_14_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_15_rolledGenerated_: _dafny.Seq
                            d_16_rolledCurrent_: _dafny.Seq
                            out11_: _dafny.Seq
                            out12_: _dafny.Seq
                            out11_, out12_ = (d_0_helpers_).RollbackConstrainedSpan(parser, d_14_stablePrefix_, generated, currentConstrainedOut)
                            d_15_rolledGenerated_ = out11_
                            d_16_rolledCurrent_ = out12_
                            generated = d_15_rolledGenerated_
                            insideConstrainedOut = False
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

