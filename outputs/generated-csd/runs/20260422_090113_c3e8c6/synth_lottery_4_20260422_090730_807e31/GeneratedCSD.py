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
        (d_0_helpers_).cost = 0
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_openedAny_: bool
        d_2_openedAny_ = insideConstrained
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if (not(d_2_openedAny_)) and (((d_1_steps_) + (1)) < (maxSteps)):
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
                            d_2_openedAny_ = True
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_6_nextOutside_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_6_nextOutside_ = out3_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_6_nextOutside_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_6_nextOutside_]))
                                if ((d_6_nextOutside_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))) or (VerifiedDecoderAgent.default__.Contains(d_6_nextOutside_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))):
                                    insideConstrainedOut = True
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                    d_2_openedAny_ = True
                    elif True:
                        d_7_complete_: bool
                        d_7_complete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_7_complete_:
                            d_8_closedGenerated_: _dafny.Seq
                            d_9_closedInside_: bool
                            d_10_closedCurrent_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_8_closedGenerated_ = out4_
                            d_9_closedInside_ = out5_
                            d_10_closedCurrent_ = out6_
                            generated = d_8_closedGenerated_
                            insideConstrainedOut = d_9_closedInside_
                            currentConstrainedOut = d_10_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            if ((d_1_steps_) + (1)) >= (maxSteps):
                                raise _dafny.Break("0")
                            elif True:
                                d_11_nextInside_: _dafny.Seq
                                out7_: _dafny.Seq
                                out7_ = (d_0_helpers_).ConstrainedStep(lm, parser, prompt, currentConstrainedOut, eosToken)
                                d_11_nextInside_ = out7_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_11_nextInside_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_12_appendedGenerated_: _dafny.Seq
                                    d_13_appendedInside_: bool
                                    d_14_appendedCurrent_: _dafny.Seq
                                    out8_: _dafny.Seq
                                    out9_: bool
                                    out10_: _dafny.Seq
                                    out8_, out9_, out10_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_11_nextInside_)
                                    d_12_appendedGenerated_ = out8_
                                    d_13_appendedInside_ = out9_
                                    d_14_appendedCurrent_ = out10_
                                    generated = d_12_appendedGenerated_
                                    insideConstrainedOut = d_13_appendedInside_
                                    currentConstrainedOut = d_14_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

