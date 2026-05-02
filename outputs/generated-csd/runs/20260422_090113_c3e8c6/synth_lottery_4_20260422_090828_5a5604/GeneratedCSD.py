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
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_2_nextOutside_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_2_nextOutside_ = out0_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_2_nextOutside_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_2_nextOutside_]))
                            if ((d_2_nextOutside_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))) or (VerifiedDecoderAgent.default__.Contains(d_2_nextOutside_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))):
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        d_3_complete_: bool
                        d_3_complete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_3_complete_:
                            d_4_closedGenerated_: _dafny.Seq
                            d_5_closedInside_: bool
                            d_6_closedCurrent_: _dafny.Seq
                            out1_: _dafny.Seq
                            out2_: bool
                            out3_: _dafny.Seq
                            out1_, out2_, out3_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_4_closedGenerated_ = out1_
                            d_5_closedInside_ = out2_
                            d_6_closedCurrent_ = out3_
                            generated = d_4_closedGenerated_
                            insideConstrainedOut = d_5_closedInside_
                            currentConstrainedOut = d_6_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            if ((d_1_steps_) + (1)) >= (maxSteps):
                                raise _dafny.Break("0")
                            elif True:
                                d_7_nextInside_: _dafny.Seq
                                out4_: _dafny.Seq
                                out4_ = (d_0_helpers_).ConstrainedStep(lm, parser, prompt, currentConstrainedOut, eosToken)
                                d_7_nextInside_ = out4_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_7_nextInside_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_8_appendedGenerated_: _dafny.Seq
                                    d_9_appendedInside_: bool
                                    d_10_appendedCurrent_: _dafny.Seq
                                    out5_: _dafny.Seq
                                    out6_: bool
                                    out7_: _dafny.Seq
                                    out5_, out6_, out7_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_7_nextInside_)
                                    d_8_appendedGenerated_ = out5_
                                    d_9_appendedInside_ = out6_
                                    d_10_appendedCurrent_ = out7_
                                    generated = d_8_appendedGenerated_
                                    insideConstrainedOut = d_9_appendedInside_
                                    currentConstrainedOut = d_10_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

