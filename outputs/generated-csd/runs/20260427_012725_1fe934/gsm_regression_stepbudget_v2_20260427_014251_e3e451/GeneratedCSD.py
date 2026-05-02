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
    def MyCSDStrategy(lm, parser, prompt, generatedPrefix, insideConstrained, currentConstrained, maxSteps, stepTokenBudget, eosToken):
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
        generated = generatedPrefix
        insideConstrainedOut = insideConstrained
        currentConstrainedOut = currentConstrained
        cost = 0
        d_1_steps_: int
        d_1_steps_ = 0
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if insideConstrainedOut:
                        d_2_complete_: bool
                        d_2_complete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_2_complete_:
                            d_3_gClose_: _dafny.Seq
                            d_4_iClose_: bool
                            d_5_cClose_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_3_gClose_ = out0_
                            d_4_iClose_ = out1_
                            d_5_cClose_ = out2_
                            generated = d_3_gClose_
                            insideConstrainedOut = d_4_iClose_
                            currentConstrainedOut = d_5_cClose_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_6_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).ConstrainedStep(lm, parser, prompt, currentConstrainedOut, eosToken)
                            d_6_next_ = out3_
                            if (d_6_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_7_gAppend_: _dafny.Seq
                                d_8_iAppend_: bool
                                d_9_cAppend_: _dafny.Seq
                                out4_: _dafny.Seq
                                out5_: bool
                                out6_: _dafny.Seq
                                out4_, out5_, out6_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_6_next_)
                                d_7_gAppend_ = out4_
                                d_8_iAppend_ = out5_
                                d_9_cAppend_ = out6_
                                generated = d_7_gAppend_
                                insideConstrainedOut = d_8_iAppend_
                                currentConstrainedOut = d_9_cAppend_
                                d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        if ((maxSteps) - (d_1_steps_)) >= (3):
                            d_10_gOpen_: _dafny.Seq
                            d_11_iOpen_: bool
                            d_12_cOpen_: _dafny.Seq
                            out7_: _dafny.Seq
                            out8_: bool
                            out9_: _dafny.Seq
                            out7_, out8_, out9_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_10_gOpen_ = out7_
                            d_11_iOpen_ = out8_
                            d_12_cOpen_ = out9_
                            generated = d_10_gOpen_
                            insideConstrainedOut = d_11_iOpen_
                            currentConstrainedOut = d_12_cOpen_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_13_next2_: _dafny.Seq
                            out10_: _dafny.Seq
                            out10_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_13_next2_ = out10_
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_13_next2_]))
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_13_next2_) == (eosToken):
                                raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

