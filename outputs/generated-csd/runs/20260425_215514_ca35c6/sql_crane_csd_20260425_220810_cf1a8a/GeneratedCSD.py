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
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_2_gOpen_: _dafny.Seq
                        d_3_inOpen_: bool
                        d_4_cOpen_: _dafny.Seq
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: _dafny.Seq
                        out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                        d_2_gOpen_ = out0_
                        d_3_inOpen_ = out1_
                        d_4_cOpen_ = out2_
                        generated = d_2_gOpen_
                        insideConstrainedOut = d_3_inOpen_
                        currentConstrainedOut = d_4_cOpen_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_5_complete_: bool
                        d_5_complete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_5_complete_:
                            d_6_gClose_: _dafny.Seq
                            d_7_inClose_: bool
                            d_8_cClose_: _dafny.Seq
                            out3_: _dafny.Seq
                            out4_: bool
                            out5_: _dafny.Seq
                            out3_, out4_, out5_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_6_gClose_ = out3_
                            d_7_inClose_ = out4_
                            d_8_cClose_ = out5_
                            generated = d_6_gClose_
                            insideConstrainedOut = d_7_inClose_
                            currentConstrainedOut = d_8_cClose_
                            d_1_steps_ = (d_1_steps_) + (1)
                            raise _dafny.Break("0")
                        elif True:
                            d_9_constrainedPrompt_: _dafny.Seq
                            d_9_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_10_next_: _dafny.Seq
                            out6_: _dafny.Seq
                            out6_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_9_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_10_next_ = out6_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_10_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_11_gApp_: _dafny.Seq
                                d_12_inApp_: bool
                                d_13_cApp_: _dafny.Seq
                                out7_: _dafny.Seq
                                out8_: bool
                                out9_: _dafny.Seq
                                out7_, out8_, out9_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_10_next_)
                                d_11_gApp_ = out7_
                                d_12_inApp_ = out8_
                                d_13_cApp_ = out9_
                                generated = d_11_gApp_
                                insideConstrainedOut = d_12_inApp_
                                currentConstrainedOut = d_13_cApp_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

