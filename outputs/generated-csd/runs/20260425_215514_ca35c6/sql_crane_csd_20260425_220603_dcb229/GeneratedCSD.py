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
                        d_2_next0_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_2_next0_ = out0_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_2_next0_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_2_next0_]))
                            if (d_2_next0_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        d_3_complete_: bool
                        d_3_complete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_3_complete_:
                            d_4_gClose_: _dafny.Seq
                            d_5_inClose_: bool
                            d_6_cClose_: _dafny.Seq
                            out1_: _dafny.Seq
                            out2_: bool
                            out3_: _dafny.Seq
                            out1_, out2_, out3_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_4_gClose_ = out1_
                            d_5_inClose_ = out2_
                            d_6_cClose_ = out3_
                            generated = d_4_gClose_
                            insideConstrainedOut = d_5_inClose_
                            currentConstrainedOut = d_6_cClose_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_7_constrainedPrompt_: _dafny.Seq
                            d_7_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_8_next1_: _dafny.Seq
                            out4_: _dafny.Seq
                            out4_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_7_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_8_next1_ = out4_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_8_next1_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_9_gApp_: _dafny.Seq
                                d_10_inApp_: bool
                                d_11_cApp_: _dafny.Seq
                                out5_: _dafny.Seq
                                out6_: bool
                                out7_: _dafny.Seq
                                out5_, out6_, out7_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_8_next1_)
                                d_9_gApp_ = out5_
                                d_10_inApp_ = out6_
                                d_11_cApp_ = out7_
                                generated = d_9_gApp_
                                insideConstrainedOut = d_10_inApp_
                                currentConstrainedOut = d_11_cApp_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

