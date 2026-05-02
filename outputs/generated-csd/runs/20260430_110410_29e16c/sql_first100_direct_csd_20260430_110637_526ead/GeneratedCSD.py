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
    def MyCSDStrategy(lm, parser, prompt, generatedPrefix, insideConstrained, currentConstrained, maxSteps, stepTokenBudget, validTokenGroups, eosToken):
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
                            d_3_g2_: _dafny.Seq
                            d_4_in2_: bool
                            d_5_c2_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_3_g2_ = out0_
                            d_4_in2_ = out1_
                            d_5_c2_ = out2_
                            generated = d_3_g2_
                            insideConstrainedOut = d_4_in2_
                            currentConstrainedOut = d_5_c2_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            raise _dafny.Break("0")
                    elif True:
                        d_6_next_: _dafny.Seq
                        out3_: _dafny.Seq
                        out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_6_next_ = out3_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_6_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            if (d_6_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                if (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))) in ((lm).Tokens):
                                    d_7_g3_: _dafny.Seq
                                    d_8_in3_: bool
                                    d_9_c3_: _dafny.Seq
                                    out4_: _dafny.Seq
                                    out5_: bool
                                    out6_: _dafny.Seq
                                    out4_, out5_, out6_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                    d_7_g3_ = out4_
                                    d_8_in3_ = out5_
                                    d_9_c3_ = out6_
                                    generated = d_7_g3_
                                    insideConstrainedOut = d_8_in3_
                                    currentConstrainedOut = d_9_c3_
                                elif True:
                                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_6_next_]))
                                    insideConstrainedOut = False
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_6_next_]))
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

