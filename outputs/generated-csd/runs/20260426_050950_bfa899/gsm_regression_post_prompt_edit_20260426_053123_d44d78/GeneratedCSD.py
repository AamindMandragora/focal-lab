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
                    if insideConstrainedOut:
                        d_2_completeNow_: bool
                        d_2_completeNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_2_completeNow_:
                            d_3_g1_: _dafny.Seq
                            d_4_i1_: bool
                            d_5_c1_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_3_g1_ = out0_
                            d_4_i1_ = out1_
                            d_5_c1_ = out2_
                            generated = d_3_g1_
                            insideConstrainedOut = d_4_i1_
                            currentConstrainedOut = d_5_c1_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_6_validCount_: int
                            out3_: int
                            out3_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                            d_6_validCount_ = out3_
                            if (d_6_validCount_) == (0):
                                raise _dafny.Break("0")
                            elif True:
                                d_7_basePrompt_: _dafny.Seq
                                d_7_basePrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_8_nextC_: _dafny.Seq
                                out4_: _dafny.Seq
                                out4_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_7_basePrompt_, currentConstrainedOut, eosToken)
                                d_8_nextC_ = out4_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_8_nextC_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_9_ok_: bool
                                    out5_: bool
                                    out5_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_8_nextC_)
                                    d_9_ok_ = out5_
                                    if d_9_ok_:
                                        d_10_g2_: _dafny.Seq
                                        d_11_i2_: bool
                                        d_12_c2_: _dafny.Seq
                                        out6_: _dafny.Seq
                                        out7_: bool
                                        out8_: _dafny.Seq
                                        out6_, out7_, out8_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_8_nextC_)
                                        d_10_g2_ = out6_
                                        d_11_i2_ = out7_
                                        d_12_c2_ = out8_
                                        generated = d_10_g2_
                                        insideConstrainedOut = d_11_i2_
                                        currentConstrainedOut = d_12_c2_
                                    elif True:
                                        raise _dafny.Break("0")
                    elif True:
                        if ((len(generated)) > (0)) and (((generated)[(len(generated)) - (1)]) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")))):
                            d_13_g3_: _dafny.Seq
                            d_14_i3_: bool
                            d_15_c3_: _dafny.Seq
                            out9_: _dafny.Seq
                            out10_: bool
                            out11_: _dafny.Seq
                            out9_, out10_, out11_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_13_g3_ = out9_
                            d_14_i3_ = out10_
                            d_15_c3_ = out11_
                            generated = d_13_g3_
                            insideConstrainedOut = d_14_i3_
                            currentConstrainedOut = d_15_c3_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_16_nextU_: _dafny.Seq
                            out12_: _dafny.Seq
                            out12_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_16_nextU_ = out12_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_16_nextU_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_16_nextU_]))
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

