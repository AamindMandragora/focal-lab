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
                    d_6_basePrompt_: _dafny.Seq
                    d_6_basePrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                    d_7_nextC_: _dafny.Seq
                    out3_: _dafny.Seq
                    out3_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_6_basePrompt_, currentConstrainedOut, eosToken)
                    d_7_nextC_ = out3_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if (d_7_nextC_) == (eosToken):
                        pass
                    elif True:
                        d_8_g2_: _dafny.Seq
                        d_9_i2_: bool
                        d_10_c2_: _dafny.Seq
                        out4_: _dafny.Seq
                        out5_: bool
                        out6_: _dafny.Seq
                        out4_, out5_, out6_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_7_nextC_)
                        d_8_g2_ = out4_
                        d_9_i2_ = out5_
                        d_10_c2_ = out6_
                        generated = d_8_g2_
                        insideConstrainedOut = d_9_i2_
                        currentConstrainedOut = d_10_c2_
            elif True:
                (lm).GenerateLogits((prompt) + (generated))
                (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]), _dafny.BigRational('1e2'))
                d_11_top_: _dafny.Seq
                out7_: _dafny.Seq
                out7_ = (d_0_helpers_).GetHighestLogitToken(lm)
                d_11_top_ = out7_
                d_12_shouldOpen_: bool
                d_12_shouldOpen_ = False
                if (len(generated)) == (len(generatedPrefix)):
                    d_12_shouldOpen_ = True
                elif (d_11_top_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                    d_12_shouldOpen_ = True
                if d_12_shouldOpen_:
                    d_13_g3_: _dafny.Seq
                    d_14_i3_: bool
                    d_15_c3_: _dafny.Seq
                    out8_: _dafny.Seq
                    out9_: bool
                    out10_: _dafny.Seq
                    out8_, out9_, out10_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                    d_13_g3_ = out8_
                    d_14_i3_ = out9_
                    d_15_c3_ = out10_
                    generated = d_13_g3_
                    insideConstrainedOut = d_14_i3_
                    currentConstrainedOut = d_15_c3_
                    d_1_steps_ = (d_1_steps_) + (1)
                elif True:
                    d_16_nextU_: _dafny.Seq
                    out11_: _dafny.Seq
                    out11_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                    d_16_nextU_ = out11_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if (d_16_nextU_) == (eosToken):
                        pass
                    elif True:
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_16_nextU_]))
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

