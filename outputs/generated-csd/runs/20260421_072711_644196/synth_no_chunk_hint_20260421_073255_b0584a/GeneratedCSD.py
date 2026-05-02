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
        if (maxSteps) == (0):
            cost = d_1_steps_
            return generated, insideConstrainedOut, currentConstrainedOut, cost
        if insideConstrainedOut:
            if (parser).IsCompletePrefix(currentConstrainedOut):
                d_2_generatedTmp_: _dafny.Seq
                d_3_insideTmp_: bool
                d_4_currentTmp_: _dafny.Seq
                out0_: _dafny.Seq
                out1_: bool
                out2_: _dafny.Seq
                out0_, out1_, out2_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_2_generatedTmp_ = out0_
                d_3_insideTmp_ = out1_
                d_4_currentTmp_ = out2_
                generated = d_2_generatedTmp_
                insideConstrainedOut = d_3_insideTmp_
                currentConstrainedOut = d_4_currentTmp_
                d_1_steps_ = (d_1_steps_) + (1)
                cost = d_1_steps_
                return generated, insideConstrainedOut, currentConstrainedOut, cost
            elif True:
                d_5_next_: _dafny.Seq
                out3_: _dafny.Seq
                out3_ = (d_0_helpers_).ConstrainedStep(lm, parser, prompt, currentConstrainedOut, eosToken)
                d_5_next_ = out3_
                d_1_steps_ = (d_1_steps_) + (1)
                if (d_5_next_) == (eosToken):
                    cost = d_1_steps_
                    return generated, insideConstrainedOut, currentConstrainedOut, cost
                elif True:
                    d_6_generatedTmp2_: _dafny.Seq
                    d_7_insideTmp2_: bool
                    d_8_currentTmp2_: _dafny.Seq
                    out4_: _dafny.Seq
                    out5_: bool
                    out6_: _dafny.Seq
                    out4_, out5_, out6_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_5_next_)
                    d_6_generatedTmp2_ = out4_
                    d_7_insideTmp2_ = out5_
                    d_8_currentTmp2_ = out6_
                    generated = d_6_generatedTmp2_
                    insideConstrainedOut = d_7_insideTmp2_
                    currentConstrainedOut = d_8_currentTmp2_
                    cost = d_1_steps_
                    return generated, insideConstrainedOut, currentConstrainedOut, cost
        elif True:
            d_9_next_: _dafny.Seq
            out7_: _dafny.Seq
            out7_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
            d_9_next_ = out7_
            d_1_steps_ = (d_1_steps_) + (1)
            if (d_9_next_) == (eosToken):
                cost = d_1_steps_
                return generated, insideConstrainedOut, currentConstrainedOut, cost
            elif (d_9_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_9_next_]))
                insideConstrainedOut = True
                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                cost = d_1_steps_
                return generated, insideConstrainedOut, currentConstrainedOut, cost
            elif True:
                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_9_next_]))
                insideConstrainedOut = False
                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                cost = d_1_steps_
                return generated, insideConstrainedOut, currentConstrainedOut, cost
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

