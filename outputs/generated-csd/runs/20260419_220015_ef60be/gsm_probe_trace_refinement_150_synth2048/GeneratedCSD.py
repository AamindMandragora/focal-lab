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
        if (maxSteps) > (0):
            if insideConstrainedOut:
                d_2_completeBefore_: bool
                d_2_completeBefore_ = (parser).IsCompletePrefix(currentConstrainedOut)
                if d_2_completeBefore_:
                    d_3_g0_: _dafny.Seq
                    d_4_i0_: bool
                    d_5_c0_: _dafny.Seq
                    out0_: _dafny.Seq
                    out1_: bool
                    out2_: _dafny.Seq
                    out0_, out1_, out2_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                    d_3_g0_ = out0_
                    d_4_i0_ = out1_
                    d_5_c0_ = out2_
                    generated = d_3_g0_
                    insideConstrainedOut = d_4_i0_
                    currentConstrainedOut = d_5_c0_
                elif True:
                    d_6_next_: _dafny.Seq
                    out3_: _dafny.Seq
                    out3_ = (d_0_helpers_).ConstrainedStep(lm, parser, prompt, currentConstrainedOut, eosToken)
                    d_6_next_ = out3_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if (d_6_next_) == (eosToken):
                        pass
                    elif True:
                        d_7_g1_: _dafny.Seq
                        d_8_i1_: bool
                        d_9_c1_: _dafny.Seq
                        out4_: _dafny.Seq
                        out5_: bool
                        out6_: _dafny.Seq
                        out4_, out5_, out6_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_6_next_)
                        d_7_g1_ = out4_
                        d_8_i1_ = out5_
                        d_9_c1_ = out6_
                        generated = d_7_g1_
                        insideConstrainedOut = d_8_i1_
                        currentConstrainedOut = d_9_c1_
            elif True:
                (lm).GenerateLogits((prompt) + (generated))
                (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]), _dafny.BigRational('8e0'))
                if (len(generated)) > (0):
                    d_10_last_: _dafny.Seq
                    d_10_last_ = (generated)[(len(generated)) - (1)]
                    if (((d_10_last_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ".")))) or ((d_10_last_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "!"))))) or ((d_10_last_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "?")))):
                        (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([eosToken]), _dafny.BigRational('8e0'))
                if (len(generated)) > (1):
                    d_11_last2_: _dafny.Seq
                    d_11_last2_ = (generated)[(len(generated)) - (2)]
                    d_12_last1_: _dafny.Seq
                    d_12_last1_ = (generated)[(len(generated)) - (1)]
                    if ((d_11_last2_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ".")))) and ((d_12_last1_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\n")))):
                        (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([eosToken]), _dafny.BigRational('8e0'))
                d_13_next_: _dafny.Seq
                out7_: _dafny.Seq
                out7_ = (lm).ChooseNextTokenUnconstrained()
                d_13_next_ = out7_
                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_13_next_]))
                d_1_steps_ = (d_1_steps_) + (1)
                if (d_13_next_) == (eosToken):
                    pass
                elif True:
                    if (d_13_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                        insideConstrainedOut = True
                        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

