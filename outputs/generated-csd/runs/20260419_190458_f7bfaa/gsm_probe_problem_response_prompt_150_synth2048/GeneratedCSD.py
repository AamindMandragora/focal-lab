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
        if True:
            generated = generatedPrefix
            insideConstrainedOut = insideConstrained
            currentConstrainedOut = currentConstrained
            cost = 0
            d_1_steps_: int
            d_1_steps_ = 0
            if (0) < (maxSteps):
                if insideConstrainedOut:
                    d_2_completeNow_: bool
                    d_2_completeNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                    if d_2_completeNow_:
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: _dafny.Seq
                        out0_, out1_, out2_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        generated = out0_
                        insideConstrainedOut = out1_
                        currentConstrainedOut = out2_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_3_next_: _dafny.Seq
                        out3_: _dafny.Seq
                        out3_ = (d_0_helpers_).ConstrainedStep(lm, parser, prompt, currentConstrainedOut, eosToken)
                        d_3_next_ = out3_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_3_next_) == (eosToken):
                            pass
                        elif True:
                            d_4_stillIncomplete_: bool
                            d_4_stillIncomplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                            if not(d_4_stillIncomplete_):
                                out4_: _dafny.Seq
                                out5_: bool
                                out6_: _dafny.Seq
                                out4_, out5_, out6_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_3_next_)
                                generated = out4_
                                insideConstrainedOut = out5_
                                currentConstrainedOut = out6_
                elif True:
                    (lm).GenerateLogits((prompt) + (generated))
                    (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]), _dafny.BigRational('6e0'))
                    if (0) < (len(generated)):
                        if ((((generated)[(len(generated)) - (1)]) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ".")))) or (((generated)[(len(generated)) - (1)]) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "!"))))) or (((generated)[(len(generated)) - (1)]) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "?")))):
                            (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([eosToken]), _dafny.BigRational('8e0'))
                    if (1) < (len(generated)):
                        if ((((generated)[(len(generated)) - (2)]) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ".")))) or (((generated)[(len(generated)) - (2)]) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "!"))))) or (((generated)[(len(generated)) - (2)]) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "?")))):
                            (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([eosToken]), _dafny.BigRational('3e0'))
                    d_5_next_: _dafny.Seq
                    out7_: _dafny.Seq
                    out7_ = (lm).ChooseNextTokenUnconstrained()
                    d_5_next_ = out7_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if (d_5_next_) == (eosToken):
                        pass
                    elif True:
                        if (d_5_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                            out8_: _dafny.Seq
                            out9_: bool
                            out10_: _dafny.Seq
                            out8_, out9_, out10_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            generated = out8_
                            insideConstrainedOut = out9_
                            currentConstrainedOut = out10_
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_5_next_]))
            cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

