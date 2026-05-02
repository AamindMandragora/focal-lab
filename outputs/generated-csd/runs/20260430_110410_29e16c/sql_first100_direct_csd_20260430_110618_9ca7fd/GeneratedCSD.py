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
        d_2_flatGroups_: _dafny.Seq
        out0_: _dafny.Seq
        out0_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
        d_2_flatGroups_ = out0_
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if insideConstrainedOut:
                        d_3_complete_: bool
                        d_3_complete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_3_complete_:
                            d_4_g2_: _dafny.Seq
                            d_5_in2_: bool
                            d_6_c2_: _dafny.Seq
                            out1_: _dafny.Seq
                            out2_: bool
                            out3_: _dafny.Seq
                            out1_, out2_, out3_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_4_g2_ = out1_
                            d_5_in2_ = out2_
                            d_6_c2_ = out3_
                            generated = d_4_g2_
                            insideConstrainedOut = d_5_in2_
                            currentConstrainedOut = d_6_c2_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            (lm).GenerateLogits((prompt) + (generated))
                            (d_0_helpers_).BoostValidGroups(lm, parser, currentConstrainedOut, validTokenGroups, _dafny.BigRational('3e0'))
                            (d_0_helpers_).PenalizeTokenLogits(lm, d_2_flatGroups_, _dafny.BigRational('5e-1'))
                            d_7_next_: _dafny.Seq
                            out4_: _dafny.Seq
                            out4_ = (d_0_helpers_).ConstrainedSample(lm, parser, currentConstrainedOut, eosToken)
                            d_7_next_ = out4_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_7_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_8_g3_: _dafny.Seq
                                d_9_in3_: bool
                                d_10_c3_: _dafny.Seq
                                out5_: _dafny.Seq
                                out6_: bool
                                out7_: _dafny.Seq
                                out5_, out6_, out7_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_7_next_)
                                d_8_g3_ = out5_
                                d_9_in3_ = out6_
                                d_10_c3_ = out7_
                                generated = d_8_g3_
                                insideConstrainedOut = d_9_in3_
                                currentConstrainedOut = d_10_c3_
                    elif True:
                        d_11_next2_: _dafny.Seq
                        out8_: _dafny.Seq
                        out8_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_11_next2_ = out8_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_11_next2_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            if (d_11_next2_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_11_next2_]))
                                if (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))) in ((lm).Tokens):
                                    insideConstrainedOut = True
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                elif True:
                                    insideConstrainedOut = False
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_11_next2_]))
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

