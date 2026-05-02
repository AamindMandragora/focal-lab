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
                    if not(insideConstrainedOut):
                        d_3_next_: _dafny.Seq
                        out1_: _dafny.Seq
                        out1_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_3_next_ = out1_
                        if (d_3_next_) == (eosToken):
                            d_1_steps_ = (d_1_steps_) + (1)
                            raise _dafny.Break("0")
                        elif (VerifiedDecoderAgent.default__.Contains(d_3_next_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))) and ((_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))) in ((lm).Tokens)):
                            d_4_openedGenerated_: _dafny.Seq
                            d_5_openedInside_: bool
                            d_6_openedCurrent_: _dafny.Seq
                            out2_: _dafny.Seq
                            out3_: bool
                            out4_: _dafny.Seq
                            out2_, out3_, out4_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_4_openedGenerated_ = out2_
                            d_5_openedInside_ = out3_
                            d_6_openedCurrent_ = out4_
                            generated = d_4_openedGenerated_
                            insideConstrainedOut = d_5_openedInside_
                            currentConstrainedOut = d_6_openedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_3_next_]))
                            d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        (lm).GenerateLogits((prompt) + (generated))
                        if (len(validTokenGroups)) > (0):
                            (d_0_helpers_).BoostValidGroups(lm, parser, currentConstrainedOut, validTokenGroups, _dafny.BigRational('3e0'))
                        elif (len(d_2_flatGroups_)) > (0):
                            (d_0_helpers_).BoostTokenLogits(lm, d_2_flatGroups_, _dafny.BigRational('3e0'))
                        d_7_nextConstrained_: _dafny.Seq
                        out5_: _dafny.Seq
                        out5_ = (d_0_helpers_).ConstrainedSample(lm, parser, currentConstrainedOut, eosToken)
                        d_7_nextConstrained_ = out5_
                        if (d_7_nextConstrained_) == (eosToken):
                            d_1_steps_ = (d_1_steps_) + (1)
                            raise _dafny.Break("0")
                        elif True:
                            d_8_appendedGenerated_: _dafny.Seq
                            d_9_appendedInside_: bool
                            d_10_appendedCurrent_: _dafny.Seq
                            out6_: _dafny.Seq
                            out7_: bool
                            out8_: _dafny.Seq
                            out6_, out7_, out8_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_7_nextConstrained_)
                            d_8_appendedGenerated_ = out6_
                            d_9_appendedInside_ = out7_
                            d_10_appendedCurrent_ = out8_
                            generated = d_8_appendedGenerated_
                            insideConstrainedOut = d_9_appendedInside_
                            currentConstrainedOut = d_10_appendedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

