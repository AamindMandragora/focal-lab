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
        d_2_openSpan_: _dafny.Seq
        d_2_openSpan_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))
        d_3_flatGroups_: _dafny.Seq
        out0_: _dafny.Seq
        out0_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
        d_3_flatGroups_ = out0_
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        (lm).GenerateLogits((prompt) + (generated))
                        (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([d_2_openSpan_]), _dafny.BigRational('1e2'))
                        if (len(d_3_flatGroups_)) > (0):
                            d_4_maybeIdx_: int
                            out1_: int
                            out1_ = VerifiedDecoderAgent.CSDHelpers.GroupContaining(validTokenGroups, d_2_openSpan_)
                            d_4_maybeIdx_ = out1_
                        d_5_nextOutside_: _dafny.Seq
                        out2_: _dafny.Seq
                        out2_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_5_nextOutside_ = out2_
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_5_nextOutside_]))
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_5_nextOutside_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            if VerifiedDecoderAgent.default__.Contains(d_5_nextOutside_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        d_6_complete_: bool
                        d_6_complete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_6_complete_:
                            d_7_closedGenerated_: _dafny.Seq
                            d_8_closedInside_: bool
                            d_9_closedCurrent_: _dafny.Seq
                            out3_: _dafny.Seq
                            out4_: bool
                            out5_: _dafny.Seq
                            out3_, out4_, out5_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_7_closedGenerated_ = out3_
                            d_8_closedInside_ = out4_
                            d_9_closedCurrent_ = out5_
                            generated = d_7_closedGenerated_
                            insideConstrainedOut = d_8_closedInside_
                            currentConstrainedOut = d_9_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            (lm).GenerateLogits((prompt) + (generated))
                            (lm).MaskValidNextAndEos(parser, currentConstrainedOut, eosToken)
                            d_10_next_: _dafny.Seq
                            out6_: _dafny.Seq
                            out6_ = (lm).ChooseNextToken()
                            d_10_next_ = out6_
                            (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_10_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_11_appendedGenerated_: _dafny.Seq
                                d_12_appendedInside_: bool
                                d_13_appendedCurrent_: _dafny.Seq
                                out7_: _dafny.Seq
                                out8_: bool
                                out9_: _dafny.Seq
                                out7_, out8_, out9_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_10_next_)
                                d_11_appendedGenerated_ = out7_
                                d_12_appendedInside_ = out8_
                                d_13_appendedCurrent_ = out9_
                                generated = d_11_appendedGenerated_
                                insideConstrainedOut = d_12_appendedInside_
                                currentConstrainedOut = d_13_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

