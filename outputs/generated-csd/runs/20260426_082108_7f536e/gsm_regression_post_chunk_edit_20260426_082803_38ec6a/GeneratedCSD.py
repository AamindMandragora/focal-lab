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
                            d_3_closedGenerated_: _dafny.Seq
                            d_4_closedInside_: bool
                            d_5_closedCurrent_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_3_closedGenerated_ = out0_
                            d_4_closedInside_ = out1_
                            d_5_closedCurrent_ = out2_
                            generated = d_3_closedGenerated_
                            insideConstrainedOut = d_4_closedInside_
                            currentConstrainedOut = d_5_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_6_nextInside_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).ConstrainedStep(lm, parser, prompt, currentConstrainedOut, eosToken)
                            d_6_nextInside_ = out3_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_6_nextInside_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_7_appendedGenerated_: _dafny.Seq
                                d_8_appendedInside_: bool
                                d_9_appendedCurrent_: _dafny.Seq
                                out4_: _dafny.Seq
                                out5_: bool
                                out6_: _dafny.Seq
                                out4_, out5_, out6_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_6_nextInside_)
                                d_7_appendedGenerated_ = out4_
                                d_8_appendedInside_ = out5_
                                d_9_appendedCurrent_ = out6_
                                generated = d_7_appendedGenerated_
                                insideConstrainedOut = d_8_appendedInside_
                                currentConstrainedOut = d_9_appendedCurrent_
                    elif True:
                        d_10_remaining_: int
                        d_10_remaining_ = (maxSteps) - (d_1_steps_)
                        (lm).GenerateLogits((prompt) + (generated))
                        if (d_10_remaining_) <= (2):
                            (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]), _dafny.BigRational('8e0'))
                        elif True:
                            (d_0_helpers_).PenalizeTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]), _dafny.BigRational('8e0'))
                        d_11_next_: _dafny.Seq
                        out7_: _dafny.Seq
                        out7_ = (lm).ChooseNextToken()
                        d_11_next_ = out7_
                        (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_11_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif (d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                            if (d_10_remaining_) <= (2):
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_11_next_]))
                            elif True:
                                d_12_openedGenerated_: _dafny.Seq
                                d_13_openedInside_: bool
                                d_14_openedCurrent_: _dafny.Seq
                                out8_: _dafny.Seq
                                out9_: bool
                                out10_: _dafny.Seq
                                out8_, out9_, out10_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                d_12_openedGenerated_ = out8_
                                d_13_openedInside_ = out9_
                                d_14_openedCurrent_ = out10_
                                generated = d_12_openedGenerated_
                                insideConstrainedOut = d_13_openedInside_
                                currentConstrainedOut = d_14_openedCurrent_
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_11_next_]))
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

